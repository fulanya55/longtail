"""One-shot frozen comparison of M5-periodicity and FullAlpha U2 on development-v8."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power, _predict
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_rawsaw
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features, _head_scores, _replace_isrj
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _block_permutation, _block_shuffle, _extract
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration, _load_signals, _metrics
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import ISRJ_INDEX, _clipped_logit, _load_expert, _standardize_scores
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import SAW_INDEX, _raw_scores

EXPECTED_V8_SHA256 = "985b6892e7c989f76dd08539b0555fe406232dcea5aaf8dfd2d76dec4f85c332"
TARGET_LABELS = ("MFT", "VFT", "ISRJ")
LABEL_INDEX = {"MFT": 3, "VFT": 4, "ISRJ": 5}
VARIANTS = ("B0_U0_C6RawSaw", "B1_U2_FullAlphaAligned", "C0_Q1_M5RawSaw", "C1_Q2_M5Aligned", "C2_Q3_M5Shuffled")


def _metric(result, key):
    metrics = result["metrics"]
    if key in metrics:
        return float(metrics[key])
    label, field = key.split("__", 1)
    return float(metrics["per_label"][label][field])


def _compare(rows, candidate, baseline, seed_offset):
    keys = (
        "macro_aupr", "macro_f1", "target_mean_weak_recall", "target_mean_strong_recall",
        "target_mean_negative_fpr", "target_weak_vs_negative_auroc", "ISRJ__aupr",
        "ISRJ__weak_recall", "ISRJ__weak_vs_negative_auroc", "SawFM__aupr", "SawFM__weak_recall",
    )
    per_seed = []
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        per_seed.append({"seed": row["seed"], **{key: _metric(variants[candidate], key)-_metric(variants[baseline], key) for key in keys}})
    aggregate = {key: _mean_ci([row[key] for row in per_seed], seed_offset+index) for index,key in enumerate(keys)}
    aggregate["positive_target_weak_seeds"] = int(sum(row["target_mean_weak_recall"] > 0 for row in per_seed))
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate}


def _apply_source(logits, source_root, variant):
    calibration_path = source_root / variant / "calibration.json"
    threshold_path = source_root / variant / "thresholds.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    thresholds = np.asarray(json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"], dtype=np.float32)
    probability = _apply_frozen_calibration(logits, calibration)
    prediction = probability >= thresholds[None, :]
    return probability, prediction, thresholds, _sha256(calibration_path), _sha256(threshold_path)


def _order_fpr(labels, prediction):
    order = labels.sum(axis=1)
    output = {}
    for label, index in LABEL_INDEX.items():
        output[label] = {}
        negative = labels[:, index] == 0
        for value in range(1, 6):
            mask = negative & (order == value)
            if mask.any():
                output[label][str(value)] = {"negative_count": int(mask.sum()), "fpr": float(prediction[mask, index].mean())}
    return output


def _evaluate(name, logits, labels, weak, receiver, source_root, source_variant, out):
    probability, prediction, thresholds, calibration_sha, threshold_sha = _apply_source(logits, source_root, source_variant)
    metrics = _metrics(labels, weak, probability, prediction)
    order_fpr = _order_fpr(labels, prediction)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "composition_order_fpr.json", order_fpr)
    np.savez_compressed(
        out / "predictions.npz", sample_ids=np.arange(len(labels), dtype=np.int64),
        receiver_noise_seed=receiver, targets=labels.astype(np.uint8), weak_mask=weak.astype(np.uint8),
        logits=logits.astype(np.float32), probabilities=probability,
        predictions=prediction.astype(np.uint8), thresholds=thresholds,
    )
    return {"variant": name, "metrics": metrics, "composition_order_fpr": order_fpr, "calibration_sha256": calibration_sha, "thresholds_sha256": threshold_sha}


def _run_seed(args, signals, labels, weak, receiver, seed, device):
    root = args.output_root / f"seed_{seed}"
    receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    full_root = args.fullalpha_root / f"seed_{seed}"
    m5_root = args.m5_root / f"seed_{seed}"
    full_summary = json.loads((full_root / "seed_summary.json").read_text(encoding="utf-8"))
    m5_summary = json.loads((m5_root / "seed_summary.json").read_text(encoding="utf-8"))
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    paths = {
        "c6": args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth",
        "fullalpha": args.c6_root / f"seed_{seed}" / "GradientContrastFixed1p5" / "best_model.pth",
        "m5": args.c6_root / f"seed_{seed}" / "M5_GradientIsolated" / "best_model.pth",
    }
    base = {}
    for name, path in paths.items():
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        model = _load_model_from_state(f2_dir, device, state)
        base[name] = _predict(model, signals, device, args.eval_batch_size)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    if _sha256(paths["c6"]) != full_summary["c6_checkpoint_sha256"] or _sha256(paths["fullalpha"]) != full_summary["fullalpha_checkpoint_sha256"]:
        raise ValueError("FullAlpha source checkpoint changed")
    if _sha256(paths["m5"]) != m5_summary["m5_checkpoint_sha256"]:
        raise ValueError("M5 source checkpoint changed")
    raw, raw_path = _load_rawsaw(args, seed, device)
    raw_score = _raw_scores(raw, signals, device, args.raw_batch_size)
    del raw
    for name in base:
        base[name] = np.asarray(base[name], dtype=np.float32).copy()
        base[name][:, SAW_INDEX] = raw_score
    aligned_model, aligned_mean, aligned_std, aligned_hashes = _load_expert(args.periodicity_root, seed, "H_AlignedHGB")
    shuffled_model, shuffled_mean, shuffled_std, shuffled_hashes = _load_expert(args.periodicity_root, seed, "S_BlockShuffledHGB")
    aligned_features = _extract(signals, device, args.feature_batch_size)
    permutation = _block_permutation(120, 20301000 + seed)[0]
    shuffled_signals = _block_shuffle(signals, permutation, args.block_size)
    shuffled_features = _extract(shuffled_signals, device, args.feature_batch_size)
    del shuffled_signals
    aligned_raw = _clipped_logit(aligned_model.predict_proba(((aligned_features-aligned_mean)/np.maximum(aligned_std,1e-5)).astype(np.float32))[:,1])
    shuffled_raw = _clipped_logit(shuffled_model.predict_proba(((shuffled_features-shuffled_mean)/np.maximum(shuffled_std,1e-5)).astype(np.float32))[:,1])
    full_stats_source = json.loads((args.fullalpha_score_stats_root / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))["score_stats"]
    full_aligned_z = _standardize_scores(aligned_raw, full_stats_source["aligned"])
    m5_aligned_z = _standardize_scores(aligned_raw, m5_summary["score_stats"]["aligned"])
    m5_shuffled_z = _standardize_scores(shuffled_raw, m5_summary["score_stats"]["block_shuffled"])

    def fused(source_logits, z_score, source_root, head_name):
        head_root = source_root / f"head_{head_name}"
        model = joblib.load(head_root / "model.joblib")
        norm = np.load(head_root / "normalization.npz")
        return _replace_isrj(source_logits, _head_scores(model, _fusion_features(source_logits, z_score), norm["mean"], norm["std"]))

    logits = {
        VARIANTS[0]: base["c6"],
        VARIANTS[1]: fused(base["fullalpha"], full_aligned_z, full_root, "aligned"),
        VARIANTS[2]: base["m5"],
        VARIANTS[3]: fused(base["m5"], m5_aligned_z, m5_root, "aligned"),
        VARIANTS[4]: fused(base["m5"], m5_shuffled_z, m5_root, "shuffled"),
    }
    sources = {
        VARIANTS[0]: (full_root, "U0_C6RawSaw"),
        VARIANTS[1]: (full_root, "U2_FullAlphaAlignedPeriodicity"),
        VARIANTS[2]: (m5_root, "Q1_M5RawSaw"),
        VARIANTS[3]: (m5_root, "Q2_M5AlignedPeriodicity"),
        VARIANTS[4]: (m5_root, "Q3_M5BlockShuffled"),
    }
    variants = [_evaluate(name, logits[name], labels, weak, receiver, sources[name][0], sources[name][1], root / name) for name in VARIANTS]
    row = {
        "seed": seed, "status": "completed", "development_v8_sha256": EXPECTED_V8_SHA256,
        "source_artifacts": {
            "c6_checkpoint_sha256": _sha256(paths["c6"]), "fullalpha_checkpoint_sha256": _sha256(paths["fullalpha"]),
            "m5_checkpoint_sha256": _sha256(paths["m5"]), "rawsaw_checkpoint_sha256": _sha256(raw_path),
            "periodicity_expert_sha256": {"aligned": aligned_hashes, "shuffled": shuffled_hashes},
        },
        "variants": variants, "external_refit": False, "locked_test_used": False, "test_v3_used": False, "test_v8_used": False,
    }
    _write_json(receipt, row)
    return row


def _mean_order(rows, variant):
    output = {}
    for label in TARGET_LABELS:
        output[label] = {}
        for order in range(1, 6):
            values = []
            for row in rows:
                item = {value["variant"]: value for value in row["variants"]}[variant]
                cell = item["composition_order_fpr"][label].get(str(order))
                if cell is not None: values.append(cell["fpr"])
            if values: output[label][str(order)] = _mean_ci(values, 20357000 + LABEL_INDEX[label]*100 + order)
    return output


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["sha256"].lower() != EXPECTED_V8_SHA256:
        raise ValueError("development-v8 audit failed")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    data_path = Path(audit["development_v8_path"])
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    with h5py.File(data_path, "r") as handle: receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [_run_seed(args, signals, labels, weak, receiver, seed, device) for seed in args.seeds]
    comparisons = {
        "Q2_vs_U0": _compare(rows, VARIANTS[3], VARIANTS[0], 20358000),
        "Q2_vs_U2": _compare(rows, VARIANTS[3], VARIANTS[1], 20358100),
        "Q2_vs_Q1": _compare(rows, VARIANTS[3], VARIANTS[2], 20358200),
        "Q2_vs_Q3": _compare(rows, VARIANTS[3], VARIANTS[4], 20358300),
    }
    order = {variant: _mean_order(rows, variant) for variant in (VARIANTS[0], VARIANTS[1], VARIANTS[3])}
    candidate_cells = [(label, value, cell["mean"]) for label, items in order[VARIANTS[3]].items() for value, cell in items.items()]
    delta_cells = []
    for label, value, candidate_fpr in candidate_cells:
        delta_cells.append((label, value, candidate_fpr - order[VARIANTS[0]][label][value]["mean"]))
    worst_cell = max(candidate_cells, key=lambda item:item[2])
    worst_delta = max(delta_cells, key=lambda item:item[2])
    q20 = comparisons["Q2_vs_U0"]["aggregate"]
    q2u2 = comparisons["Q2_vs_U2"]["aggregate"]
    q2q1 = comparisons["Q2_vs_Q1"]["aggregate"]
    q2q3 = comparisons["Q2_vs_Q3"]["aggregate"]
    candidate_results = [{item["variant"]:item for item in row["variants"]}[VARIANTS[3]] for row in rows]
    target_aupr_ok = all(np.mean([_metric(candidate, f"{label}__aupr")-_metric({item["variant"]:item for item in row["variants"]}[VARIANTS[0]], f"{label}__aupr") for row,candidate in zip(rows,candidate_results)]) >= -0.010 for label in TARGET_LABELS)
    candidate_fpr = float(np.mean([_metric(item, "target_mean_negative_fpr") for item in candidate_results]))
    mft_order5_reduction_vs_u2 = order[VARIANTS[3]]["MFT"]["5"]["mean"] - order[VARIANTS[1]]["MFT"]["5"]["mean"]
    checks = {
        "macro_aupr_gain_ge_0p005": q20["macro_aupr"]["mean"] >= 0.005,
        "macro_f1_gain_ge_0p010": q20["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_ge_0p030": q20["target_mean_weak_recall"]["mean"] >= 0.030,
        "target_weak_auroc_gain_ge_0p015": q20["target_weak_vs_negative_auroc"]["mean"] >= 0.015,
        "candidate_target_fpr_le_0p20": candidate_fpr <= 0.20,
        "target_aupr_noninferior": target_aupr_ok,
        "target_strong_recall_noninferior": q20["target_mean_strong_recall"]["mean"] >= -0.010,
        "four_of_five_positive_target_weak": q20["positive_target_weak_seeds"] >= 4,
        "sawfm_exact": abs(q20["SawFM__aupr"]["mean"]) < 1e-12 and abs(q20["SawFM__weak_recall"]["mean"]) < 1e-12,
        "worst_order_fpr_le_0p30": worst_cell[2] <= 0.30,
        "worst_order_fpr_delta_le_0p05": worst_delta[2] <= 0.05,
        "mft_order5_fpr_reduction_vs_u2_ge_0p05": mft_order5_reduction_vs_u2 <= -0.05,
        "macro_aupr_within_0p005_of_u2": q2u2["macro_aupr"]["mean"] >= -0.005,
        "target_weak_recall_within_0p015_of_u2": q2u2["target_mean_weak_recall"]["mean"] >= -0.015,
        "isrj_weak_auroc_vs_shuffle_ge_0p005": q2q3["ISRJ__weak_vs_negative_auroc"]["mean"] >= 0.005,
        "isrj_weak_recall_vs_q1_ge_0p010": q2q1["ISRJ__weak_recall"]["mean"] >= 0.010,
        "isrj_weak_auroc_vs_q1_ge_0p005": q2q1["ISRJ__weak_vs_negative_auroc"]["mean"] >= 0.005,
        "external_refit_false": all(not row["external_refit"] for row in rows),
        "test_v3_boundary_safe": all(not row["test_v3_used"] for row in rows),
    }
    result = {
        "status": "completed", "experiment": "RouteA_M5_Periodicity_DevelopmentV8_Frozen",
        "evidence_role": "fresh_identity_isolated_development_v8_frozen_confirmation",
        "development_v8_sha256": EXPECTED_V8_SHA256, "rows": rows,
        "comparisons": comparisons, "composition_order_fpr": order,
        "worst_candidate_order_fpr": {"label":worst_cell[0],"order":worst_cell[1],"fpr":worst_cell[2]},
        "worst_candidate_order_fpr_delta_vs_u0": {"label":worst_delta[0],"order":worst_delta[1],"delta":worst_delta[2]},
        "mft_order5_fpr_delta_vs_u2": mft_order5_reduction_vs_u2,
        "candidate_target_mean_fpr": candidate_fpr,
        "confirmation_gate": {"passed": bool(all(checks.values())), "checks": {key:bool(value) for key,value in checks.items()}},
        "external_refit": False, "locked_test_used": False, "test_v3_used": False, "test_v8_used": False,
    }
    output = args.output_root / "route_a_m5_periodicity_development_v8_frozen_summary.json"
    _write_json(output, result)
    print(json.dumps({"status":"completed","gate":result["confirmation_gate"],"worst":result["worst_candidate_order_fpr"],"summary":str(output)},ensure_ascii=False,indent=2))
    return result


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v8_generation/identity_audit.json")
    parser.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--fullalpha-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--fullalpha-score-stats-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--m5-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_m5_periodicity_unified_v3")
    parser.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_m5_periodicity_development_v8_frozen")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device",default=None)
    parser.add_argument("--io-batch-size",type=int,default=128)
    parser.add_argument("--eval-batch-size",type=int,default=128)
    parser.add_argument("--raw-batch-size",type=int,default=64)
    parser.add_argument("--feature-batch-size",type=int,default=64)
    parser.add_argument("--raw-hidden-channels",type=int,default=16)
    parser.add_argument("--raw-topk",type=int,default=16)
    parser.add_argument("--raw-dropout",type=float,default=0.1)
    parser.add_argument("--block-size",type=int,default=25)
    parser.add_argument("--weak-threshold-db",type=float,default=10.0)
    parser.add_argument("--resume",action="store_true")
    return parser.parse_args()


if __name__=="__main__": run(parse_args())
