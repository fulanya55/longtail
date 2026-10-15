"""One-shot frozen confirmation of the unified U2 candidate on development-v7."""
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
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import TARGET_LABELS, VARIANTS
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features, _head_scores, _replace_isrj
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _block_permutation, _block_shuffle, _extract
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration, _load_signals, _metrics
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import ISRJ_INDEX, _clipped_logit, _load_expert, _standardize_scores
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import SAW_INDEX, _raw_scores

EXPECTED_V7_SHA256 = "cb7bc5014fba526c40b61b0ba34d358bceb322b075e8dff867f8f2c4cf08d88d"
SEEDS = (2039, 2040, 2041, 2042, 2043)


def _metric(result, key):
    metrics = result["metrics"]
    if key in metrics:
        return float(metrics[key])
    label, field = key.split("__", 1)
    return float(metrics["per_label"][label][field])


def _compare(rows, candidate, baseline, seed_offset):
    keys = (
        "macro_aupr", "macro_f1", "target_mean_weak_recall",
        "target_mean_strong_recall", "target_mean_negative_fpr",
        "target_weak_vs_negative_auroc", "ISRJ__aupr", "ISRJ__weak_recall",
        "ISRJ__strong_recall", "ISRJ__negative_fpr",
        "ISRJ__weak_vs_negative_auroc", "SawFM__aupr", "SawFM__weak_recall",
    )
    per_seed = []
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        per_seed.append({
            "seed": row["seed"],
            **{key: _metric(variants[candidate], key) - _metric(variants[baseline], key) for key in keys},
        })
    aggregate = {
        key: _mean_ci([row[key] for row in per_seed], seed_offset + index)
        for index, key in enumerate(keys)
    }
    aggregate["positive_target_weak_seeds"] = int(
        sum(row["target_mean_weak_recall"] > 0 for row in per_seed)
    )
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate}


def _promotion_checks(rows, comparisons, external_refit=False, test_v3_used=False):
    u20 = comparisons["U2_vs_U0"]["aggregate"]
    u21 = comparisons["U2_vs_U1"]["aggregate"]
    u23 = comparisons["U2_vs_U3"]["aggregate"]
    candidate_results = [
        {item["variant"]: item for item in row["variants"]}[VARIANTS[2]] for row in rows
    ]
    target_aupr_ok = all(
        np.mean([
            _metric(candidate, f"{label}__aupr")
            - _metric({item["variant"]: item for item in row["variants"]}[VARIANTS[0]], f"{label}__aupr")
            for row, candidate in zip(rows, candidate_results)
        ]) >= -0.010
        for label in TARGET_LABELS
    )
    candidate_target_fpr = float(np.mean([
        _metric(candidate, "target_mean_negative_fpr") for candidate in candidate_results
    ]))
    checks = {
        "macro_aupr_gain_ge_0p005": u20["macro_aupr"]["mean"] >= 0.005,
        "macro_f1_gain_ge_0p010": u20["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_ge_0p030": u20["target_mean_weak_recall"]["mean"] >= 0.030,
        "target_weak_auroc_gain_ge_0p015": u20["target_weak_vs_negative_auroc"]["mean"] >= 0.015,
        "candidate_target_fpr_le_0p20": candidate_target_fpr <= 0.20,
        "target_aupr_noninferior": bool(target_aupr_ok),
        "target_strong_recall_noninferior": u20["target_mean_strong_recall"]["mean"] >= -0.010,
        "four_of_five_positive_target_weak": u20["positive_target_weak_seeds"] >= 4,
        "isrj_weak_auroc_vs_u1_ge_0p005": u21["ISRJ__weak_vs_negative_auroc"]["mean"] >= 0.005,
        "isrj_weak_recall_vs_u1_ge_0p010": u21["ISRJ__weak_recall"]["mean"] >= 0.010,
        "isrj_weak_auroc_vs_shuffle_ge_0p005": u23["ISRJ__weak_vs_negative_auroc"]["mean"] >= 0.005,
        "sawfm_exact": abs(u20["SawFM__aupr"]["mean"]) < 1e-12 and abs(u20["SawFM__weak_recall"]["mean"]) < 1e-12,
        "external_refit_false": not external_refit,
        "test_v3_boundary_safe": not test_v3_used,
    }
    return candidate_target_fpr, {key: bool(value) for key, value in checks.items()}


def _source_variant(root, variant):
    calibration_path = root / variant / "calibration.json"
    thresholds_path = root / variant / "thresholds.json"
    return (
        json.loads(calibration_path.read_text(encoding="utf-8")),
        json.loads(thresholds_path.read_text(encoding="utf-8")),
        _sha256(calibration_path),
        _sha256(thresholds_path),
    )


def _evaluate_variant(name, logits, labels, weak, receiver, source_root, output_root):
    calibration, threshold_payload, calibration_sha, thresholds_sha = _source_variant(source_root, name)
    probabilities = _apply_frozen_calibration(logits, calibration)
    thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float32)
    predictions = probabilities >= thresholds[None, :]
    metrics = _metrics(labels, weak, probabilities, predictions)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "metrics.json", metrics)
    np.savez_compressed(
        output_root / "predictions.npz",
        sample_ids=np.arange(len(labels), dtype=np.int64),
        receiver_noise_seed=receiver,
        targets=labels.astype(np.uint8),
        weak_mask=weak.astype(np.uint8),
        logits=np.asarray(logits, dtype=np.float32),
        probabilities=probabilities,
        predictions=predictions.astype(np.uint8),
        thresholds=thresholds,
    )
    return {
        "variant": name,
        "metrics": metrics,
        "calibration_sha256": calibration_sha,
        "thresholds_sha256": thresholds_sha,
    }


def _verify_standardization_provenance(unified_summary, old_summary):
    if unified_summary["matched_receipt"] != old_summary["fit_matched_receipt"]:
        raise ValueError("matched-view receipt differs between the source experiments")
    unified_hashes = unified_summary["periodicity_expert_sha256"]
    old_hashes = old_summary["periodicity_expert_sha256"]
    if unified_hashes["aligned"] != old_hashes["aligned"]:
        raise ValueError("aligned periodicity expert changed")
    if unified_hashes["shuffled"] != old_hashes["block_shuffled"]:
        raise ValueError("shuffled periodicity expert changed")
    return {
        "source": "route_a_isrj_frozen_score_fusion_head_v3",
        "reason": "same expert, fit role, matched selection, and matched receipt as unified-v3",
        "aligned": old_summary["score_stats"]["aligned"],
        "block_shuffled": old_summary["score_stats"]["block_shuffled"],
        "equivalence_verified": True,
    }


def _run_seed(args, signals, labels, weak, receiver, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    receipt_path = root / "seed_summary.json"
    if args.resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root / f"seed_{seed}"
    unified_summary = json.loads((source_root / "seed_summary.json").read_text(encoding="utf-8"))
    old_summary = json.loads((args.score_stats_root / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))
    score_provenance = _verify_standardization_provenance(unified_summary, old_summary)

    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    base_logits = {}
    checkpoint_paths = {}
    for source_name, output_name in (("C6_Joint", "c6"), ("GradientContrastFixed1p5", "fullalpha")):
        checkpoint_path = args.c6_root / f"seed_{seed}" / source_name / "best_model.pth"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
        model = _load_model_from_state(f2_dir, device, state)
        base_logits[output_name] = _predict(model, signals, device, args.eval_batch_size)
        del model
        checkpoint_paths[output_name] = checkpoint_path
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if _sha256(checkpoint_paths["c6"]) != unified_summary["c6_checkpoint_sha256"]:
        raise ValueError("C6 checkpoint hash changed")
    if _sha256(checkpoint_paths["fullalpha"]) != unified_summary["fullalpha_checkpoint_sha256"]:
        raise ValueError("FullAlpha checkpoint hash changed")

    raw_model, raw_path = _load_rawsaw(args, seed, device)
    raw_score = _raw_scores(raw_model, signals, device, args.raw_batch_size)
    del raw_model
    if _sha256(raw_path) != unified_summary["rawsaw_checkpoint_sha256"]:
        raise ValueError("RawSaw checkpoint hash changed")
    c6 = np.asarray(base_logits["c6"], dtype=np.float32).copy()
    full = np.asarray(base_logits["fullalpha"], dtype=np.float32).copy()
    c6[:, SAW_INDEX] = raw_score
    full[:, SAW_INDEX] = raw_score

    aligned_model, aligned_mean, aligned_std, aligned_hashes = _load_expert(
        args.periodicity_root, seed, "H_AlignedHGB"
    )
    shuffled_model, shuffled_mean, shuffled_std, shuffled_hashes = _load_expert(
        args.periodicity_root, seed, "S_BlockShuffledHGB"
    )
    if aligned_hashes != unified_summary["periodicity_expert_sha256"]["aligned"]:
        raise ValueError("aligned periodicity artifact hash changed")
    if shuffled_hashes != unified_summary["periodicity_expert_sha256"]["shuffled"]:
        raise ValueError("shuffled periodicity artifact hash changed")

    aligned_features = _extract(signals, device, args.feature_batch_size)
    permutation = _block_permutation(120, 20301000 + seed)[0]
    shuffled_signals = _block_shuffle(signals, permutation, args.block_size)
    shuffled_features = _extract(shuffled_signals, device, args.feature_batch_size)
    del shuffled_signals
    aligned_raw = _clipped_logit(aligned_model.predict_proba(
        ((aligned_features - aligned_mean) / np.maximum(aligned_std, 1e-5)).astype(np.float32)
    )[:, 1])
    shuffled_raw = _clipped_logit(shuffled_model.predict_proba(
        ((shuffled_features - shuffled_mean) / np.maximum(shuffled_std, 1e-5)).astype(np.float32)
    )[:, 1])
    aligned_z = _standardize_scores(aligned_raw, score_provenance["aligned"])
    shuffled_z = _standardize_scores(shuffled_raw, score_provenance["block_shuffled"])

    fused_logits = {}
    head_hashes = {}
    for name, z_score in (("aligned", aligned_z), ("shuffled", shuffled_z)):
        head_root = source_root / f"head_{name}"
        head = joblib.load(head_root / "model.joblib")
        normalization = np.load(head_root / "normalization.npz")
        score = _head_scores(
            head,
            _fusion_features(full, z_score),
            normalization["mean"],
            normalization["std"],
        )
        fused_logits[name] = _replace_isrj(full, score)
        head_hashes[name] = {
            "model_sha256": _sha256(head_root / "model.joblib"),
            "normalization_sha256": _sha256(head_root / "normalization.npz"),
        }

    logits = {
        VARIANTS[0]: c6,
        VARIANTS[1]: full,
        VARIANTS[2]: fused_logits["aligned"],
        VARIANTS[3]: fused_logits["shuffled"],
    }
    non_isrj_indices = [index for index in range(labels.shape[1]) if index != ISRJ_INDEX]
    non_isrj_error = float(np.max(np.abs(logits[VARIANTS[2]][:, non_isrj_indices] - full[:, non_isrj_indices])))
    saw_error = float(np.max(np.abs(logits[VARIANTS[2]][:, SAW_INDEX] - c6[:, SAW_INDEX])))
    if non_isrj_error != 0.0 or saw_error != 0.0:
        raise AssertionError("frozen routing invariants failed")

    variants = [
        _evaluate_variant(name, logits[name], labels, weak, receiver, source_root, root / name)
        for name in VARIANTS
    ]
    row = {
        "seed": seed,
        "status": "completed",
        "development_v7_sha256": data_sha,
        "source_development_v3_sha256": unified_summary["data_sha256"],
        "source_artifacts": {
            "c6_checkpoint_sha256": _sha256(checkpoint_paths["c6"]),
            "fullalpha_checkpoint_sha256": _sha256(checkpoint_paths["fullalpha"]),
            "rawsaw_checkpoint_sha256": _sha256(raw_path),
            "periodicity_expert_sha256": {"aligned": aligned_hashes, "shuffled": shuffled_hashes},
            "fusion_head_sha256": head_hashes,
        },
        "score_standardization_provenance": score_provenance,
        "non_isrj_u2_u1_max_abs_error": non_isrj_error,
        "sawfm_u2_u0_max_abs_error": saw_error,
        "variants": variants,
        "external_refit": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v7_used": False,
    }
    _write_json(receipt_path, row)
    return row


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["sha256"].lower() != EXPECTED_V7_SHA256:
        raise ValueError("development-v7 audit failed")
    if audit["test_files_opened"] or audit["test_v3_used"] or audit["test_v7_used"]:
        raise ValueError("development-v7 evidence boundary failed")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    data_path = Path(args.data_path) if args.data_path is not None else Path(audit["development_v7_path"])
    if _sha256(data_path).lower() != EXPECTED_V7_SHA256:
        raise ValueError("development-v7 data hash changed")

    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_seed(args, signals, labels, weak, receiver, seed, device, EXPECTED_V7_SHA256)
        for seed in args.seeds
    ]
    comparisons = {
        "U2_vs_U0": _compare(rows, VARIANTS[2], VARIANTS[0], 20348000),
        "U2_vs_U1": _compare(rows, VARIANTS[2], VARIANTS[1], 20348100),
        "U2_vs_U3": _compare(rows, VARIANTS[2], VARIANTS[3], 20348200),
    }
    candidate_target_fpr, checks = _promotion_checks(
        rows, comparisons, external_refit=False, test_v3_used=False
    )
    result = {
        "status": "completed",
        "experiment": "RouteA_FullAlpha_Periodicity_DevelopmentV7_Frozen",
        "evidence_role": "fresh_identity_isolated_development_v7_frozen_confirmation",
        "seeds": list(args.seeds),
        "development_v7_sha256": EXPECTED_V7_SHA256,
        "rows": rows,
        "comparisons": comparisons,
        "candidate_target_mean_fpr": candidate_target_fpr,
        "confirmation_gate": {"passed": bool(all(checks.values())), "checks": checks},
        "external_refit": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v7_used": False,
    }
    summary_path = args.output_root / "route_a_fullalpha_periodicity_development_v7_frozen_summary.json"
    _write_json(summary_path, result)
    print(json.dumps({
        "status": "completed",
        "gate": result["confirmation_gate"],
        "comparisons": {key: value["aggregate"] for key, value in comparisons.items()},
        "summary": str(summary_path),
    }, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v7_generation/identity_audit.json")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--score-stats-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_development_v7_frozen")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
