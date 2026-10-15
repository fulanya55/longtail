"""Frozen end-to-end U2 MFT peak-count guard confirmation on development-v10."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import SCORE_NAMES, extract_scores  # noqa: E402
from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_fullalpha_periodicity_development_v7_frozen import _run_seed as _run_u2_seed  # noqa: E402
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS as U_VARIANTS  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration, _load_signals  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import expert_logit, guarded_logits  # noqa: E402


MFT_INDEX = 3
PEAK_INDEX = SCORE_NAMES.index("nms_count_gt_bg_q99")
SEEDS = (2039, 2040, 2041, 2042, 2043)
VARIANTS = ("K0_U2Frozen", "K1_AlignedPeakNegativeGuard", "K2_WrongChirpNegativeGuard", "K3_ShuffledAlignedGuard")


def _hard_metrics(labels, weak, probabilities, predictions):
    order = labels.sum(axis=1)
    positive = labels[:, MFT_INDEX] == 1
    weak_positive = positive & weak[:, MFT_INDEX]
    hard_negative = (~positive) & (order >= 4)
    order5_negative = (~positive) & (order == 5)
    selected = weak_positive | hard_negative
    return {
        "hard_auroc": float(roc_auc_score(weak_positive[selected].astype(np.uint8), probabilities[selected, MFT_INDEX])),
        "hard_fpr": float(predictions[hard_negative, MFT_INDEX].mean()),
        "order5_fpr": float(predictions[order5_negative, MFT_INDEX].mean()),
        "weak_positive_count": int(weak_positive.sum()),
        "hard_negative_count": int(hard_negative.sum()),
        "order5_negative_count": int(order5_negative.sum()),
    }


def _evaluate(name, logits, labels, weak, receiver, calibration, thresholds, output_root):
    probabilities = _apply_frozen_calibration(logits, calibration)
    predictions = probabilities >= thresholds[None, :]
    metrics = _metrics(labels, weak, probabilities, predictions)
    hard = _hard_metrics(labels, weak, probabilities, predictions)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "metrics.json", {"metrics": metrics, "hard": hard})
    np.savez_compressed(
        output_root / "predictions.npz", receiver_noise_seed=receiver, targets=labels,
        weak_mask=weak.astype(np.uint8), logits=logits, probabilities=probabilities,
        predictions=predictions.astype(np.uint8), thresholds=thresholds,
    )
    return {"variant": name, "metrics": metrics, **hard}


def _run_guard_seed(args, labels, weak, receiver, count, seed, u2_row, manifest):
    source_root = args.source_root / f"seed_{seed}"
    u2_prediction = np.load(args.u2_work_root / f"seed_{seed}" / U_VARIANTS[2] / "predictions.npz")
    base_logits = np.asarray(u2_prediction["logits"], dtype=np.float32)
    if not np.array_equal(u2_prediction["targets"], labels):
        raise AssertionError("U2 development-v10 targets differ")
    calibration = json.loads((source_root / U_VARIANTS[2] / "calibration.json").read_text(encoding="utf-8"))
    threshold_payload = json.loads((source_root / U_VARIANTS[2] / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float32)
    expert_root = args.guard_root / f"seed_{seed}" / "experts"
    aligned_path = expert_root / "aligned_model.joblib"
    wrong_path = expert_root / "wrong_model.joblib"
    normalization_path = expert_root / "normalization.npz"
    frozen = manifest["seed_artifacts"][str(seed)]
    if _sha256(aligned_path).lower() != frozen["aligned_model_sha256"] or _sha256(normalization_path).lower() != frozen["normalization_sha256"]:
        raise ValueError(f"seed {seed} guard artifact hash mismatch")
    aligned_model = joblib.load(aligned_path)
    wrong_model = joblib.load(wrong_path)
    normalization = np.load(normalization_path)
    aligned = expert_logit(aligned_model, count["MF_AlignedLFM"], normalization["aligned_mean"], normalization["aligned_std"])
    wrong = expert_logit(wrong_model, count["MF_WrongChirp"], normalization["wrong_mean"], normalization["wrong_std"])
    permutation = np.random.default_rng(20407200 + seed * 10 + 2).permutation(len(aligned))
    evidence = {VARIANTS[1]: aligned, VARIANTS[2]: wrong, VARIANTS[3]: aligned[permutation]}
    logits = {VARIANTS[0]: base_logits}
    residuals = {VARIANTS[0]: np.zeros(len(base_logits), dtype=np.float32)}
    for variant in VARIANTS[1:]:
        logits[variant], residuals[variant] = guarded_logits(base_logits, evidence[variant], 0.75)
    root = args.output_root / f"seed_{seed}"
    results = [_evaluate(name, logits[name], labels, weak, receiver, calibration, thresholds, root / name) for name in VARIANTS]
    non_mft = [0, 1, 2, 4, 5]
    preservation = float(np.max(np.abs(logits[VARIANTS[1]][:, non_mft] - base_logits[:, non_mft])))
    row = {
        "seed": int(seed), "status": "completed", "u2_source": u2_row,
        "guard_artifacts": {"aligned_model_sha256": _sha256(aligned_path), "normalization_sha256": _sha256(normalization_path)},
        "variants": results, "non_mft_logit_max_abs": preservation,
        "residual_cap": 0.75, "external_refit": False, "test_v3_used": False, "test_v10_used": False,
    }
    _write_json(root / "seed_summary.json", row)
    return row


def _comparison(rows):
    per_seed = []
    for row in rows:
        v = {item["variant"]: item for item in row["variants"]}
        base, aligned, wrong, shuffled = v[VARIANTS[0]], v[VARIANTS[1]], v[VARIANTS[2]], v[VARIANTS[3]]
        per_seed.append({
            "seed": row["seed"],
            "macro_aupr": aligned["metrics"]["macro_aupr"] - base["metrics"]["macro_aupr"],
            "target_weak_recall": aligned["metrics"]["target_mean_weak_recall"] - base["metrics"]["target_mean_weak_recall"],
            "target_weak_auroc": aligned["metrics"]["target_weak_vs_negative_auroc"] - base["metrics"]["target_weak_vs_negative_auroc"],
            "hard_fpr": aligned["hard_fpr"] - base["hard_fpr"],
            "order5_fpr": aligned["order5_fpr"] - base["order5_fpr"],
            "hard_auroc": aligned["hard_auroc"] - base["hard_auroc"],
            "vs_wrong_hard_auroc": aligned["hard_auroc"] - wrong["hard_auroc"],
            "vs_shuffled_hard_auroc": aligned["hard_auroc"] - shuffled["hard_auroc"],
        })
    aggregate = {key: _mean_ci([row[key] for row in per_seed], 20408000 + index) for index, key in enumerate(per_seed[0]) if key != "seed"}
    aggregate["hard_fpr_reduction_positive_seeds"] = int(sum(row["hard_fpr"] < 0 for row in per_seed))
    return per_seed, aggregate


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["evidence_role"] != "development_v10" or audit["test_files_opened"]:
        raise ValueError("development-v10 audit failed")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    if manifest["candidate_name"] != "U2_MFT_AlignedPeakNegativeGuard_v1":
        raise ValueError("wrong frozen guard")
    data_path = Path(audit["development_v10_path"])
    if _sha256(data_path).lower() != audit["sha256"].lower():
        raise ValueError("development-v10 data hash changed")
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_scores = extract_scores(signals, device, args.peak_batch_size)
    count = {name: raw_scores[name][:, PEAK_INDEX] for name in ("MF_AlignedLFM", "MF_WrongChirp")}
    args.u2_work_root.mkdir(parents=True, exist_ok=True)
    u2_args = copy.copy(args)
    u2_args.output_root = args.u2_work_root
    u2_rows = [_run_u2_seed(u2_args, signals, labels, weak, receiver, seed, device, audit["sha256"]) for seed in args.seeds]
    rows = [_run_guard_seed(args, labels, weak, receiver, count, seed, u2_row, manifest) for seed, u2_row in zip(args.seeds, u2_rows)]
    per_seed, aggregate = _comparison(rows)
    preservation = all(row["non_mft_logit_max_abs"] == 0.0 for row in rows)
    checks = {
        "macro_aupr_delta_ge_minus0p002": aggregate["macro_aupr"]["mean"] >= -0.002,
        "target_weak_recall_delta_ge_minus0p010": aggregate["target_weak_recall"]["mean"] >= -0.010,
        "target_weak_auroc_delta_ge_0p005": aggregate["target_weak_auroc"]["mean"] >= 0.005,
        "hard_fpr_delta_le_minus0p020": aggregate["hard_fpr"]["mean"] <= -0.020,
        "order5_fpr_delta_le_minus0p020": aggregate["order5_fpr"]["mean"] <= -0.020,
        "hard_auroc_delta_ge_0p010": aggregate["hard_auroc"]["mean"] >= 0.010,
        "aligned_minus_wrong_ge_0p005": aggregate["vs_wrong_hard_auroc"]["mean"] >= 0.005,
        "aligned_minus_shuffled_ge_0p005": aggregate["vs_shuffled_hard_auroc"]["mean"] >= 0.005,
        "four_of_five_hard_fpr_reductions": aggregate["hard_fpr_reduction_positive_seeds"] >= 4,
        "non_mft_logits_exact": preservation,
    }
    result = {
        "status": "completed", "experiment": "RouteA_U2_MFT_PeakGuard_DevelopmentV10_Frozen",
        "evidence_role": "fresh_identity_end_to_end_frozen_confirmation", "development_v10_sha256": audit["sha256"],
        "rows": rows, "comparison": {"per_seed": per_seed, "aggregate": aggregate},
        "confirmation_gate": {"checks": checks, "passed": bool(all(checks.values()))},
        "external_refit": False, "locked_test_used": False, "test_v3_used": False, "test_v10_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_u2_mft_peak_guard_development_v10_frozen_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "gate": result["confirmation_gate"], "aggregate": aggregate, "summary": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v10_generation/identity_audit.json")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_u2_mft_peak_guard_v1/u2_mft_peak_guard_frozen_manifest.json")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--score-stats-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--guard-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_development_v10_frozen")
    parser.add_argument("--u2-work-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_guard_development_v10_frozen/u2_reconstruction")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--peak-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
