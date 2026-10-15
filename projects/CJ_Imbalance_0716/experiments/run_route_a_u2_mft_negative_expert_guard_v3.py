"""Fail-fast construction audit for a frozen MFT negative-evidence guard on U2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch  # Load Windows DLLs before scipy/sklearn imports.
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS as U_VARIANTS
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config
from experiments.run_shared import _data_path, load_config

MFT_INDEX = 3
VARIANTS = ("G0_U2Frozen", "G1_AlignedNegativeExpert", "G2_ShuffledNegativeExpert", "G3_AlignedSignedExpert")


def _guarded_logits(base_logits, expert_z, cap=0.5, signed=False):
    logits = np.asarray(base_logits, dtype=np.float32).copy()
    base = logits[:, MFT_INDEX]
    probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
    uncertainty = np.sqrt(np.maximum(4.0 * probability * (1.0 - probability), 0.0))
    direction = np.tanh(np.asarray(expert_z, dtype=np.float32))
    if not signed:
        direction = np.minimum(direction, 0.0)
    residual = float(cap) * uncertainty * direction
    logits[:, MFT_INDEX] = base + residual
    return logits, residual.astype(np.float32)


def _run_seed(args, labels, weak, seed, data_sha):
    root = args.output_root / f"seed_{seed}"
    receipt_path = root / "seed_summary.json"
    if args.resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root / f"seed_{seed}"
    role_path = args.c6_root / f"seed_{seed}" / "role_indices.npz"
    role_ids = np.load(role_path)
    early_ids = role_ids["early"].astype(np.int64)
    source_logits_path = source_root / "all_role_logits.npz"
    source_logits = np.asarray(np.load(source_logits_path)[f"{U_VARIANTS[2]}__early"], dtype=np.float32)
    expert_root = args.mft_expert_root / f"seed_{seed}"
    expert_path = expert_root / "expert_residuals.npz"
    expert = np.load(expert_path)
    if not np.array_equal(expert["early_sample_ids"].astype(np.int64), early_ids):
        raise AssertionError("MFT expert and U2 early identities differ")
    aligned_z = np.asarray(expert["early_expert_z"], dtype=np.float32)
    permutation = np.asarray(expert["early_permutation"], dtype=np.int64)
    shuffled_z = aligned_z[permutation]
    logits = {VARIANTS[0]: source_logits}
    residuals = {VARIANTS[0]: np.zeros(len(source_logits), dtype=np.float32)}
    logits[VARIANTS[1]], residuals[VARIANTS[1]] = _guarded_logits(source_logits, aligned_z, args.residual_cap, False)
    logits[VARIANTS[2]], residuals[VARIANTS[2]] = _guarded_logits(source_logits, shuffled_z, args.residual_cap, False)
    logits[VARIANTS[3]], residuals[VARIANTS[3]] = _guarded_logits(source_logits, aligned_z, args.residual_cap, True)
    calibration = json.loads((source_root / U_VARIANTS[2] / "calibration.json").read_text(encoding="utf-8"))
    thresholds_payload = json.loads((source_root / U_VARIANTS[2] / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = np.asarray(thresholds_payload["thresholds"], dtype=np.float32)
    targets = labels[early_ids]
    weak_early = weak[early_ids]
    order = targets.sum(axis=1)
    results = []
    for variant in VARIANTS:
        probability = _apply_frozen_calibration(logits[variant], calibration)
        prediction = probability >= thresholds[None, :]
        metrics = _metrics(targets, weak_early, probability, prediction)
        mft_positive = targets[:, MFT_INDEX] == 1
        weak_positive = mft_positive & weak_early[:, MFT_INDEX]
        hard_negative = (~mft_positive) & (order >= 4)
        rank = weak_positive | hard_negative
        hard_auroc = float(roc_auc_score(weak_positive[rank].astype(np.uint8), probability[rank, MFT_INDEX]))
        hard_fpr = float(prediction[hard_negative, MFT_INDEX].mean())
        out = root / variant
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "metrics.json", metrics)
        np.savez_compressed(
            out / "predictions.npz", sample_ids=early_ids, targets=targets,
            weak_mask=weak_early.astype(np.uint8), logits=logits[variant],
            probabilities=probability, predictions=prediction.astype(np.uint8),
            thresholds=thresholds, expert_z=aligned_z, residual=residuals[variant],
        )
        results.append({"variant": variant, "metrics": metrics, "mft_order_ge4_fpr": hard_fpr, "mft_weak_vs_order_ge4_negative_auroc": hard_auroc})
    non_mft = [0, 1, 2, 4, 5]
    preservation = {
        variant: float(np.max(np.abs(logits[variant][:, non_mft] - source_logits[:, non_mft])))
        for variant in VARIANTS[1:]
    }
    row = {
        "seed": seed, "status": "completed", "data_sha256": data_sha,
        "source_logits_sha256": _sha256(source_logits_path), "expert_residuals_sha256": _sha256(expert_path),
        "role_indices_sha256": _sha256(role_path), "variants": results,
        "non_mft_logit_max_abs": preservation, "residual_cap": args.residual_cap,
        "calibration_reused": True, "thresholds_reused": True,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    _write_json(receipt_path, row)
    return row


def _guard(rows):
    per_seed = []
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        base = variants[VARIANTS[0]]
        candidate = variants[VARIANTS[1]]
        shuffled = variants[VARIANTS[2]]
        per_seed.append({
            "seed": row["seed"],
            "macro_aupr": candidate["metrics"]["macro_aupr"] - base["metrics"]["macro_aupr"],
            "target_weak_recall": candidate["metrics"]["target_mean_weak_recall"] - base["metrics"]["target_mean_weak_recall"],
            "target_weak_auroc": candidate["metrics"]["target_weak_vs_negative_auroc"] - base["metrics"]["target_weak_vs_negative_auroc"],
            "hard_fpr": candidate["mft_order_ge4_fpr"] - base["mft_order_ge4_fpr"],
            "aligned_vs_shuffled_hard_auroc": candidate["mft_weak_vs_order_ge4_negative_auroc"] - shuffled["mft_weak_vs_order_ge4_negative_auroc"],
        })
    mean = {key: float(np.mean([item[key] for item in per_seed])) for key in per_seed[0] if key != "seed"}
    preservation = all(row["non_mft_logit_max_abs"][VARIANTS[1]] == 0.0 for row in rows)
    checks = {
        "macro_aupr_decline_le_0p005": mean["macro_aupr"] >= -0.005,
        "target_weak_recall_decline_le_0p010": mean["target_weak_recall"] >= -0.010,
        "target_weak_auroc_decline_le_0p005": mean["target_weak_auroc"] >= -0.005,
        "mft_hard_fpr_decrease_ge_0p020": mean["hard_fpr"] <= -0.020,
        "non_mft_logits_exact": preservation,
        "aligned_hard_auroc_not_below_shuffled": mean["aligned_vs_shuffled_hard_auroc"] >= 0.0,
    }
    return {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()}, "mean_delta": mean, "per_seed": per_seed}


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [_run_seed(args, labels, weak, seed, data_sha) for seed in args.seeds]
    guard = _guard(rows)
    result = {
        "status": "completed", "experiment": "RouteA_U2_MFT_NegativeExpertGuard_v3",
        "evidence_role": "fail_fast_construction_guard_not_promotion_evidence",
        "candidate": VARIANTS[1], "rows": rows, "construction_guard": guard,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    output = args.output_root / "route_a_u2_mft_negative_expert_guard_v3_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "guard": guard, "summary": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--mft-expert-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_frozen_expert_bounded_residual_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_negative_expert_guard_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--residual-cap", type=float, default=0.5)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
