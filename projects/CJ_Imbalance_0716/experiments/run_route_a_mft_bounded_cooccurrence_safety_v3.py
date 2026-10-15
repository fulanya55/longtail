"""Fit a one-sided bounded MFT co-occurrence safeguard on development-v3 roles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_cooccurrence_safety_head_v3 import (
    _features, _fit_head, _fit_platt_parameter, _metrics, _permutation,
    _sample_weights, _select_threshold,
)
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS as U_VARIANTS
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config
from experiments.run_shared import _data_path, load_config

MFT_INDEX = 3
ALL_ROLES = ("fit", "validation", "early", "calibration", "threshold")
VARIANTS = ("R0_U2Frozen", "R1_AlignedOrderBalancedBounded", "R2_ShuffledOrderBalancedBounded", "R3_AlignedUnweightedBounded")


def _bounded_score(base, head_score, cap):
    base = np.asarray(base, dtype=np.float32)
    proposed = float(cap) * np.tanh(np.asarray(head_score, dtype=np.float32) - base)
    return (base + np.minimum(proposed, 0.0)).astype(np.float32)


def _run_seed(args, labels, weak, seed, data_sha):
    root = args.output_root / f"seed_{seed}"
    receipt_path = root / "seed_summary.json"
    if args.resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    role_path = args.c6_root / f"seed_{seed}" / "role_indices.npz"
    role_ids = np.load(role_path)
    role_labels = {role: labels[role_ids[role].astype(np.int64)] for role in ALL_ROLES}
    role_weak = {role: weak[role_ids[role].astype(np.int64)] for role in ALL_ROLES}
    source_root = args.source_root / f"seed_{seed}"
    logits_path = source_root / "all_role_logits.npz"
    loaded = np.load(logits_path)
    source = {role: np.asarray(loaded[f"{U_VARIANTS[2]}__{role}"], dtype=np.float32) for role in ALL_ROLES}
    source_calibration = json.loads((source_root / U_VARIANTS[2] / "calibration.json").read_text(encoding="utf-8"))
    source_thresholds = json.loads((source_root / U_VARIANTS[2] / "thresholds.json").read_text(encoding="utf-8"))
    variant_logits = {VARIANTS[0]: {role: source[role].copy() for role in ALL_ROLES}}
    specifications = {
        VARIANTS[1]: (False, True), VARIANTS[2]: (True, True), VARIANTS[3]: (False, False),
    }
    artifacts = {}
    for variant_index, (variant, (shuffled, balanced)) in enumerate(specifications.items()):
        feature_by_role = {}
        for role_index, role in enumerate(ALL_ROLES):
            permutation = _permutation(len(source[role]), 20353000 + seed + variant_index * 1000, role_index, MFT_INDEX) if shuffled else None
            feature_by_role[role] = _features(source[role], MFT_INDEX, permutation)
        weights = _sample_weights(role_labels["fit"], role_weak["fit"], MFT_INDEX, balanced, args.weak_positive_weight)
        model, mean, std = _fit_head(feature_by_role["fit"], role_labels["fit"][:, MFT_INDEX], weights, 20354000 + seed + variant_index, args.c_value)
        head_root = root / variant / "head_MFT"
        head_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, head_root / "model.joblib")
        np.savez_compressed(head_root / "normalization.npz", mean=mean, std=std)
        variant_logits[variant] = {role: source[role].copy() for role in ALL_ROLES}
        for role in ALL_ROLES:
            normalized = ((feature_by_role[role] - mean) / np.maximum(std, 1e-5)).astype(np.float32)
            head_score = model.decision_function(normalized).astype(np.float32)
            variant_logits[variant][role][:, MFT_INDEX] = _bounded_score(source[role][:, MFT_INDEX], head_score, args.residual_cap)
        artifacts[variant] = {
            "model_sha256": _sha256(head_root / "model.joblib"),
            "normalization_sha256": _sha256(head_root / "normalization.npz"),
            "cooccurrence_aligned": not shuffled, "order_balanced": bool(balanced),
            "residual_cap": args.residual_cap, "one_sided_downward": True,
        }

    results = []
    for variant in VARIANTS:
        out = root / variant
        out.mkdir(parents=True, exist_ok=True)
        if variant == VARIANTS[0]:
            calibration = source_calibration
            threshold_payload = source_thresholds
        else:
            parameters = [dict(item) for item in source_calibration["parameters"]]
            parameters[MFT_INDEX] = _fit_platt_parameter(
                variant_logits[variant]["calibration"][:, MFT_INDEX], role_labels["calibration"][:, MFT_INDEX]
            )
            calibration = {"method": "reuse_non_mft_platt", "source_role": "calibration", "parameters": parameters}
            threshold_values = list(source_thresholds["thresholds"])
            threshold_probability = _apply_frozen_calibration(variant_logits[variant]["threshold"], calibration)
            threshold, threshold_receipt = _select_threshold(
                role_labels["threshold"][:, MFT_INDEX], role_weak["threshold"][:, MFT_INDEX],
                threshold_probability[:, MFT_INDEX], role_labels["threshold"].sum(axis=1),
                args.max_negative_fpr, args.max_hard_negative_fpr,
            )
            threshold_values[MFT_INDEX] = threshold
            threshold_payload = {
                "source_role": "threshold_selection", "thresholds": threshold_values,
                "mft_selection_receipt": threshold_receipt,
                "max_negative_fpr": args.max_negative_fpr,
                "max_order_ge4_negative_fpr": args.max_hard_negative_fpr,
                "non_mft_thresholds_reused": True,
            }
        probability = _apply_frozen_calibration(variant_logits[variant]["early"], calibration)
        thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float32)
        prediction = probability >= thresholds[None, :]
        metrics = _metrics(role_labels["early"], role_weak["early"], probability, prediction)
        _write_json(out / "calibration.json", calibration)
        _write_json(out / "thresholds.json", threshold_payload)
        _write_json(out / "metrics.json", metrics)
        np.savez_compressed(
            out / "predictions.npz", sample_ids=role_ids["early"].astype(np.int64),
            targets=role_labels["early"], weak_mask=role_weak["early"].astype(np.uint8),
            logits=variant_logits[variant]["early"], probabilities=probability,
            predictions=prediction.astype(np.uint8), thresholds=thresholds,
        )
        results.append({"variant": variant, "metrics": metrics, "thresholds": threshold_payload})
    reference = variant_logits[VARIANTS[0]]["early"]
    non_mft = [0, 1, 2, 4, 5]
    preservation = {
        variant: float(np.max(np.abs(variant_logits[variant]["early"][:, non_mft] - reference[:, non_mft])))
        for variant in VARIANTS[1:]
    }
    row = {
        "seed": seed, "status": "completed", "data_sha256": data_sha,
        "source_logits_sha256": _sha256(logits_path), "role_indices_sha256": _sha256(role_path),
        "artifacts": artifacts, "non_mft_logit_max_abs": preservation, "variants": results,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    _write_json(receipt_path, row)
    return row


def _construction_guard(rows):
    deltas = []
    feasible = True
    preservation = True
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        base = variants[VARIANTS[0]]["metrics"]
        candidate = variants[VARIANTS[1]]["metrics"]
        deltas.append({
            "macro_aupr": candidate["macro_aupr"] - base["macro_aupr"],
            "target_weak_recall": candidate["target_mean_weak_recall"] - base["target_mean_weak_recall"],
            "target_weak_auroc": candidate["target_weak_vs_negative_auroc"] - base["target_weak_vs_negative_auroc"],
        })
        feasible &= bool(variants[VARIANTS[1]]["thresholds"]["mft_selection_receipt"]["feasible"])
        preservation &= row["non_mft_logit_max_abs"][VARIANTS[1]] == 0.0
    means = {key: float(np.mean([item[key] for item in deltas])) for key in deltas[0]}
    checks = {
        "macro_aupr_decline_le_0p005": means["macro_aupr"] >= -0.005,
        "target_weak_recall_decline_le_0p010": means["target_weak_recall"] >= -0.010,
        "target_weak_auroc_decline_le_0p005": means["target_weak_auroc"] >= -0.005,
        "non_mft_logits_exact": preservation,
        "threshold_feasible_all_seeds": feasible,
    }
    return {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()}, "mean_delta": means, "per_seed": deltas}


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
    guard = _construction_guard(rows)
    result = {
        "status": "completed", "experiment": "RouteA_MFT_BoundedCooccurrenceSafety_v3",
        "evidence_role": "fail_fast_construction_guard_not_promotion_evidence",
        "candidate": VARIANTS[1], "rows": rows, "construction_guard": guard,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    output = args.output_root / "route_a_mft_bounded_cooccurrence_safety_v3_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "guard": guard, "summary": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_bounded_cooccurrence_safety_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--weak-positive-weight", type=float, default=4.0)
    parser.add_argument("--residual-cap", type=float, default=0.5)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--max-hard-negative-fpr", type=float, default=0.30)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
