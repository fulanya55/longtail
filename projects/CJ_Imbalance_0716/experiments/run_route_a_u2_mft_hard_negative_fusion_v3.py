"""Fit and fail-fast audit an MFT physical hard-negative fusion head on U2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch  # Load Windows DLLs before sklearn.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_cooccurrence_safety_head_v3 import _fit_platt_parameter, _metrics, _select_threshold
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS as U_VARIANTS
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config
from experiments.run_shared import _data_path, load_config

MFT_INDEX = 3
ALL_ROLES = ("fit", "validation", "early", "calibration", "threshold")
VARIANTS = ("P0_U2Frozen", "P1_AlignedPhysicalFusion", "P2_BaseOnlyCapacity", "P3_ShuffledPhysicalFusion")


def _fusion_features(logits, expert_z=None):
    base = np.asarray(logits, dtype=np.float32)[:, MFT_INDEX]
    if expert_z is None:
        return base[:, None]
    probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
    uncertainty = np.sqrt(np.maximum(4.0 * probability * (1.0 - probability), 0.0))
    z = np.asarray(expert_z, dtype=np.float32)
    return np.stack([base, z, uncertainty * z, np.tanh(base) * z], axis=1).astype(np.float32)


def _weights(labels, weak, weak_weight, hard_negative_weight):
    positive = labels[:, MFT_INDEX] == 1
    hard_negative = (~positive) & (labels.sum(axis=1) >= 4)
    weights = np.ones(len(labels), dtype=np.float64)
    weights[positive & weak[:, MFT_INDEX]] = float(weak_weight)
    weights[hard_negative] = float(hard_negative_weight)
    return weights


def _fit(features, target, weights, seed, c_value):
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-5).astype(np.float32)
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=1000, random_state=int(seed))
    model.fit(((features - mean) / std).astype(np.float32), target, sample_weight=weights)
    return model, mean, std


def _score(model, features, mean, std):
    return model.decision_function(((features - mean) / np.maximum(std, 1e-5)).astype(np.float32)).astype(np.float32)


def _run_seed(args, labels, weak, seed, data_sha):
    root = args.output_root / f"seed_{seed}"
    receipt_path = root / "seed_summary.json"
    if args.resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root / f"seed_{seed}"
    role_path = args.c6_root / f"seed_{seed}" / "role_indices.npz"
    role_ids = np.load(role_path)
    role_labels = {role: labels[role_ids[role].astype(np.int64)] for role in ALL_ROLES}
    role_weak = {role: weak[role_ids[role].astype(np.int64)] for role in ALL_ROLES}
    logits_path = source_root / "all_role_logits.npz"
    loaded_logits = np.load(logits_path)
    source = {role: np.asarray(loaded_logits[f"{U_VARIANTS[2]}__{role}"], dtype=np.float32) for role in ALL_ROLES}
    expert_path = args.mft_expert_root / f"seed_{seed}" / "expert_residuals.npz"
    expert = np.load(expert_path)
    aligned_z = {}
    shuffled_z = {}
    for role in ALL_ROLES:
        if not np.array_equal(expert[f"{role}_sample_ids"].astype(np.int64), role_ids[role].astype(np.int64)):
            raise AssertionError(f"{role} identities differ")
        aligned_z[role] = np.asarray(expert[f"{role}_expert_z"], dtype=np.float32)
        shuffled_z[role] = aligned_z[role][np.asarray(expert[f"{role}_permutation"], dtype=np.int64)]
    weights = _weights(role_labels["fit"], role_weak["fit"], args.weak_positive_weight, args.hard_negative_weight)
    feature_sets = {
        VARIANTS[1]: {role: _fusion_features(source[role], aligned_z[role]) for role in ALL_ROLES},
        VARIANTS[2]: {role: _fusion_features(source[role], None) for role in ALL_ROLES},
        VARIANTS[3]: {role: _fusion_features(source[role], shuffled_z[role]) for role in ALL_ROLES},
    }
    logits = {VARIANTS[0]: {role: source[role].copy() for role in ALL_ROLES}}
    artifacts = {}
    for variant_index, variant in enumerate(VARIANTS[1:]):
        model, mean, std = _fit(
            feature_sets[variant]["fit"], role_labels["fit"][:, MFT_INDEX], weights,
            20355000 + seed + variant_index * 100, args.c_value,
        )
        head_root = root / variant / "head_MFT"
        head_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, head_root / "model.joblib")
        np.savez_compressed(head_root / "normalization.npz", mean=mean, std=std)
        logits[variant] = {role: source[role].copy() for role in ALL_ROLES}
        for role in ALL_ROLES:
            logits[variant][role][:, MFT_INDEX] = _score(model, feature_sets[variant][role], mean, std)
        artifacts[variant] = {
            "model_sha256": _sha256(head_root / "model.joblib"),
            "normalization_sha256": _sha256(head_root / "normalization.npz"),
            "feature_count": int(mean.size),
        }
    source_calibration = json.loads((source_root / U_VARIANTS[2] / "calibration.json").read_text(encoding="utf-8"))
    source_thresholds = json.loads((source_root / U_VARIANTS[2] / "thresholds.json").read_text(encoding="utf-8"))
    results = []
    for variant in VARIANTS:
        out = root / variant
        out.mkdir(parents=True, exist_ok=True)
        if variant == VARIANTS[0]:
            calibration = source_calibration
            thresholds_payload = source_thresholds
        else:
            parameters = [dict(item) for item in source_calibration["parameters"]]
            parameters[MFT_INDEX] = _fit_platt_parameter(logits[variant]["calibration"][:, MFT_INDEX], role_labels["calibration"][:, MFT_INDEX])
            calibration = {"method": "reuse_non_mft_platt", "source_role": "calibration", "parameters": parameters}
            threshold_probability = _apply_frozen_calibration(logits[variant]["threshold"], calibration)
            threshold, receipt = _select_threshold(
                role_labels["threshold"][:, MFT_INDEX], role_weak["threshold"][:, MFT_INDEX],
                threshold_probability[:, MFT_INDEX], role_labels["threshold"].sum(axis=1),
                args.max_negative_fpr, args.max_hard_negative_fpr,
            )
            threshold_values = list(source_thresholds["thresholds"])
            threshold_values[MFT_INDEX] = threshold
            thresholds_payload = {
                "source_role": "threshold_selection", "thresholds": threshold_values,
                "mft_selection_receipt": receipt, "max_negative_fpr": args.max_negative_fpr,
                "max_order_ge4_negative_fpr": args.max_hard_negative_fpr,
                "non_mft_thresholds_reused": True,
            }
        probability = _apply_frozen_calibration(logits[variant]["early"], calibration)
        thresholds = np.asarray(thresholds_payload["thresholds"], dtype=np.float32)
        prediction = probability >= thresholds[None, :]
        metrics = _metrics(role_labels["early"], role_weak["early"], probability, prediction)
        target = role_labels["early"][:, MFT_INDEX] == 1
        weak_positive = target & role_weak["early"][:, MFT_INDEX]
        hard_negative = (~target) & (role_labels["early"].sum(axis=1) >= 4)
        rank = weak_positive | hard_negative
        hard_auroc = float(roc_auc_score(weak_positive[rank].astype(np.uint8), probability[rank, MFT_INDEX]))
        hard_fpr = float(prediction[hard_negative, MFT_INDEX].mean())
        _write_json(out / "calibration.json", calibration)
        _write_json(out / "thresholds.json", thresholds_payload)
        _write_json(out / "metrics.json", metrics)
        np.savez_compressed(
            out / "predictions.npz", sample_ids=role_ids["early"].astype(np.int64),
            targets=role_labels["early"], weak_mask=role_weak["early"].astype(np.uint8),
            logits=logits[variant]["early"], probabilities=probability,
            predictions=prediction.astype(np.uint8), thresholds=thresholds,
        )
        results.append({
            "variant": variant, "metrics": metrics, "mft_order_ge4_fpr": hard_fpr,
            "mft_weak_vs_order_ge4_negative_auroc": hard_auroc,
            "thresholds": thresholds_payload,
        })
    non_mft = [0, 1, 2, 4, 5]
    preservation = {
        variant: float(np.max(np.abs(logits[variant]["early"][:, non_mft] - source["early"][:, non_mft])))
        for variant in VARIANTS[1:]
    }
    row = {
        "seed": seed, "status": "completed", "data_sha256": data_sha,
        "source_logits_sha256": _sha256(logits_path), "expert_residuals_sha256": _sha256(expert_path),
        "role_indices_sha256": _sha256(role_path), "head_artifacts": artifacts,
        "non_mft_logit_max_abs": preservation, "variants": results,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    _write_json(receipt_path, row)
    return row


def _guard(rows):
    per_seed = []
    feasible = True
    preservation = True
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        base, candidate = variants[VARIANTS[0]], variants[VARIANTS[1]]
        per_seed.append({
            "seed": row["seed"],
            "macro_aupr": candidate["metrics"]["macro_aupr"] - base["metrics"]["macro_aupr"],
            "target_weak_recall": candidate["metrics"]["target_mean_weak_recall"] - base["metrics"]["target_mean_weak_recall"],
            "target_weak_auroc": candidate["metrics"]["target_weak_vs_negative_auroc"] - base["metrics"]["target_weak_vs_negative_auroc"],
            "hard_fpr": candidate["mft_order_ge4_fpr"] - base["mft_order_ge4_fpr"],
            "hard_auroc": candidate["mft_weak_vs_order_ge4_negative_auroc"] - base["mft_weak_vs_order_ge4_negative_auroc"],
            "vs_base_only_hard_auroc": candidate["mft_weak_vs_order_ge4_negative_auroc"] - variants[VARIANTS[2]]["mft_weak_vs_order_ge4_negative_auroc"],
            "vs_shuffled_hard_auroc": candidate["mft_weak_vs_order_ge4_negative_auroc"] - variants[VARIANTS[3]]["mft_weak_vs_order_ge4_negative_auroc"],
        })
        feasible &= bool(candidate["thresholds"]["mft_selection_receipt"]["feasible"])
        preservation &= row["non_mft_logit_max_abs"][VARIANTS[1]] == 0.0
    mean = {key: float(np.mean([item[key] for item in per_seed])) for key in per_seed[0] if key != "seed"}
    checks = {
        "macro_aupr_decline_le_0p005": mean["macro_aupr"] >= -0.005,
        "target_weak_recall_decline_le_0p010": mean["target_weak_recall"] >= -0.010,
        "target_weak_auroc_decline_le_0p005": mean["target_weak_auroc"] >= -0.005,
        "hard_fpr_decrease_ge_0p030": mean["hard_fpr"] <= -0.030,
        "hard_auroc_gain_ge_0p010": mean["hard_auroc"] >= 0.010,
        "aligned_vs_base_only_hard_auroc_ge_0p005": mean["vs_base_only_hard_auroc"] >= 0.005,
        "aligned_vs_shuffled_hard_auroc_ge_0p005": mean["vs_shuffled_hard_auroc"] >= 0.005,
        "non_mft_logits_exact": preservation,
        "threshold_feasible_all_seeds": feasible,
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
        "status": "completed", "experiment": "RouteA_U2_MFT_HardNegativeFusion_v3",
        "evidence_role": "fail_fast_construction_guard_not_promotion_evidence",
        "candidate": VARIANTS[1], "rows": rows, "construction_guard": guard,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    output = args.output_root / "route_a_u2_mft_hard_negative_fusion_v3_summary.json"
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
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_hard_negative_fusion_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--weak-positive-weight", type=float, default=4.0)
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--max-hard-negative-fpr", type=float, default=0.30)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
