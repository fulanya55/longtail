"""Fit preregistered low-capacity co-occurrence safety heads on development-v3 roles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, hamming_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS as U_VARIANTS
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import CLASS_NAMES, EXPECTED_V3_SHA256, _per_label_metrics, _validate_config
from experiments.run_shared import _data_path, load_config

TARGETS = {"MFT": 3, "VFT": 4, "ISRJ": 5}
VARIANTS = ("S0_U2Frozen", "S1_AlignedOrderBalanced", "S2_ShuffledOrderBalanced", "S3_AlignedUnweighted")
ALL_ROLES = ("fit", "validation", "early", "calibration", "threshold")


def _features(logits, target_index, permutation=None):
    values = np.asarray(logits, dtype=np.float32)
    other_indices = [index for index in range(values.shape[1]) if index != target_index]
    other = 1.0 / (1.0 + np.exp(-np.clip(values[:, other_indices], -30.0, 30.0)))
    if permutation is not None:
        other = other[np.asarray(permutation, dtype=np.int64)]
    target = values[:, target_index:target_index + 1]
    summary = np.stack([other.sum(axis=1), other.max(axis=1), other.std(axis=1)], axis=1)
    return np.concatenate([target, other, summary], axis=1).astype(np.float32)


def _sample_weights(labels, weak, target_index, order_balanced, weak_positive_weight):
    target = labels[:, target_index] == 1
    weights = np.ones(len(labels), dtype=np.float64)
    weights[target & weak[:, target_index]] = float(weak_positive_weight)
    if order_balanced:
        order = labels.sum(axis=1)
        negative = ~target
        supported_orders = [value for value in range(1, 6) if np.any(negative & (order == value))]
        total_negative = int(negative.sum())
        for value in supported_orders:
            mask = negative & (order == value)
            weights[mask] = total_negative / (len(supported_orders) * int(mask.sum()))
    return weights


def _fit_head(features, target, weights, seed, c_value):
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-5).astype(np.float32)
    normalized = ((features - mean) / std).astype(np.float32)
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=1000, random_state=int(seed))
    model.fit(normalized, target, sample_weight=weights)
    return model, mean, std


def _score(model, features, mean, std):
    normalized = ((features - mean) / np.maximum(std, 1e-5)).astype(np.float32)
    return model.decision_function(normalized).astype(np.float32)


def _fit_platt_parameter(logit, target):
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(np.asarray(logit)[:, None], np.asarray(target))
    return {"coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def _select_threshold(target, weak, probability, order, max_fpr, max_hard_fpr):
    target = np.asarray(target, dtype=np.uint8)
    weak_positive = (target == 1) & np.asarray(weak, dtype=bool)
    strong_positive = (target == 1) & ~np.asarray(weak, dtype=bool)
    negative = target == 0
    hard_negative = negative & (np.asarray(order) >= 4)
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = probability >= threshold
        fpr = float(prediction[negative].mean())
        hard_fpr = float(prediction[hard_negative].mean()) if hard_negative.any() else fpr
        weak_recall = float(prediction[weak_positive].mean()) if weak_positive.any() else 0.0
        strong_recall = float(prediction[strong_positive].mean()) if strong_positive.any() else 0.0
        if fpr <= max_fpr and hard_fpr <= max_hard_fpr:
            candidates.append((weak_recall, strong_recall, -fpr, -hard_fpr, threshold, fpr, hard_fpr))
    if not candidates:
        threshold = 0.99
        prediction = probability >= threshold
        return threshold, {"feasible": False, "overall_fpr": float(prediction[negative].mean()), "hard_order_ge4_fpr": float(prediction[hard_negative].mean())}
    selected = max(candidates)
    return float(selected[4]), {
        "feasible": True, "weak_recall": float(selected[0]), "strong_recall": float(selected[1]),
        "overall_fpr": float(selected[5]), "hard_order_ge4_fpr": float(selected[6]),
    }


def _metrics(labels, weak, probability, prediction):
    per_label = _per_label_metrics(labels, probability, prediction, weak)
    target = [per_label[name] for name in TARGETS]
    return {
        "macro_aupr": float(average_precision_score(labels, probability, average="macro")),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, prediction)),
        "per_label": per_label,
        "target_mean_weak_recall": float(np.mean([item["weak_recall"] for item in target])),
        "target_mean_strong_recall": float(np.mean([item["strong_recall"] for item in target])),
        "target_mean_negative_fpr": float(np.mean([item["negative_fpr"] for item in target])),
        "target_weak_vs_negative_auroc": float(np.mean([item["weak_vs_negative_auroc"] for item in target])),
    }


def _permutation(length, seed, role_index, target_index):
    return np.random.default_rng(seed + role_index * 100 + target_index).permutation(length)


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
    source_logits_file = source_root / "all_role_logits.npz"
    loaded = np.load(source_logits_file)
    source_logits = {role: np.asarray(loaded[f"{U_VARIANTS[2]}__{role}"], dtype=np.float32) for role in ALL_ROLES}
    source_calibration = json.loads((source_root / U_VARIANTS[2] / "calibration.json").read_text(encoding="utf-8"))
    source_thresholds = json.loads((source_root / U_VARIANTS[2] / "thresholds.json").read_text(encoding="utf-8"))

    variant_logits = {VARIANTS[0]: {role: source_logits[role].copy() for role in ALL_ROLES}}
    artifacts = {}
    specifications = {
        VARIANTS[1]: (False, True),
        VARIANTS[2]: (True, True),
        VARIANTS[3]: (False, False),
    }
    for variant_index, (variant, (shuffled, order_balanced)) in enumerate(specifications.items()):
        variant_logits[variant] = {role: source_logits[role].copy() for role in ALL_ROLES}
        artifacts[variant] = {}
        for target_name, target_index in TARGETS.items():
            feature_by_role = {}
            for role_index, role in enumerate(ALL_ROLES):
                permutation = _permutation(len(source_logits[role]), 20351000 + seed + variant_index * 1000, role_index, target_index) if shuffled else None
                feature_by_role[role] = _features(source_logits[role], target_index, permutation)
            weights = _sample_weights(role_labels["fit"], role_weak["fit"], target_index, order_balanced, args.weak_positive_weight)
            model, mean, std = _fit_head(feature_by_role["fit"], role_labels["fit"][:, target_index], weights, 20352000 + seed + variant_index * 100 + target_index, args.c_value)
            head_root = root / variant / "heads" / target_name
            head_root.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, head_root / "model.joblib")
            np.savez_compressed(head_root / "normalization.npz", mean=mean, std=std)
            for role in ALL_ROLES:
                variant_logits[variant][role][:, target_index] = _score(model, feature_by_role[role], mean, std)
            artifacts[variant][target_name] = {
                "model_sha256": _sha256(head_root / "model.joblib"),
                "normalization_sha256": _sha256(head_root / "normalization.npz"),
                "feature_count": int(mean.size), "order_balanced": bool(order_balanced),
                "cooccurrence_aligned": not shuffled,
            }

    variants = []
    for variant in VARIANTS:
        out = root / variant
        out.mkdir(parents=True, exist_ok=True)
        if variant == VARIANTS[0]:
            calibration = source_calibration
            threshold_payload = source_thresholds
        else:
            parameters = [dict(item) for item in source_calibration["parameters"]]
            for _, target_index in TARGETS.items():
                parameters[target_index] = _fit_platt_parameter(
                    variant_logits[variant]["calibration"][:, target_index],
                    role_labels["calibration"][:, target_index],
                )
            calibration = {"method": "hybrid_reuse_non_target_platt", "source_role": "calibration", "parameters": parameters}
            threshold_values = list(source_thresholds["thresholds"])
            threshold_receipt = {}
            order = role_labels["threshold"].sum(axis=1)
            threshold_probability = _apply_frozen_calibration(variant_logits[variant]["threshold"], calibration)
            for target_name, target_index in TARGETS.items():
                threshold, receipt = _select_threshold(
                    role_labels["threshold"][:, target_index], role_weak["threshold"][:, target_index],
                    threshold_probability[:, target_index], order, args.max_negative_fpr, args.max_hard_negative_fpr,
                )
                threshold_values[target_index] = threshold
                threshold_receipt[target_name] = receipt
            threshold_payload = {
                "source_role": "threshold_selection", "thresholds": threshold_values,
                "max_negative_fpr": args.max_negative_fpr,
                "max_order_ge4_negative_fpr": args.max_hard_negative_fpr,
                "target_selection_receipt": threshold_receipt,
                "non_target_thresholds_reused": True,
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
        variants.append({
            "variant": variant, "metrics": metrics,
            "calibration_sha256": _sha256(out / "calibration.json"),
            "thresholds_sha256": _sha256(out / "thresholds.json"),
            "predictions_sha256": _sha256(out / "predictions.npz"),
        })
    preservation = {
        variant: {
            "non_target_logit_max_abs": float(np.max(np.abs(variant_logits[variant]["early"][:, :3] - source_logits["early"][:, :3]))),
        }
        for variant in VARIANTS[1:]
    }
    row = {
        "seed": seed, "status": "completed", "data_sha256": data_sha,
        "source_logits_sha256": _sha256(source_logits_file), "role_indices_sha256": _sha256(role_path),
        "head_artifacts": artifacts, "preservation": preservation, "variants": variants,
        "candidate_fixed_before_development_v8": VARIANTS[1],
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    _write_json(receipt_path, row)
    return row


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
    result = {
        "status": "completed", "experiment": "RouteA_CooccurrenceSafetyHead_v3Fit",
        "evidence_role": "development_v3_fit_only_candidate_construction_not_confirmation",
        "candidate": VARIANTS[1], "seeds": args.seeds, "rows": rows,
        "candidate_selection_used_development_v7_metrics": False,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    output = args.output_root / "route_a_cooccurrence_safety_head_v3_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "candidate": result["candidate"], "summary": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_cooccurrence_safety_head_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--weak-positive-weight", type=float, default=4.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--max-hard-negative-fpr", type=float, default=0.30)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
