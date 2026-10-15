"""Fit a tiny VFT head with explicit ISRJ hard-negative context and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch  # Load Windows DLLs before sklearn.
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_cooccurrence_safety_head_v3 import _fit_head, _fit_platt_parameter, _metrics, _score  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import _hard_metrics  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _load_roles, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


MFT_INDEX = 3
VFT_INDEX = 4
ISRJ_INDEX = 5
SEEDS = (2039, 2040, 2041, 2042, 2043)
HEADS = ("X1_AlignedCrossExpert", "X2_CapacityNoISRJ", "X3_ShuffledISRJ")
ROLES = ("fit", "validation", "early", "calibration", "threshold")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def cross_features(q1, q2, expert, kind, permutation=None):
    base = np.asarray(q1[:, VFT_INDEX], dtype=np.float32)
    expert_value = np.asarray(expert[:, VFT_INDEX], dtype=np.float32)
    delta = expert_value - np.asarray(q2[:, VFT_INDEX], dtype=np.float32)
    uncertainty = np.sqrt(np.maximum(4.0 * _sigmoid(base) * (1.0 - _sigmoid(base)), 0.0))
    if kind == HEADS[1]:
        return np.column_stack([base, expert_value, delta, uncertainty * delta, np.tanh(base) * delta]).astype(np.float32)
    isrj = np.asarray(q1[:, ISRJ_INDEX], dtype=np.float32)
    if kind == HEADS[2]:
        if permutation is None:
            raise ValueError("shuffled head requires a permutation")
        isrj = isrj[np.asarray(permutation, dtype=np.int64)]
    return np.column_stack([base, expert_value, delta, isrj, delta * _sigmoid(isrj)]).astype(np.float32)


def _platt_probability(score, parameter):
    return _sigmoid(float(parameter["coefficient"]) * score + float(parameter["intercept"])).astype(np.float32)


def _select_vft_threshold(labels, weak, probability, max_fpr, max_isrj_hard_fpr):
    positive = labels[:, VFT_INDEX] == 1
    weak_positive = positive & weak[:, VFT_INDEX]
    strong_positive = positive & ~weak[:, VFT_INDEX]
    negative = ~positive
    hard = negative & (labels[:, ISRJ_INDEX] == 1)
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = probability >= threshold
        fpr = float(prediction[negative].mean())
        hard_fpr = float(prediction[hard].mean())
        weak_recall = float(prediction[weak_positive].mean())
        strong_recall = float(prediction[strong_positive].mean())
        if fpr <= max_fpr and hard_fpr <= max_isrj_hard_fpr:
            candidates.append((weak_recall, strong_recall, -fpr, -hard_fpr, threshold, fpr, hard_fpr))
    if not candidates:
        return 0.99, {"feasible": False}
    selected = max(candidates)
    return float(selected[4]), {"feasible": True, "weak_recall": selected[0], "strong_recall": selected[1], "fpr": selected[5], "isrj_hard_fpr": selected[6]}


def _fit_seed(args, config, labels3, weak3, seed):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    loaded = np.load(args.construction_root / f"seed_{seed}" / "all_role_logits.npz")
    logits = {name: {role: np.asarray(loaded[f"{name}__{role}"], dtype=np.float32) for role in ROLES} for name in ("Q0_H1", "Q1_H1B2", "Q2_H1B2_MFTGuard", "Q3_VFTBaseOnlyCapacity", "Q4_VFTAlignedRidgeFusion", "Q5_VFTShuffledRidgeFusion")}
    role_labels = {role: np.asarray(roles[role]["targets6"], dtype=np.uint8) for role in ROLES}
    role_weak = {role: weak3[np.asarray(roles[role]["sample_ids"], dtype=np.int64)] for role in ROLES}
    models = {}
    for head_index, head in enumerate(HEADS):
        permutations = {role: np.random.default_rng(20432000 + seed * 10 + head_index * 100 + index).permutation(len(role_labels[role])) for index, role in enumerate(ROLES)}
        expert_name = {HEADS[0]: "Q4_VFTAlignedRidgeFusion", HEADS[1]: "Q3_VFTBaseOnlyCapacity", HEADS[2]: "Q4_VFTAlignedRidgeFusion"}[head]
        features = {role: cross_features(logits["Q1_H1B2"][role], logits["Q2_H1B2_MFTGuard"][role], logits[expert_name][role], head, permutations[role] if head == HEADS[2] else None) for role in ROLES}
        target = role_labels["fit"][:, VFT_INDEX]
        weights = np.ones(len(target), dtype=np.float64)
        weights[(target == 1) & role_weak["fit"][:, VFT_INDEX]] = args.weak_positive_weight
        weights[(target == 0) & (role_labels["fit"][:, ISRJ_INDEX] == 1)] = args.isrj_hard_negative_weight
        model, mean, std = _fit_head(features["fit"], target, weights, 20433000 + seed + head_index, args.c_value)
        scores = {role: _score(model, features[role], mean, std) for role in ROLES}
        platt = _fit_platt_parameter(scores["calibration"], role_labels["calibration"][:, VFT_INDEX])
        probabilities = {role: _platt_probability(scores[role], platt) for role in ROLES}
        threshold, threshold_receipt = _select_vft_threshold(role_labels["threshold"], role_weak["threshold"], probabilities["threshold"], args.max_fpr, args.max_isrj_hard_fpr)
        head_root = root / head; head_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, head_root / "model.joblib")
        np.savez_compressed(head_root / "normalization.npz", mean=mean, std=std)
        _write_json(head_root / "calibration.json", platt)
        _write_json(head_root / "threshold.json", {"threshold": threshold, "selection": threshold_receipt})
        models[head] = {"model": model, "mean": mean, "std": std, "platt": platt, "threshold": threshold, "permutations": permutations, "features": features}
    receipt = {"seed": seed, "role_indices_path": str(role_path), "heads": {name: {"threshold": item["threshold"], "trainable_parameters": int(item["model"].coef_.size + item["model"].intercept_.size)} for name, item in models.items()}, "locked_test_used": False, "test_v3_used": False}
    _write_json(root / "training_receipt.json", receipt)
    return models, logits, role_labels, role_weak


def _compose(q0, q1, q2, vft_score, vft_probability, vft_threshold, q0_cal, q0_thr, q1_cal, q1_thr, labels, weak):
    candidate_logits = np.asarray(q1, dtype=np.float32).copy()
    candidate_logits[:, MFT_INDEX] = q1[:, MFT_INDEX] + (0.50 / 0.75) * (q2[:, MFT_INDEX] - q1[:, MFT_INDEX])
    candidate_logits[:, VFT_INDEX] = vft_score
    probability = _apply_frozen_calibration(candidate_logits, q1_cal)
    prediction = probability >= np.asarray(q1_thr, dtype=np.float32)[None, :]
    probability[:, VFT_INDEX] = vft_probability
    prediction[:, VFT_INDEX] = vft_probability >= vft_threshold
    base_probability = _apply_frozen_calibration(q0, q0_cal)
    base_prediction = base_probability >= np.asarray(q0_thr, dtype=np.float32)[None, :]
    parent_probability = _apply_frozen_calibration(q1, q1_cal)
    parent_prediction = parent_probability >= np.asarray(q1_thr, dtype=np.float32)[None, :]
    metrics = _metrics(labels, weak, probability, prediction)
    base_metrics = _metrics(labels, weak, base_probability, base_prediction)
    parent_metrics = _metrics(labels, weak, parent_probability, parent_prediction)
    hard_negative = (labels[:, VFT_INDEX] == 0) & (labels[:, ISRJ_INDEX] == 1)
    return {
        "metrics": metrics, "base_metrics": base_metrics, "parent_metrics": parent_metrics, "mft_hard": _hard_metrics(labels, weak, probability, prediction),
        "vft_isrj_hard_fpr": float(prediction[hard_negative, VFT_INDEX].mean()),
        "sawfm_probability_max_abs": float(np.max(np.abs(probability[:, 1] - base_probability[:, 1]))),
        "non_target_prediction_mismatch": int((prediction[:, [0, 1, 2, 5]] != parent_prediction[:, [0, 1, 2, 5]]).sum()),
        "probabilities": probability, "predictions": prediction,
    }


def _external_seed(args, dataset_root, seed, models, suffix):
    arrays = {name: {key: np.asarray(value) for key, value in np.load(dataset_root / f"seed_{seed}" / name / "predictions.npz").items()} for name in ("Q0_H1", "Q1_H1B2", "Q2_H1B2_MFTGuard", "Q3_VFTBaseOnlyCapacity", "Q4_VFTAlignedRidgeFusion", "Q5_VFTShuffledRidgeFusion")}
    labels = arrays["Q0_H1"]["targets"].astype(np.uint8); weak = arrays["Q0_H1"]["weak_mask"].astype(bool)
    source = args.construction_root / f"seed_{seed}"
    q0_cal = _json(source / "Q0_H1/calibration.json"); q0_thr = _json(source / "Q0_H1/thresholds.json")["thresholds"]
    q1_cal = _json(source / "Q1_H1B2/calibration.json"); q1_thr = _json(source / "Q1_H1B2/thresholds.json")["thresholds"]
    output = {}
    for head_index, head in enumerate(HEADS):
        expert_name = {HEADS[0]: "Q4_VFTAlignedRidgeFusion", HEADS[1]: "Q3_VFTBaseOnlyCapacity", HEADS[2]: "Q4_VFTAlignedRidgeFusion"}[head]
        permutation = np.random.default_rng(20434000 + seed * 10 + head_index).permutation(len(labels)) if head == HEADS[2] else None
        feature = cross_features(arrays["Q1_H1B2"]["logits"], arrays["Q2_H1B2_MFTGuard"]["logits"], arrays[expert_name]["logits"], head, permutation)
        item = models[head]
        score = _score(item["model"], feature, item["mean"], item["std"])
        probability = _platt_probability(score, item["platt"])
        output[head] = _compose(arrays["Q0_H1"]["logits"], arrays["Q1_H1B2"]["logits"], arrays["Q2_H1B2_MFTGuard"]["logits"], score, probability, item["threshold"], q0_cal, q0_thr, q1_cal, q1_thr, labels, weak)
    return output


def _construction_seed(args, seed, models, logits, labels, weak):
    source = args.construction_root / f"seed_{seed}"
    q0_cal = _json(source / "Q0_H1/calibration.json"); q0_thr = _json(source / "Q0_H1/thresholds.json")["thresholds"]
    q1_cal = _json(source / "Q1_H1B2/calibration.json"); q1_thr = _json(source / "Q1_H1B2/thresholds.json")["thresholds"]
    output = {}
    for head in HEADS:
        item = models[head]
        score = _score(item["model"], item["features"]["threshold"], item["mean"], item["std"])
        probability = _platt_probability(score, item["platt"])
        output[head] = _compose(logits["Q0_H1"]["threshold"], logits["Q1_H1B2"]["threshold"], logits["Q2_H1B2_MFTGuard"]["threshold"], score, probability, item["threshold"], q0_cal, q0_thr, q1_cal, q1_thr, labels["threshold"], weak["threshold"])
    return output


def _summarize(name, rows, macro_floor, auroc_floor):
    per_seed = []
    for seed, variants in rows:
        aligned, capacity, shuffled = variants[HEADS[0]], variants[HEADS[1]], variants[HEADS[2]]
        metric, base = aligned["metrics"], aligned["base_metrics"]
        parent_vft = aligned["parent_metrics"]["per_label"]["VFT"]
        per_seed.append({
            "seed": seed, "macro_aupr": metric["macro_aupr"] - base["macro_aupr"], "macro_f1": metric["macro_f1"] - base["macro_f1"],
            "target_weak_recall": metric["target_mean_weak_recall"] - base["target_mean_weak_recall"], "target_weak_auroc": metric["target_weak_vs_negative_auroc"] - base["target_weak_vs_negative_auroc"], "target_fpr": metric["target_mean_negative_fpr"],
            "vft_aupr": metric["per_label"]["VFT"]["aupr"] - parent_vft["aupr"], "vft_weak_recall": metric["per_label"]["VFT"]["weak_recall"] - parent_vft["weak_recall"], "vft_weak_auroc": metric["per_label"]["VFT"]["weak_vs_negative_auroc"] - parent_vft["weak_vs_negative_auroc"],
            "vft_auroc_vs_capacity": metric["per_label"]["VFT"]["weak_vs_negative_auroc"] - capacity["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"], "vft_auroc_vs_shuffled": metric["per_label"]["VFT"]["weak_vs_negative_auroc"] - shuffled["metrics"]["per_label"]["VFT"]["weak_vs_negative_auroc"],
            "vft_fpr": metric["per_label"]["VFT"]["negative_fpr"], "vft_isrj_hard_fpr": aligned["vft_isrj_hard_fpr"], "sawfm_probability_max_abs": aligned["sawfm_probability_max_abs"], "non_target_prediction_mismatch": aligned["non_target_prediction_mismatch"],
        })
    keys = tuple(key for key in per_seed[0] if key not in ("seed", "non_target_prediction_mismatch"))
    aggregate = {key: _mean_ci([row[key] for row in per_seed], 20435000 + index) for index, key in enumerate(keys)}
    checks = {
        "macro_aupr_gain_meets_floor": aggregate["macro_aupr"]["mean"] >= macro_floor,
        "target_weak_recall_gain_ge_0p030": aggregate["target_weak_recall"]["mean"] >= 0.030,
        "target_weak_auroc_gain_meets_floor": aggregate["target_weak_auroc"]["mean"] >= auroc_floor,
        "target_fpr_le_0p20": aggregate["target_fpr"]["mean"] <= 0.20,
        "vft_aupr_nonnegative": aggregate["vft_aupr"]["mean"] >= 0.0,
        "vft_weak_recall_gain_ge_0p010": aggregate["vft_weak_recall"]["mean"] >= 0.010,
        "vft_weak_auroc_gain_ge_0p010": aggregate["vft_weak_auroc"]["mean"] >= 0.010,
        "vft_beats_capacity_ge_0p005": aggregate["vft_auroc_vs_capacity"]["mean"] >= 0.005,
        "vft_beats_shuffled_ge_0p005": aggregate["vft_auroc_vs_shuffled"]["mean"] >= 0.005,
        "vft_fpr_le_0p20": aggregate["vft_fpr"]["mean"] <= 0.20,
        "vft_isrj_hard_fpr_le_0p25": aggregate["vft_isrj_hard_fpr"]["mean"] <= 0.25,
        "sawfm_probability_exact": max(row["sawfm_probability_max_abs"] for row in per_seed) == 0.0,
        "non_target_predictions_exact": sum(row["non_target_prediction_mismatch"] for row in per_seed) == 0,
    }
    return {"dataset": name, "per_seed": per_seed, "aggregate": aggregate, "checks": checks, "passed": all(checks.values())}


def run(args):
    seal = _json(args.test_v3_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    config = load_config(args.config); _validate_config(config)
    labels3 = read_labels(_data_path(config)); params3, _ = read_parameters(_data_path(config))
    weak3 = compute_weak_component_mask(labels3, _component_power(params3), args.weak_threshold_db)
    datasets = {"construction_v3": [], "development_v12_diagnostic": [], "isrj_parameter_ood_diagnostic": []}
    for seed in args.seeds:
        models, logits, role_labels, role_weak = _fit_seed(args, config, labels3, weak3, seed)
        datasets["construction_v3"].append((seed, _construction_seed(args, seed, models, logits, role_labels, role_weak)))
        datasets["development_v12_diagnostic"].append((seed, _external_seed(args, args.v12_root, seed, models, "v12")))
        datasets["isrj_parameter_ood_diagnostic"].append((seed, _external_seed(args, args.ood_root, seed, models, "ood")))
    summaries = {
        "construction_v3": _summarize("construction_v3", datasets["construction_v3"], 0.005, 0.015),
        "development_v12_diagnostic": _summarize("development_v12_diagnostic", datasets["development_v12_diagnostic"], 0.005, 0.015),
        "isrj_parameter_ood_diagnostic": _summarize("isrj_parameter_ood_diagnostic", datasets["isrj_parameter_ood_diagnostic"], 0.0, 0.0),
    }
    result = {"status": "completed", "experiment": "RouteA_VFT_ISRJ_CrossExpertHead", "heads": list(HEADS), "trainable_parameters_per_head": 6, "summaries": summaries, "passed": all(item["passed"] for item in summaries.values()), "external_refit": False, "hyperparameter_search": False, "locked_test_used": False, "test_v3_used": False}
    output = args.output_root / "route_a_vft_isrj_cross_expert_head_summary.json"; args.output_root.mkdir(parents=True, exist_ok=True); _write_json(output, result)
    print(json.dumps({"passed": result["passed"], "summaries": {name: {"passed": value["passed"], "checks": value["checks"], "aggregate": value["aggregate"]} for name, value in summaries.items()}, "output": str(output)}, indent=2)); return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    p.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    p.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    p.add_argument("--v12-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen")
    p.add_argument("--ood-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_shared_policy_isrj_ood_frozen")
    p.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_cross_expert_head")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); p.add_argument("--io-batch-size", type=int, default=256)
    p.add_argument("--weak-threshold-db", type=float, default=10.0); p.add_argument("--weak-positive-weight", type=float, default=4.0); p.add_argument("--isrj-hard-negative-weight", type=float, default=4.0)
    p.add_argument("--c-value", type=float, default=0.1); p.add_argument("--max-fpr", type=float, default=0.20); p.add_argument("--max-isrj-hard-fpr", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
