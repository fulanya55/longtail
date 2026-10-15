"""Frozen stratified robustness audit for unified U2 on development-v7."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch  # Load Windows DLLs before scipy/sklearn.
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import CLASS_NAMES, SAW_INDEX

EXPECTED_V7_SHA256 = "cb7bc5014fba526c40b61b0ba34d358bceb322b075e8dff867f8f2c4cf08d88d"
TARGETS = {"MFT": 3, "VFT": 4, "ISRJ": 5}
METRICS = ("aupr", "weak_recall", "strong_recall", "negative_fpr", "weak_auroc")
COMPARISONS = {
    "U2_vs_U0": (VARIANTS[2], VARIANTS[0]),
    "U2_vs_U1": (VARIANTS[2], VARIANTS[1]),
    "U2_vs_U3": (VARIANTS[2], VARIANTS[3]),
}


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _subset_metrics(target, weak, probability, prediction, mask, min_positive, min_negative, min_weak):
    selected = np.asarray(mask, dtype=bool)
    y = np.asarray(target, dtype=np.uint8)[selected]
    w = np.asarray(weak, dtype=bool)[selected]
    p = np.asarray(probability, dtype=np.float64)[selected]
    q = np.asarray(prediction, dtype=np.uint8)[selected]
    positive = y == 1
    negative = ~positive
    weak_positive = positive & w
    strong_positive = positive & ~w
    counts = {
        "sample": int(len(y)), "positive": int(positive.sum()), "negative": int(negative.sum()),
        "weak_positive": int(weak_positive.sum()), "strong_positive": int(strong_positive.sum()),
    }
    supported = counts["positive"] >= min_positive and counts["negative"] >= min_negative
    weak_supported = supported and counts["weak_positive"] >= min_weak
    weak_auroc = None
    if weak_supported:
        rank = weak_positive | negative
        weak_auroc = float(roc_auc_score(weak_positive[rank].astype(np.uint8), p[rank]))
    return {
        "counts": counts,
        "supported": bool(supported),
        "weak_supported": bool(weak_supported),
        "aupr": float(average_precision_score(y, p)) if supported else None,
        "weak_recall": _safe_div(int(q[weak_positive].sum()), counts["weak_positive"]),
        "strong_recall": _safe_div(int(q[strong_positive].sum()), counts["strong_positive"]),
        "negative_fpr": _safe_div(int(q[negative].sum()), counts["negative"]),
        "weak_auroc": weak_auroc,
    }


def _positive_bin_mask(target, inside):
    positive = np.asarray(target, dtype=np.uint8) == 1
    return (~positive) | (positive & np.asarray(inside, dtype=bool))


def _groups(labels, target_index, powers, values, parameter_mask, receiver):
    target = labels[:, target_index]
    positive = target == 1
    other = np.where(labels == 1, powers, -np.inf).copy()
    other[:, target_index] = -np.inf
    relative_gap = powers[:, target_index] - np.max(other, axis=1)
    relative_gap[labels.sum(axis=1) < 2] = np.nan
    output = {"overall/all": np.ones(len(labels), dtype=bool)}

    for name, low, high in (
        ("le_m15", -np.inf, -15.0), ("m15_m10", -15.0, -10.0),
        ("m10_m5", -10.0, -5.0), ("m5_0", -5.0, 0.0), ("gt_0", 0.0, np.inf),
    ):
        if np.isneginf(low):
            inside = relative_gap <= high
        elif np.isposinf(high):
            inside = relative_gap > low
        else:
            inside = (relative_gap > low) & (relative_gap <= high)
        output[f"relative_gap_db/{name}"] = _positive_bin_mask(target, positive & inside)

    jnr_column = {3: 27, 4: 48, 5: 52}[target_index]
    jnr = values[:, jnr_column]
    for name, low, high in (("10_15", 10, 15), ("15_20", 15, 20), ("20_25", 20, 25), ("25_30p1", 25, 30.1)):
        inside = parameter_mask[:, jnr_column] & np.isfinite(jnr) & (jnr >= low) & (jnr < high)
        output[f"jnr_db/{name}"] = _positive_bin_mask(target, positive & inside)

    if target_index == 3:
        column, scale = 28, 1.0
        bins = (("3_4", 3, 5), ("5_6", 5, 7), ("7_8", 7, 9))
        axis = "num_targets"
    elif target_index == 4:
        column, scale = 49, 1e-3
        bins = (("1_3khz", 1, 3), ("3_5khz", 3, 5), ("5_7p1khz", 5, 7.1))
        axis = "doppler_khz"
    else:
        column, scale = 53, 1e6
        bins = (("5_10us", 5, 10), ("10_15us", 10, 15), ("15_20p1us", 15, 20.1))
        axis = "sample_width_us"
    auxiliary = values[:, column] * scale
    for name, low, high in bins:
        inside = parameter_mask[:, column] & np.isfinite(auxiliary) & (auxiliary >= low) & (auxiliary < high)
        output[f"{axis}/{name}"] = _positive_bin_mask(target, positive & inside)

    order = labels.sum(axis=1)
    for value in range(1, 7):
        output[f"composition_order/{value}"] = order == value
    for index, name in enumerate(CLASS_NAMES):
        if index == target_index:
            continue
        output[f"co_label/{name}_present"] = labels[:, index] == 1
        output[f"co_label/{name}_absent"] = labels[:, index] == 0
    for cohort in range(5):
        output[f"receiver_cohort/{cohort}"] = receiver % 5 == cohort
    return output


def _cohen_dz(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return None
    return float(values.mean() / values.std(ddof=1))


def _wilcoxon(values):
    values = np.asarray(values, dtype=np.float64)
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) < 2:
        return {"statistic": None, "p_greater": None, "nonzero_count": int(len(nonzero))}
    result = wilcoxon(nonzero, alternative="greater", zero_method="wilcox", method="auto")
    return {"statistic": float(result.statistic), "p_greater": float(result.pvalue), "nonzero_count": int(len(nonzero))}


def _load_seed(root, seed):
    loaded = {}
    for variant in VARIANTS:
        path = root / f"seed_{seed}" / variant / "predictions.npz"
        loaded[variant] = {key: np.asarray(value) for key, value in np.load(path).items()}
        loaded[variant]["_path"] = path
    reference = loaded[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        for key in ("sample_ids", "receiver_noise_seed", "targets", "weak_mask"):
            if not np.array_equal(reference[key], loaded[variant][key]):
                raise AssertionError(f"{variant} {key} differs from U0")
    return loaded


def _run_seed(args, labels, powers, values, parameter_mask, receiver, seed):
    loaded = _load_seed(args.prediction_root, seed)
    metrics = {}
    for target_name, target_index in TARGETS.items():
        metrics[target_name] = {}
        groups = _groups(labels, target_index, powers, values, parameter_mask, receiver)
        for group_name, group_mask in groups.items():
            metrics[target_name][group_name] = {}
            for variant in VARIANTS:
                item = loaded[variant]
                metrics[target_name][group_name][variant] = _subset_metrics(
                    labels[:, target_index], item["weak_mask"][:, target_index],
                    item["probabilities"][:, target_index], item["predictions"][:, target_index],
                    group_mask, args.min_positive, args.min_negative, args.min_weak_positive,
                )
    preservation = {}
    for variant in (VARIANTS[2], VARIANTS[3]):
        preservation[variant] = {
            "sawfm_logit_max_abs": float(np.max(np.abs(loaded[variant]["logits"][:, SAW_INDEX] - loaded[VARIANTS[0]]["logits"][:, SAW_INDEX]))),
            "sawfm_probability_max_abs": float(np.max(np.abs(loaded[variant]["probabilities"][:, SAW_INDEX] - loaded[VARIANTS[0]]["probabilities"][:, SAW_INDEX]))),
            "sawfm_prediction_mismatch": int(np.sum(loaded[variant]["predictions"][:, SAW_INDEX] != loaded[VARIANTS[0]]["predictions"][:, SAW_INDEX])),
        }
    return {
        "seed": int(seed), "metrics": metrics, "preservation": preservation,
        "input_sha256": {variant: _sha256(loaded[variant]["_path"]) for variant in VARIANTS},
        "external_refit": False, "test_v3_used": False,
    }


def _aggregate(rows):
    output = {}
    for target_index, target_name in enumerate(TARGETS):
        output[target_name] = {}
        group_names = sorted(set.intersection(*[set(row["metrics"][target_name]) for row in rows]))
        for group_index, group_name in enumerate(group_names):
            group = {"variants": {}, "comparisons": {}}
            for variant_index, variant in enumerate(VARIANTS):
                group["variants"][variant] = {}
                for metric_index, metric in enumerate(METRICS):
                    values = [row["metrics"][target_name][group_name][variant][metric] for row in rows]
                    values = [value for value in values if value is not None]
                    group["variants"][variant][metric] = None if not values else _mean_ci(
                        values, 20349000 + target_index * 10000 + group_index * 100 + variant_index * 10 + metric_index
                    )
            for comparison_index, (comparison_name, (candidate, baseline)) in enumerate(COMPARISONS.items()):
                group["comparisons"][comparison_name] = {}
                for metric_index, metric in enumerate(METRICS):
                    deltas = []
                    per_seed = []
                    for row in rows:
                        candidate_value = row["metrics"][target_name][group_name][candidate][metric]
                        baseline_value = row["metrics"][target_name][group_name][baseline][metric]
                        if candidate_value is not None and baseline_value is not None:
                            delta = float(candidate_value - baseline_value)
                            deltas.append(delta)
                            per_seed.append({"seed": row["seed"], "delta": delta})
                    group["comparisons"][comparison_name][metric] = None if not deltas else {
                        **_mean_ci(deltas, 20350000 + target_index * 10000 + group_index * 100 + comparison_index * 10 + metric_index),
                        "per_seed": per_seed,
                        "positive_seeds": int(sum(value > 0 for value in deltas)),
                        "nonnegative_seeds": int(sum(value >= 0 for value in deltas)),
                        "cohen_dz": _cohen_dz(deltas),
                        "wilcoxon_greater": _wilcoxon(deltas),
                    }
            output[target_name][group_name] = group
    return output


def _entries(aggregate, comparison, metric, prefix=None, targets=None):
    rows = []
    for target_name in targets or TARGETS:
        for group_name, group in aggregate[target_name].items():
            if prefix is not None and not group_name.startswith(prefix):
                continue
            value = group["comparisons"][comparison][metric]
            if value is not None:
                rows.append((target_name, group_name, value))
    return rows


def _fraction(entries, nonnegative=False):
    if not entries:
        return 0.0
    return float(sum(item[2]["mean"] >= 0 if nonnegative else item[2]["mean"] > 0 for item in entries) / len(entries))


def _gate(aggregate, rows):
    weak_auroc = _entries(aggregate, "U2_vs_U0", "weak_auroc")
    weak_recall = _entries(aggregate, "U2_vs_U0", "weak_recall")
    overall = _entries(aggregate, "U2_vs_U0", "weak_auroc", "overall/")
    weakest = [entry for entry in _entries(aggregate, "U2_vs_U0", "weak_recall", "relative_gap_db/") if entry[1].endswith("le_m15") or entry[1].endswith("m15_m10")]
    jnr = _entries(aggregate, "U2_vs_U0", "weak_auroc", "jnr_db/")
    receiver = _entries(aggregate, "U2_vs_U0", "weak_auroc", "receiver_cohort/")
    isrj_u1 = _entries(aggregate, "U2_vs_U1", "weak_auroc", targets=("ISRJ",))
    isrj_u3 = _entries(aggregate, "U2_vs_U3", "weak_auroc", targets=("ISRJ",))
    composition = []
    for target_name in TARGETS:
        for group_name, group in aggregate[target_name].items():
            if not group_name.startswith("composition_order/"):
                continue
            fpr = group["variants"][VARIANTS[2]]["negative_fpr"]
            delta = group["comparisons"]["U2_vs_U0"]["negative_fpr"]
            if fpr is not None and delta is not None:
                composition.append((target_name, group_name, fpr["mean"], delta["mean"]))
    saw_aupr = [
        entry for entry in _entries(aggregate, "U2_vs_U0", "aupr", "co_label/")
        if entry[1] in ("co_label/SawFM_present", "co_label/SawFM_absent")
    ]
    saw_group_means = {}
    for suffix in ("present", "absent"):
        values = [entry[2]["mean"] for entry in saw_aupr if entry[1].endswith(suffix)]
        saw_group_means[suffix] = float(np.mean(values)) if values else float("-inf")
    preservation_safe = all(
        item[variant]["sawfm_logit_max_abs"] == 0.0
        and item[variant]["sawfm_probability_max_abs"] == 0.0
        and item[variant]["sawfm_prediction_mismatch"] == 0
        for item in (row["preservation"] for row in rows)
        for variant in (VARIANTS[2], VARIANTS[3])
    )
    diagnostics = {
        "u2_u0_positive_weak_auroc_fraction": _fraction(weak_auroc),
        "u2_u0_nonnegative_weak_recall_fraction": _fraction(weak_recall, nonnegative=True),
        "overall_target_weak_auroc_delta": {target: next(value["mean"] for name, _, value in overall if name == target) for target in TARGETS},
        "weakest_power_mean_weak_recall_gain": float(np.mean([entry[2]["mean"] for entry in weakest])) if weakest else float("-inf"),
        "positive_jnr_cells": int(sum(entry[2]["mean"] > 0 for entry in jnr)),
        "supported_jnr_cells": len(jnr),
        "positive_receiver_cells": int(sum(entry[2]["mean"] > 0 for entry in receiver)),
        "supported_receiver_cells": len(receiver),
        "isrj_u2_u1_positive_weak_auroc_fraction": _fraction(isrj_u1),
        "isrj_u2_u3_positive_weak_auroc_fraction": _fraction(isrj_u3),
        "worst_composition_fpr": max((item[2] for item in composition), default=float("inf")),
        "worst_composition_fpr_delta": max((item[3] for item in composition), default=float("inf")),
        "worst_composition_cell": max(composition, key=lambda item: item[2]) if composition else None,
        "sawfm_subset_target_mean_aupr_delta": saw_group_means,
    }
    checks = {
        "u2_u0_positive_weak_auroc_fraction_ge_0p70": diagnostics["u2_u0_positive_weak_auroc_fraction"] >= 0.70,
        "u2_u0_nonnegative_weak_recall_fraction_ge_0p70": diagnostics["u2_u0_nonnegative_weak_recall_fraction"] >= 0.70,
        "overall_each_target_weak_auroc_positive": all(value > 0 for value in diagnostics["overall_target_weak_auroc_delta"].values()),
        "weakest_power_mean_weak_recall_gain_ge_0p020": diagnostics["weakest_power_mean_weak_recall_gain"] >= 0.020,
        "jnr_nine_of_twelve_positive": diagnostics["supported_jnr_cells"] >= 12 and diagnostics["positive_jnr_cells"] >= 9,
        "receiver_twelve_of_fifteen_positive": diagnostics["supported_receiver_cells"] >= 15 and diagnostics["positive_receiver_cells"] >= 12,
        "isrj_u2_u1_positive_fraction_ge_0p70": diagnostics["isrj_u2_u1_positive_weak_auroc_fraction"] >= 0.70,
        "isrj_u2_u3_positive_fraction_ge_0p70": diagnostics["isrj_u2_u3_positive_weak_auroc_fraction"] >= 0.70,
        "worst_composition_fpr_le_0p30": diagnostics["worst_composition_fpr"] <= 0.30,
        "worst_composition_fpr_delta_le_0p05": diagnostics["worst_composition_fpr_delta"] <= 0.05,
        "sawfm_present_target_aupr_noninferior": saw_group_means["present"] >= -0.005,
        "sawfm_absent_target_aupr_noninferior": saw_group_means["absent"] >= -0.005,
        "sawfm_exact": preservation_safe,
        "external_refit_false": all(not row["external_refit"] for row in rows),
        "test_v3_boundary_safe": all(not row["test_v3_used"] for row in rows),
    }
    return {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()}, "diagnostics": diagnostics}


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["sha256"].lower() != EXPECTED_V7_SHA256:
        raise ValueError("development-v7 audit failed")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    data_path = Path(audit["development_v7_path"])
    labels = read_labels(data_path)
    values, parameter_mask = read_parameters(data_path)
    powers = _component_power(values)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    rows = [_run_seed(args, labels, powers, values, parameter_mask, receiver, seed) for seed in args.seeds]
    aggregate = _aggregate(rows)
    gate = _gate(aggregate, rows)
    result = {
        "status": "completed", "experiment": "RouteA_FullAlpha_Periodicity_V7_Stratified",
        "evidence_role": "frozen_development_v7_subgroup_audit",
        "development_v7_sha256": EXPECTED_V7_SHA256,
        "minimum_support": {"positive": args.min_positive, "negative": args.min_negative, "weak_positive": args.min_weak_positive},
        "rows": rows, "aggregate": aggregate, "robustness_gate": gate,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_fullalpha_periodicity_v7_stratified_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "gate": gate, "summary": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v7_generation/identity_audit.json")
    parser.add_argument("--prediction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_development_v7_frozen")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_v7_stratified")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--min-positive", type=int, default=40)
    parser.add_argument("--min-negative", type=int, default=100)
    parser.add_argument("--min-weak-positive", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
