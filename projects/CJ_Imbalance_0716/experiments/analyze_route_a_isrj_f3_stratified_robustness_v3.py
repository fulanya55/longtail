"""Frozen stratified robustness analysis for the F3 ISRJ fusion candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch  # Load Windows DLLs before scipy/sklearn imports.
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


ISRJ_INDEX = 5
OTHER_NAMES = ("FMB", "MFT", "VFT", "SawFM", "SinFM")
VARIANTS = {
    "F0": "F0_C4_Reconstructed",
    "F3": "F3_AlignedPeriodicityFusionHead",
    "F4": "F4_BlockShuffledFusionHead",
}
METRICS = ("aupr", "recall", "weak_recall", "strong_recall", "negative_fpr", "weak_auroc")


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _subset_metrics(target, weak, probability, prediction, mask, min_positive=20,
                    min_negative=50, min_weak_positive=10):
    selected = np.asarray(mask, dtype=bool)
    y = np.asarray(target, dtype=np.uint8)[selected]
    w = np.asarray(weak, dtype=bool)[selected]
    p = np.asarray(probability, dtype=np.float64)[selected]
    pred = np.asarray(prediction, dtype=np.uint8)[selected]
    positive = y == 1
    negative = y == 0
    weak_positive = positive & w
    strong_positive = positive & ~w
    counts = {
        "sample": int(len(y)), "positive": int(positive.sum()), "negative": int(negative.sum()),
        "weak_positive": int(weak_positive.sum()), "strong_positive": int(strong_positive.sum()),
    }
    supported = counts["positive"] >= min_positive and counts["negative"] >= min_negative
    weak_supported = supported and counts["weak_positive"] >= min_weak_positive
    result = {
        "counts": counts, "supported": bool(supported), "weak_supported": bool(weak_supported),
        "aupr": float(average_precision_score(y, p)) if supported else None,
        "recall": _safe_div(int(pred[positive].sum()), counts["positive"]),
        "weak_recall": _safe_div(int(pred[weak_positive].sum()), counts["weak_positive"]),
        "strong_recall": _safe_div(int(pred[strong_positive].sum()), counts["strong_positive"]),
        "negative_fpr": _safe_div(int(pred[negative].sum()), counts["negative"]),
        "weak_auroc": None,
    }
    if weak_supported:
        rank_mask = weak_positive | negative
        rank_target = weak_positive[rank_mask].astype(np.uint8)
        result["weak_auroc"] = float(roc_auc_score(rank_target, p[rank_mask]))
    return result


def _positive_bin_mask(target, positive_bin):
    target = np.asarray(target, dtype=np.uint8)
    return (target == 0) | ((target == 1) & np.asarray(positive_bin, dtype=bool))


def _combo_key(row):
    return "".join(str(int(value)) for value in row)


def _cohen_dz(values):
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return None
    std = float(array.std(ddof=1))
    if std <= 0:
        return None
    return float(array.mean() / std)


def _directional_wilcoxon(values):
    array = np.asarray(values, dtype=np.float64)
    nonzero = array[np.abs(array) > 1e-12]
    if len(nonzero) < 2:
        return {"statistic": None, "p_greater": None, "nonzero_count": int(len(nonzero))}
    result = wilcoxon(nonzero, alternative="greater", zero_method="wilcox", method="auto")
    return {"statistic": float(result.statistic), "p_greater": float(result.pvalue),
            "nonzero_count": int(len(nonzero))}


def _read_receiver_seeds(data_path):
    with h5py.File(str(data_path), "r") as handle:
        values = np.asarray(handle["receiver_noise_seed"][:, :], dtype=np.float64).reshape(-1)
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded):
        raise ValueError("receiver_noise_seed contains non-integral values")
    return rounded


def _load_predictions(root, seed):
    loaded = {}
    for short, directory in VARIANTS.items():
        path = root / f"seed_{seed}" / directory / "predictions.npz"
        loaded[short] = {key: np.asarray(value) for key, value in np.load(path).items()}
        loaded[short]["_path"] = path
    reference = loaded["F0"]
    for short in ("F3", "F4"):
        for key in ("sample_ids", "targets", "weak_mask"):
            if not np.array_equal(reference[key], loaded[short][key]):
                raise AssertionError(f"{short} {key} differs from F0")
    return loaded


def _strata(labels, target, weak, powers, values79, mask79, receiver_seeds,
            fit_labels):
    positive = target == 1
    other_power = powers.copy()
    other_power[:, ISRJ_INDEX] = -np.inf
    relative_gap = powers[:, ISRJ_INDEX] - np.max(np.where(labels == 1, other_power, -np.inf), axis=1)
    relative_gap[labels.sum(axis=1) < 2] = np.nan
    jnr = values79[:, get_parameter_specs("ISRJ")[0].column_index]
    width_us = values79[:, get_parameter_specs("ISRJ")[1].column_index] * 1e6
    order = labels.sum(axis=1)
    fit_combos = {_combo_key(row) for row in fit_labels}
    seen = np.asarray([_combo_key(row) in fit_combos for row in labels], dtype=bool)
    output = {}

    def add(axis, name, sample_mask):
        output[f"{axis}/{name}"] = np.asarray(sample_mask, dtype=bool)

    add("overall", "all", np.ones(len(target), dtype=bool))
    gap_bins = (("le_m15", -np.inf, -15.0, True), ("m15_m10", -15.0, -10.0, False),
                ("m10_m5", -10.0, -5.0, False), ("m5_0", -5.0, 0.0, False),
                ("gt_0", 0.0, np.inf, False))
    for name, low, high, include_low in gap_bins:
        inside = (relative_gap <= high) if np.isneginf(low) else ((relative_gap > low) & (relative_gap <= high))
        add("relative_gap_db", name, _positive_bin_mask(target, positive & inside))
    for name, low, high in (("10_15", 10, 15), ("15_20", 15, 20),
                            ("20_25", 20, 25), ("25_30p1", 25, 30.1)):
        valid = mask79[:, 52] & np.isfinite(jnr) & (jnr >= low) & (jnr < high)
        add("jnr_db", name, _positive_bin_mask(target, positive & valid))
    for name, low, high in (("5_10", 5, 10), ("10_15", 10, 15), ("15_20p1", 15, 20.1)):
        valid = mask79[:, 53] & np.isfinite(width_us) & (width_us >= low) & (width_us < high)
        add("sample_width_us", name, _positive_bin_mask(target, positive & valid))
    for value in range(1, 7):
        add("composition_order", str(value), order == value)
    for index, name in enumerate(OTHER_NAMES):
        add("co_label", f"{name}_present", labels[:, index] == 1)
        add("co_label", f"{name}_absent", labels[:, index] == 0)
    add("fit_combo", "seen", seen)
    add("fit_combo", "unseen", ~seen)
    for cohort in range(5):
        add("receiver_cohort", str(cohort), receiver_seeds % 5 == cohort)
    return output


def _run_seed(args, labels6, weak6, powers, values79, mask79, receiver_seeds, seed):
    loaded = _load_predictions(args.fusion_root, seed)
    ids = loaded["F0"]["sample_ids"].astype(np.int64)
    role_path = args.c6_root / f"seed_{seed}" / "role_indices.npz"
    roles = np.load(role_path)
    fit_ids = np.asarray(roles["fit"], dtype=np.int64)
    labels = labels6[ids]
    target = labels[:, ISRJ_INDEX]
    weak = weak6[ids, ISRJ_INDEX]
    groups = _strata(labels, target, weak, powers[ids], values79[ids], mask79[ids],
                     receiver_seeds[ids], labels6[fit_ids])
    metrics = {}
    for group_name, group_mask in groups.items():
        metrics[group_name] = {}
        for short in VARIANTS:
            metrics[group_name][short] = _subset_metrics(
                target, weak, loaded[short]["probabilities"][:, ISRJ_INDEX],
                loaded[short]["predictions"][:, ISRJ_INDEX], group_mask,
                args.min_positive, args.min_negative, args.min_weak_positive,
            )
    preservation = {}
    for short in ("F3", "F4"):
        preservation[short] = {
            "non_isrj_logit_max_abs": float(np.max(np.abs(loaded[short]["logits"][:, :ISRJ_INDEX] - loaded["F0"]["logits"][:, :ISRJ_INDEX]))),
            "non_isrj_probability_max_abs": float(np.max(np.abs(loaded[short]["probabilities"][:, :ISRJ_INDEX] - loaded["F0"]["probabilities"][:, :ISRJ_INDEX]))),
            "non_isrj_prediction_mismatch": int(np.sum(loaded[short]["predictions"][:, :ISRJ_INDEX] != loaded["F0"]["predictions"][:, :ISRJ_INDEX])),
        }
    return {
        "seed": int(seed), "metrics": metrics, "non_isrj_preservation": preservation,
        "input_sha256": {short: _sha256(loaded[short]["_path"]) for short in VARIANTS},
        "role_indices_sha256": _sha256(role_path), "test_v3_used": False,
    }


def _aggregate(rows):
    group_names = sorted(set.intersection(*[set(row["metrics"]) for row in rows]))
    groups = {}
    for group_index, group_name in enumerate(group_names):
        group = {"variants": {}, "comparisons": {}}
        for variant_index, short in enumerate(VARIANTS):
            group["variants"][short] = {}
            for metric_index, metric in enumerate(METRICS):
                values = [row["metrics"][group_name][short][metric] for row in rows
                          if row["metrics"][group_name][short][metric] is not None]
                group["variants"][short][metric] = (
                    _mean_ci(values, 20316000 + group_index * 100 + variant_index * 10 + metric_index)
                    if values else None
                )
        for comparison_index, (name, candidate, baseline) in enumerate((
            ("F3_vs_F0", "F3", "F0"), ("F3_vs_F4", "F3", "F4"),
        )):
            comparison = {}
            for metric_index, metric in enumerate(METRICS):
                values = []
                per_seed = []
                for row in rows:
                    c = row["metrics"][group_name][candidate][metric]
                    b = row["metrics"][group_name][baseline][metric]
                    if c is not None and b is not None:
                        delta = float(c - b)
                        values.append(delta)
                        per_seed.append({"seed": row["seed"], "delta": delta})
                comparison[metric] = None if not values else {
                    **_mean_ci(values, 20317000 + group_index * 100 + comparison_index * 10 + metric_index),
                    "per_seed": per_seed, "positive_seeds": int(sum(value > 0 for value in values)),
                    "nonnegative_seeds": int(sum(value >= 0 for value in values)),
                    "cohen_dz": _cohen_dz(values), "wilcoxon_greater": _directional_wilcoxon(values),
                }
            group["comparisons"][name] = comparison
        groups[group_name] = group
    return groups


def _supported_values(groups, comparison, metric, axis=None):
    rows = []
    for name, group in groups.items():
        if axis is not None and not name.startswith(axis + "/"):
            continue
        value = group["comparisons"][comparison][metric]
        if value is not None:
            rows.append((name, value))
    return rows


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
    labels6 = read_labels(data_path)
    values79, mask79 = read_parameters(data_path)
    powers = _component_power(values79)
    weak6 = compute_weak_component_mask(labels6, powers, args.weak_threshold_db)
    receiver_seeds = _read_receiver_seeds(data_path)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_ISRJ_F3_StratifiedRobustness_v3", "seeds": seeds,
        "frozen_predictions_only": True, "weak_threshold_db": args.weak_threshold_db,
        "minimum_support": {"positive": args.min_positive, "negative": args.min_negative,
                            "weak_positive": args.min_weak_positive},
        "locked_test_used": False, "test_v3_used": False,
    })
    rows = [_run_seed(args, labels6, weak6, powers, values79, mask79, receiver_seeds, seed) for seed in seeds]
    groups = _aggregate(rows)
    if args.smoke:
        gate = {"passed": False, "checks": {}, "evidence_level": "smoke_only"}
    else:
        f0_weak_auroc = _supported_values(groups, "F3_vs_F0", "weak_auroc")
        f0_weak_recall = _supported_values(groups, "F3_vs_F0", "weak_recall")
        f4_weak_auroc = _supported_values(groups, "F3_vs_F4", "weak_auroc")
        width = _supported_values(groups, "F3_vs_F0", "weak_auroc", "sample_width_us")
        jnr = _supported_values(groups, "F3_vs_F0", "weak_auroc", "jnr_db")
        receiver = _supported_values(groups, "F3_vs_F0", "weak_auroc", "receiver_cohort")
        weak_power = [row for row in _supported_values(groups, "F3_vs_F0", "weak_recall", "relative_gap_db")
                      if row[0].endswith("le_m15") or row[0].endswith("m15_m10")]
        order_fprs = [(name, groups[name]["variants"]["F3"]["negative_fpr"])
                      for name in groups if name.startswith("composition_order/") and groups[name]["variants"]["F3"]["negative_fpr"] is not None]
        worst_order_fpr = max((value["mean"] for _, value in order_fprs), default=float("inf"))
        general_prefixes = ("composition_order/", "co_label/", "fit_combo/", "receiver_cohort/")
        general_fprs = [(name, group["variants"]["F3"]["negative_fpr"]["mean"])
                        for name, group in groups.items()
                        if name.startswith(general_prefixes)
                        and group["variants"]["F3"]["negative_fpr"] is not None]
        worst_general_name, worst_general_fpr = (
            max(general_fprs, key=lambda item: item[1]) if general_fprs else (None, None)
        )
        preservation_safe = all(
            entry[short]["non_isrj_logit_max_abs"] == 0.0
            and entry[short]["non_isrj_probability_max_abs"] == 0.0
            and entry[short]["non_isrj_prediction_mismatch"] == 0
            for entry in (row["non_isrj_preservation"] for row in rows) for short in ("F3", "F4")
        )
        def fraction_positive(entries):
            return float(sum(value["mean"] > 0 for _, value in entries) / len(entries)) if entries else 0.0
        def fraction_nonnegative(entries):
            return float(sum(value["mean"] >= 0 for _, value in entries) / len(entries)) if entries else 0.0
        weak_power_gain = float(np.mean([value["mean"] for _, value in weak_power])) if weak_power else float("-inf")
        checks = {
            "f3_vs_f0_positive_weak_auroc_fraction_ge_0p80": fraction_positive(f0_weak_auroc) >= .80,
            "f3_vs_f0_nonnegative_weak_recall_fraction_ge_0p80": fraction_nonnegative(f0_weak_recall) >= .80,
            "f3_vs_f4_positive_weak_auroc_fraction_ge_0p70": fraction_positive(f4_weak_auroc) >= .70,
            "weak_power_layers_recall_gain_ge_0p030": weak_power_gain >= .030,
            "width_two_of_three_positive": sum(value["mean"] > 0 for _, value in width) >= min(2, len(width)) and len(width) >= 3,
            "jnr_three_of_four_positive": sum(value["mean"] > 0 for _, value in jnr) >= min(3, len(jnr)) and len(jnr) >= 4,
            "receiver_four_of_five_positive": sum(value["mean"] > 0 for _, value in receiver) >= min(4, len(receiver)) and len(receiver) >= 5,
            "worst_composition_order_fpr_le_0p25": worst_order_fpr <= .25,
            "non_isrj_outputs_identical": preservation_safe,
            "test_boundary_safe": all(not row["test_v3_used"] for row in rows),
        }
        gate = {
            "passed": bool(all(checks.values())), "checks": checks,
            "diagnostics": {
                "f3_vs_f0_positive_weak_auroc_fraction": fraction_positive(f0_weak_auroc),
                "f3_vs_f0_nonnegative_weak_recall_fraction": fraction_nonnegative(f0_weak_recall),
                "f3_vs_f4_positive_weak_auroc_fraction": fraction_positive(f4_weak_auroc),
                "weak_power_mean_recall_gain": weak_power_gain,
                "worst_composition_order_fpr": worst_order_fpr,
                "worst_general_subgroup_fpr_name": worst_general_name,
                "worst_general_subgroup_fpr": worst_general_fpr,
                "supported_jnr_weak_auroc_groups": [name for name, _ in jnr],
                "supported_width_weak_auroc_groups": [name for name, _ in width],
                "supported_group_counts": {"f0_weak_auroc": len(f0_weak_auroc),
                                           "f0_weak_recall": len(f0_weak_recall),
                                           "f4_weak_auroc": len(f4_weak_auroc)},
            },
            "evidence_level": "five_seed_development_v3_frozen_stratified_robustness",
        }
    payload = {
        "status": "completed", "experiment": "RouteA_ISRJ_F3_StratifiedRobustness_v3",
        "smoke": bool(args.smoke), "seeds": seeds, "data_sha256": data_sha,
        "rows": rows, "groups": groups, "robustness_gate": gate,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    path = args.output_root / "route_a_isrj_f3_stratified_robustness_v3_summary.json"
    _write_json(path, payload)
    print(json.dumps({"status": payload["status"], "smoke": payload["smoke"],
                      "robustness_gate": gate, "summary": str(path)}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--fusion-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_f3_stratified_robustness_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=50)
    parser.add_argument("--min-weak-positive", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
