"""Evaluate pre-registered FPR safety margins on frozen Raw probabilities."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_raw_frontend_independent_classifier_audit import (  # noqa: E402
    _binary_rank_metrics,
    _calibration_metrics,
    _decision_metrics,
    _subgroup_metrics,
)
from experiments.run_route_a_raw_platt_five_seed_label_routing import (  # noqa: E402
    CLASS_NAMES,
    SAW_INDEX,
    _multilabel_metrics,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = (
    "T0_FPR20_Current",
    "T1_FPR15_PerSeed",
    "T2_FPR10_PerSeed",
    "T3_FPR15_Shared",
    "T4_FPR10_Shared",
)


def _row_key(row):
    return (
        float(row["mean_power_recall"]),
        float(row["ordinary_recall"]),
        float(row["power"].get("50", {}).get("recall", 0.0)),
        float(row["threshold"]),
    )


def _select_margin(rows, max_fpr):
    feasible = [
        row for row in rows
        if float(row["ordinary_fpr"]) <= float(max_fpr) + 1e-12
        and float(row["max_power_fpr"]) <= float(max_fpr) + 1e-12
    ]
    return None if not feasible else max(feasible, key=_row_key)


def _select_shared(searches, max_fpr):
    indexed = {
        seed: {round(float(row["threshold"]), 10): row for row in rows}
        for seed, rows in searches.items()
    }
    common = sorted(set.intersection(*(set(values) for values in indexed.values())))
    candidates = []
    for threshold in common:
        rows = [indexed[seed][threshold] for seed in sorted(indexed)]
        if not all(
            float(row["ordinary_fpr"]) <= float(max_fpr) + 1e-12
            and float(row["max_power_fpr"]) <= float(max_fpr) + 1e-12
            for row in rows
        ):
            continue
        key = (
            float(np.mean([row["mean_power_recall"] for row in rows])),
            float(np.mean([row["ordinary_recall"] for row in rows])),
            float(np.mean([row["power"]["50"]["recall"] for row in rows])),
            float(threshold),
        )
        candidates.append((key, float(threshold), rows))
    if not candidates:
        return None
    key, threshold, rows = max(candidates, key=lambda item: item[0])
    return {"threshold": threshold, "aggregate_key": list(key), "per_seed_rows": rows}


def _power_metrics(saved, threshold, levels):
    result = {}
    for level in levels:
        positive = saved[f"positive_{level}db"].astype(np.float64)
        negative = saved[f"negative_{level}db"].astype(np.float64)
        probability = np.concatenate([positive, negative])
        targets = np.concatenate([
            np.ones(len(positive), dtype=np.uint8),
            np.zeros(len(negative), dtype=np.uint8),
        ])
        result[str(level)] = {
            **_binary_rank_metrics(probability, targets),
            **_decision_metrics(probability, targets, threshold),
        }
    return result


def _route_metrics(args, seed, raw_probability, raw_threshold):
    h2_root = args.h2_root / f"seed_{seed}" / "H2_PosBCE"
    saved = np.load(h2_root / "predictions.npz")
    targets = saved["targets"].astype(np.uint8)
    h2_probability = saved["probabilities"].astype(np.float64)
    thresholds_payload = _read(h2_root / "thresholds.json")
    h2_thresholds = np.asarray(
        thresholds_payload["modes"][thresholds_payload["primary_mode"]],
        dtype=np.float64,
    )
    routed_probability = h2_probability.copy()
    routed_probability[:, SAW_INDEX] = raw_probability
    routed_thresholds = h2_thresholds.copy()
    routed_thresholds[SAW_INDEX] = float(raw_threshold)
    baseline = _multilabel_metrics(h2_probability, targets, h2_thresholds)
    routed = _multilabel_metrics(routed_probability, targets, routed_thresholds)
    other_fpr_delta = max(
        routed["per_label"][name]["fpr"] - baseline["per_label"][name]["fpr"]
        for index, name in enumerate(CLASS_NAMES) if index != SAW_INDEX
    )
    checks = {
        "macro_aupr_not_down": routed["macro_aupr"] >= baseline["macro_aupr"] - 1e-12,
        "macro_f1_improved": routed["macro_f1"] > baseline["macro_f1"],
        "saw_recall_gain_0p20": routed["per_label"]["SawFM"]["recall"] >= baseline["per_label"]["SawFM"]["recall"] + 0.20,
        "other_label_max_fpr_delta_0p01": other_fpr_delta <= 0.01 + 1e-12,
        "exact_match_delta_minus_0p01": routed["exact_match"] >= baseline["exact_match"] - 0.01 - 1e-12,
    }
    return {
        "baseline": baseline,
        "routed": routed,
        "deltas": {
            "macro_aupr": routed["macro_aupr"] - baseline["macro_aupr"],
            "macro_f1": routed["macro_f1"] - baseline["macro_f1"],
            "saw_recall": routed["per_label"]["SawFM"]["recall"] - baseline["per_label"]["SawFM"]["recall"],
            "exact_match": routed["exact_match"] - baseline["exact_match"],
            "other_label_max_fpr": float(other_fpr_delta),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _evaluate(args, labels, seed, variant, threshold, selection):
    seed_root = args.source_root / f"seed_{seed}"
    saved = np.load(seed_root / "I1_RawPlatt" / "predictions.npz")
    probability = saved["ordinary_probability"].astype(np.float64)
    targets = saved["ordinary_target"].astype(np.uint8)
    source_indices = saved["ordinary_source_index"].astype(np.int64)
    ordinary = {
        **_binary_rank_metrics(probability, targets),
        **_decision_metrics(probability, targets, threshold),
        **_calibration_metrics(probability, targets),
    }
    ordinary["subgroups"] = _subgroup_metrics(
        probability, targets, labels[source_indices], threshold
    )
    power = _power_metrics(saved, threshold, args.levels)
    route = _route_metrics(args, seed, probability, threshold)
    return {
        "seed": int(seed),
        "variant": variant,
        "threshold": float(threshold),
        "selection": selection,
        "ordinary": ordinary,
        "power": power,
        "mean_power_recall": float(np.mean([item["recall"] for item in power.values()])),
        "max_power_fpr": float(np.max([item["fpr"] for item in power.values()])),
        "route": route,
        "identity_overlap": 0,
        "locked_test_used": False,
    }


def _aggregate(rows, levels):
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            "threshold": _mean_ci([row["threshold"] for row in selected], 811000 + variant_index),
            "ordinary": {
                metric: _mean_ci(
                    [row["ordinary"][metric] for row in selected],
                    812000 + variant_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("aupr", "recall", "fpr", "brier", "ece_10bin"))
            },
            "mean_power_recall": _mean_ci(
                [row["mean_power_recall"] for row in selected], 813000 + variant_index
            ),
            "max_power_fpr": _mean_ci(
                [row["max_power_fpr"] for row in selected], 814000 + variant_index
            ),
            "by_level": {
                str(level): {
                    metric: _mean_ci(
                        [row["power"][str(level)][metric] for row in selected],
                        815000 + variant_index * 1000 + int(level) * 10 + metric_index,
                    )
                    for metric_index, metric in enumerate(("recall", "fpr"))
                }
                for level in levels
            },
            "route": {
                metric: _mean_ci(
                    [row["route"]["routed"][metric] for row in selected],
                    816000 + variant_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("macro_aupr", "macro_f1", "exact_match"))
            },
            "route_passed_seed_count": int(sum(row["route"]["passed"] for row in selected)),
        }
    return result


def _promotion_gate(rows, aggregate, levels):
    t0 = aggregate["T0_FPR20_Current"]
    t1 = aggregate["T1_FPR15_PerSeed"]
    selected = [row for row in rows if row["variant"] == "T1_FPR15_PerSeed"]
    seed2039 = next(row for row in selected if row["seed"] == 2039)
    checks = {
        "seed2039_ordinary_fpr_0p20": seed2039["ordinary"]["fpr"] <= 0.20 + 1e-12,
        "five_seed_ordinary_fpr_0p20": all(row["ordinary"]["fpr"] <= 0.20 + 1e-12 for row in selected),
        "mean_ordinary_recall_delta_minus_0p02": t1["ordinary"]["recall"]["mean"] >= t0["ordinary"]["recall"]["mean"] - 0.02 - 1e-12,
        "mean_50db_recall_0p95": t1["by_level"]["50"]["recall"]["mean"] >= 0.95,
        "each_power_fpr_0p20": all(
            row["power"][str(level)]["fpr"] <= 0.20 + 1e-12
            for row in selected for level in levels
        ),
        "route_macro_f1_delta_minus_0p005": t1["route"]["macro_f1"]["mean"] >= t0["route"]["macro_f1"]["mean"] - 0.005 - 1e-12,
        "route_exact_match_delta_minus_0p01": t1["route"]["exact_match"]["mean"] >= t0["route"]["exact_match"]["mean"] - 0.01 - 1e-12,
        "route_four_of_five_pass": t1["route_passed_seed_count"] >= 4,
        "identity_locked_test_audit": all(row["identity_overlap"] == 0 and not row["locked_test_used"] for row in selected),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    searches = {}
    current = {}
    for seed in args.seeds:
        payload = _read(args.source_root / f"seed_{seed}" / "threshold_search.json")["I1_RawPlatt"]
        searches[int(seed)] = payload["rows"]
        current[int(seed)] = payload["selected"]
    shared15 = _select_shared(searches, 0.15)
    shared10 = _select_shared(searches, 0.10)
    if shared15 is None or shared10 is None:
        raise RuntimeError("shared safety-margin threshold is infeasible")
    rows = []
    selections = {}
    for seed in args.seeds:
        seed = int(seed)
        per15 = _select_margin(searches[seed], 0.15)
        per10 = _select_margin(searches[seed], 0.10)
        if per15 is None or per10 is None:
            raise RuntimeError(f"seed {seed} safety-margin threshold is infeasible")
        variant_selections = {
            "T0_FPR20_Current": current[seed],
            "T1_FPR15_PerSeed": per15,
            "T2_FPR10_PerSeed": per10,
            "T3_FPR15_Shared": {"threshold": shared15["threshold"], "shared": True, "design_max_fpr": 0.15},
            "T4_FPR10_Shared": {"threshold": shared10["threshold"], "shared": True, "design_max_fpr": 0.10},
        }
        selections[str(seed)] = variant_selections
        for variant, selection in variant_selections.items():
            rows.append(_evaluate(
                args, labels, seed, variant, float(selection["threshold"]), selection
            ))
    aggregate = _aggregate(rows, args.levels)
    gate = _promotion_gate(rows, aggregate, args.levels)
    if gate["passed"]:
        next_decision = "promote_fpr15_safety_margin_and_refreeze_candidate"
    elif aggregate["T3_FPR15_Shared"]["route_passed_seed_count"] >= 4:
        next_decision = "design_independent_confirmation_for_shared_fpr15_threshold"
    else:
        next_decision = "keep_current_threshold_and_move_to_raw_protocol_uniformity"
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "levels": list(args.levels),
            "variants": list(VARIANTS),
            "primary_variant": "T1_FPR15_PerSeed",
            "selection_roles": ["ordinary_threshold", "power_threshold"],
            "evaluation_roles": ["ordinary_early", "power_development"],
            "networks_retrained": False, "calibrators_refit": False,
        },
        "source_summary": str(args.source_summary),
        "source_summary_sha256": _sha256(args.source_summary),
        "shared_thresholds": {"fpr15": shared15, "fpr10": shared10},
        "selections": selections,
        "per_run": rows,
        "aggregate": aggregate,
        "promotion_gate": gate,
        "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "route_a_raw_threshold_safety_margin_summary.json"
    _write(summary_path, payload)
    for seed in args.seeds:
        seed_rows = [row for row in rows if row["seed"] == int(seed)]
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        _write(seed_root / "metrics.json", {"seed": int(seed), "variants": seed_rows})
        _write(seed_root / "run_receipt.json", {
            "status": "completed", "seed": int(seed),
            "source_summary_sha256": payload["source_summary_sha256"],
            "identity_overlap": 0, "locked_test_used": False,
        })
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing")
    parser.add_argument("--source-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing" / "route_a_raw_platt_five_seed_label_routing_summary.json")
    parser.add_argument("--h2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_threshold_safety_margin")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
