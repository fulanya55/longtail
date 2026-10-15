"""Cross-seed stress test for frozen Raw SawFM thresholds."""

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
from experiments.run_route_a_raw_threshold_safety_margin import (  # noqa: E402
    _evaluate,
    _select_shared,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = ("C0_Current", "C1_LOSO_FPR15_Shared", "C2_LOSO_FPR10_Shared")


def _loso_selections(searches, max_fpr):
    result = {}
    for held_out in sorted(searches):
        sources = {seed: rows for seed, rows in searches.items() if seed != held_out}
        selected = _select_shared(sources, max_fpr)
        if selected is None:
            raise RuntimeError(f"held-out seed {held_out}: no shared threshold for FPR {max_fpr}")
        result[held_out] = {
            "threshold": float(selected["threshold"]),
            "held_out_seed": int(held_out),
            "source_seeds": sorted(sources),
            "design_max_fpr": float(max_fpr),
            "source_aggregate_key": selected["aggregate_key"],
        }
    return result


def _aggregate(rows, levels):
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            "threshold": _mean_ci([row["threshold"] for row in selected], 821000 + variant_index),
            "ordinary": {
                metric: _mean_ci(
                    [row["ordinary"][metric] for row in selected],
                    822000 + variant_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("aupr", "recall", "fpr", "brier", "ece_10bin"))
            },
            "by_level": {
                str(level): {
                    metric: _mean_ci(
                        [row["power"][str(level)][metric] for row in selected],
                        823000 + variant_index * 1000 + int(level) * 10 + metric_index,
                    )
                    for metric_index, metric in enumerate(("recall", "fpr"))
                }
                for level in levels
            },
            "route": {
                metric: _mean_ci(
                    [row["route"]["routed"][metric] for row in selected],
                    824000 + variant_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("macro_aupr", "macro_f1", "exact_match"))
            },
            "route_passed_seed_count": int(sum(row["route"]["passed"] for row in selected)),
        }
    return result


def _promotion_gate(rows, aggregate, levels):
    baseline = aggregate["C0_Current"]
    primary = aggregate["C1_LOSO_FPR15_Shared"]
    selected = [row for row in rows if row["variant"] == "C1_LOSO_FPR15_Shared"]
    thresholds = [row["threshold"] for row in selected]
    checks = {
        "five_seed_ordinary_fpr_0p20": all(row["ordinary"]["fpr"] <= 0.20 + 1e-12 for row in selected),
        "mean_ordinary_fpr_0p16": primary["ordinary"]["fpr"]["mean"] <= 0.16 + 1e-12,
        "mean_ordinary_recall_0p84": primary["ordinary"]["recall"]["mean"] >= 0.84 - 1e-12,
        "mean_50db_recall_0p95": primary["by_level"]["50"]["recall"]["mean"] >= 0.95 - 1e-12,
        "each_power_fpr_0p20": all(
            row["power"][str(level)]["fpr"] <= 0.20 + 1e-12
            for row in selected for level in levels
        ),
        "route_macro_f1_delta_minus_0p005": primary["route"]["macro_f1"]["mean"] >= baseline["route"]["macro_f1"]["mean"] - 0.005 - 1e-12,
        "route_exact_match_delta_minus_0p01": primary["route"]["exact_match"]["mean"] >= baseline["route"]["exact_match"]["mean"] - 0.01 - 1e-12,
        "route_four_of_five_pass": primary["route_passed_seed_count"] >= 4,
        "threshold_range_0p05": max(thresholds) - min(thresholds) <= 0.05 + 1e-12,
        "identity_locked_test_audit": all(row["identity_overlap"] == 0 and not row["locked_test_used"] for row in selected),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    searches, current = {}, {}
    for seed in args.seeds:
        payload = _read(args.source_root / f"seed_{seed}" / "threshold_search.json")["I1_RawPlatt"]
        searches[int(seed)] = payload["rows"]
        current[int(seed)] = payload["selected"]
    loso15 = _loso_selections(searches, 0.15)
    loso10 = _loso_selections(searches, 0.10)
    rows = []
    for seed in args.seeds:
        seed = int(seed)
        selections = {
            "C0_Current": current[seed],
            "C1_LOSO_FPR15_Shared": loso15[seed],
            "C2_LOSO_FPR10_Shared": loso10[seed],
        }
        for variant, selection in selections.items():
            rows.append(_evaluate(args, labels, seed, variant, float(selection["threshold"]), selection))
    aggregate = _aggregate(rows, args.levels)
    gate = _promotion_gate(rows, aggregate, args.levels)
    if gate["passed"]:
        next_decision = "refreeze_cross_seed_stable_threshold_candidate"
    elif gate["checks"]["five_seed_ordinary_fpr_0p20"]:
        next_decision = "keep_current_candidate_and_unify_raw_training_protocol"
    else:
        next_decision = "require_new_split_calibration_confirmation"
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "levels": list(args.levels), "variants": list(VARIANTS),
            "primary_variant": "C1_LOSO_FPR15_Shared",
            "held_out_seed_excluded_from_threshold_selection": True,
            "networks_retrained": False, "calibrators_refit": False,
            "evidence_scope": "development_cross_seed_stress_test_not_independent_locked_test",
        },
        "source_summary": str(args.source_summary),
        "source_summary_sha256": _sha256(args.source_summary),
        "loso_selections": {"fpr15": loso15, "fpr10": loso10},
        "per_run": rows,
        "aggregate": aggregate,
        "promotion_gate": gate,
        "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "route_a_raw_cross_seed_threshold_summary.json"
    _write(summary_path, payload)
    _write(args.output_root / "run_receipt.json", {
        "status": "completed", "summary_sha256": _sha256(summary_path),
        "source_summary_sha256": payload["source_summary_sha256"],
        "held_out_seed_excluded": True, "locked_test_used": False,
    })
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing")
    parser.add_argument("--source-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing" / "route_a_raw_platt_five_seed_label_routing_summary.json")
    parser.add_argument("--h2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_cross_seed_threshold")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

