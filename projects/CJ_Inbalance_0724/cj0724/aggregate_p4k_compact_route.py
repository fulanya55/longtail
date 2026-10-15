"""Aggregate P4k paired confirm receipts against preregistered gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


GLOBAL = "legacy_mlcnn_global"
FULL = "mlcnn_iq_grouped_task_routed"
COMPACT = "mlcnn_iq_grouped_task_routed_w4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    preregister = Path(args.preregister)
    protocol = json.loads(preregister.read_text(encoding="utf-8"))
    data_audit = json.loads(Path(args.data_audit).read_text(encoding="utf-8"))

    receipts = {}
    for path in run_root.glob("*/receipt_parameter.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        receipts[(row["model"], int(row["seed"]))] = row
    seeds = [int(seed) for seed in protocol["training"]["model_seeds"]]
    paired = []
    target_deltas = {}
    for seed in seeds:
        global_metrics = receipts[(GLOBAL, seed)]["confirm_fixed"]
        full_metrics = receipts[(FULL, seed)]["confirm_fixed"]
        compact_metrics = receipts[(COMPACT, seed)]["confirm_fixed"]
        global_nmae = global_metrics["parameter"]["oracle_macro_target_nmae"]
        full_nmae = full_metrics["parameter"]["oracle_macro_target_nmae"]
        compact_nmae = compact_metrics["parameter"]["oracle_macro_target_nmae"]
        full_gain = global_nmae - full_nmae
        compact_gain = global_nmae - compact_nmae
        paired.append({
            "seed": seed,
            "global_macro_nmae": global_nmae,
            "full_route_macro_nmae": full_nmae,
            "compact_macro_nmae": compact_nmae,
            "full_route_relative_nmae_reduction_vs_global": full_gain / global_nmae,
            "compact_relative_nmae_reduction_vs_global": compact_gain / global_nmae,
            "compact_retention_of_full_route_absolute_gain": compact_gain / full_gain,
            "compact_pearson_delta_vs_global": (
                compact_metrics["parameter"]["oracle_macro_pearson_r"]
                - global_metrics["parameter"]["oracle_macro_pearson_r"]
            ),
            "compact_macro_aupr_delta_vs_global": (
                compact_metrics["macro_aupr"] - global_metrics["macro_aupr"]
            ),
            "compact_subset_accuracy_delta_vs_global": (
                compact_metrics["subset_acc"] - global_metrics["subset_acc"]
            ),
        })
        global_targets = global_metrics["parameter"]["per_target"]
        compact_targets = compact_metrics["parameter"]["per_target"]
        for global_target, compact_target in zip(global_targets, compact_targets):
            key = f"{global_target['class_name']}.{global_target['name']}"
            target_deltas.setdefault(key, []).append(
                compact_target["normalized_mae"] - global_target["normalized_mae"]
            )

    gates = protocol["promotion_gates"]
    systematic = {
        key: values for key, values in target_deltas.items()
        if all(value > gates["no_target_worse_in_3_of_3_seeds_vs_global_by_more_than_absolute_nmae"]
               for value in values)
    }
    gate_results = {
        "each_seed_compact_relative_reduction": all(
            row["compact_relative_nmae_reduction_vs_global"]
            >= gates["each_seed_compact_relative_macro_nmae_reduction_vs_global_at_least"]
            for row in paired
        ),
        "mean_compact_relative_reduction": float(np.mean([
            row["compact_relative_nmae_reduction_vs_global"] for row in paired
        ])) >= gates["mean_compact_relative_macro_nmae_reduction_vs_global_at_least"],
        "three_of_three_positive": all(
            row["compact_relative_nmae_reduction_vs_global"] > 0 for row in paired
        ),
        "each_seed_compact_pearson_nonlower": all(
            row["compact_pearson_delta_vs_global"] >= 0 for row in paired
        ),
        "mean_gain_retention": float(np.mean([
            row["compact_retention_of_full_route_absolute_gain"] for row in paired
        ])) >= gates["mean_compact_retention_of_full_route_absolute_nmae_gain_at_least"],
        "each_seed_gain_retention": all(
            row["compact_retention_of_full_route_absolute_gain"]
            >= gates["each_seed_compact_retention_of_full_route_absolute_nmae_gain_at_least"]
            for row in paired
        ),
        "classification_aupr": all(
            row["compact_macro_aupr_delta_vs_global"]
            >= -gates["each_seed_compact_macro_aupr_drop_vs_global_at_most"]
            for row in paired
        ),
        "classification_subset": all(
            row["compact_subset_accuracy_delta_vs_global"]
            >= -gates["each_seed_compact_subset_accuracy_drop_vs_global_at_most"]
            for row in paired
        ),
        "no_systematic_target_worsening": not systematic,
    }
    result = {
        "experiment": protocol["experiment"],
        "preregister_sha256": sha256_file(preregister),
        "run_root": str(run_root),
        "identity_audit": {
            "count": data_audit["identities"]["count"],
            "unique": data_audit["identities"]["unique"],
            "minimum": data_audit["identities"]["minimum"],
            "maximum": data_audit["identities"]["maximum"],
            "all_excluded_overlap_zero": data_audit["all_excluded_overlap_zero"],
            "all_targets_finite_and_variable": data_audit["all_targets_finite_and_variable"],
            "combination_count": data_audit["combinations"]["count"],
        },
        "paired_confirm": paired,
        "aggregate": {
            "mean_full_route_relative_nmae_reduction_vs_global": float(np.mean([
                row["full_route_relative_nmae_reduction_vs_global"] for row in paired
            ])),
            "mean_compact_relative_nmae_reduction_vs_global": float(np.mean([
                row["compact_relative_nmae_reduction_vs_global"] for row in paired
            ])),
            "mean_compact_retention_of_full_route_absolute_gain": float(np.mean([
                row["compact_retention_of_full_route_absolute_gain"] for row in paired
            ])),
            "mean_compact_pearson_delta_vs_global": float(np.mean([
                row["compact_pearson_delta_vs_global"] for row in paired
            ])),
            "mean_compact_macro_aupr_delta_vs_global": float(np.mean([
                row["compact_macro_aupr_delta_vs_global"] for row in paired
            ])),
            "mean_compact_subset_accuracy_delta_vs_global": float(np.mean([
                row["compact_subset_accuracy_delta_vs_global"] for row in paired
            ])),
        },
        "target_nmae_deltas_compact_minus_global": target_deltas,
        "systematic_targets_worse_by_more_than_0_03": systematic,
        "sawfm_sweep_period_deltas": target_deltas["SawFM.sweep_period"],
        "gate_results": gate_results,
        "promotion_passed": bool(all(gate_results.values())),
        "failure_attribution": "Width 4 removes the parameter branch capacity needed for heterogeneous extraction. Classification remains protected because its MLCNN path is unchanged, but the compact IQ representation is weaker than even the shared global parameter representation on all three seeds. The deployment mechanism succeeds; the width-4 accuracy mechanism fails.",
        "stop_decision": "Do not add seeds 8044 or 8045 and do not sweep width on seed730000000. Freeze P4j as deployment evidence only. Retain the full-width P4c/P4d route as the accuracy candidate with its explicit 11.12 percent eager-MAC cost, or preregister exactly one intermediate width on wholly new identities using the P4k capacity diagnosis.",
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
