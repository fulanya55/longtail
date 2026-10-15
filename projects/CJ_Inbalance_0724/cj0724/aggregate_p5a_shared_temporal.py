"""Aggregate P5a paired confirm receipts against frozen accuracy gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


GLOBAL = "legacy_mlcnn_global"
FULL = "mlcnn_iq_grouped_task_routed"
DEFAULT_CANDIDATE = "mlcnn_shared_temporal_task_routed"


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
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    preregister = Path(args.preregister)
    protocol = json.loads(preregister.read_text(encoding="utf-8"))
    data_audit = json.loads(Path(args.data_audit).read_text(encoding="utf-8"))
    candidate_name = str(args.candidate)
    receipts = {}
    for path in run_root.glob("*/receipt_parameter.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        receipts[(row["model"], int(row["seed"]))] = row

    seeds = [int(seed) for seed in protocol["training_if_cost_passes"]["model_seeds"]]
    paired = []
    target_deltas: dict[str, list[float]] = {}
    for seed in seeds:
        global_metrics = receipts[(GLOBAL, seed)]["confirm_fixed"]
        full_metrics = receipts[(FULL, seed)]["confirm_fixed"]
        candidate_metrics = receipts[(candidate_name, seed)]["confirm_fixed"]
        global_nmae = global_metrics["parameter"]["oracle_macro_target_nmae"]
        full_nmae = full_metrics["parameter"]["oracle_macro_target_nmae"]
        candidate_nmae = candidate_metrics["parameter"]["oracle_macro_target_nmae"]
        full_gain = global_nmae - full_nmae
        candidate_gain = global_nmae - candidate_nmae
        paired.append({
            "seed": seed,
            "global_macro_nmae": global_nmae,
            "full_route_macro_nmae": full_nmae,
            "candidate_macro_nmae": candidate_nmae,
            "full_route_relative_nmae_reduction_vs_global": full_gain / global_nmae,
            "candidate_relative_nmae_reduction_vs_global": candidate_gain / global_nmae,
            "candidate_retention_of_full_route_absolute_gain": candidate_gain / full_gain,
            "candidate_pearson_delta_vs_global": (
                candidate_metrics["parameter"]["oracle_macro_pearson_r"]
                - global_metrics["parameter"]["oracle_macro_pearson_r"]
            ),
            "candidate_macro_aupr_delta_vs_global": (
                candidate_metrics["macro_aupr"] - global_metrics["macro_aupr"]
            ),
            "candidate_subset_accuracy_delta_vs_global": (
                candidate_metrics["subset_acc"] - global_metrics["subset_acc"]
            ),
        })
        for global_target, candidate_target in zip(
            global_metrics["parameter"]["per_target"],
            candidate_metrics["parameter"]["per_target"],
        ):
            key = f"{global_target['class_name']}.{global_target['name']}"
            target_deltas.setdefault(key, []).append(
                candidate_target["normalized_mae"] - global_target["normalized_mae"]
            )

    gates = protocol["promotion_gates"]
    systematic = {
        key: values for key, values in target_deltas.items()
        if all(value > gates[
            "no_target_worse_in_3_of_3_seeds_vs_global_by_more_than_absolute_nmae"
        ] for value in values)
    }
    gate_results = {
        "each_seed_relative_reduction": all(
            row["candidate_relative_nmae_reduction_vs_global"]
            >= gates["each_seed_candidate_relative_macro_nmae_reduction_vs_global_at_least"]
            for row in paired
        ),
        "mean_relative_reduction": float(np.mean([
            row["candidate_relative_nmae_reduction_vs_global"] for row in paired
        ])) >= gates["mean_candidate_relative_macro_nmae_reduction_vs_global_at_least"],
        "three_of_three_positive": all(
            row["candidate_relative_nmae_reduction_vs_global"] > 0 for row in paired
        ),
        "each_seed_pearson_nonlower": all(
            row["candidate_pearson_delta_vs_global"] >= 0 for row in paired
        ),
        "mean_gain_retention": float(np.mean([
            row["candidate_retention_of_full_route_absolute_gain"] for row in paired
        ])) >= gates["mean_candidate_retention_of_full_route_absolute_nmae_gain_at_least"],
        "each_seed_gain_retention": all(
            row["candidate_retention_of_full_route_absolute_gain"]
            >= gates["each_seed_candidate_retention_of_full_route_absolute_nmae_gain_at_least"]
            for row in paired
        ),
        "classification_aupr": all(
            row["candidate_macro_aupr_delta_vs_global"]
            >= -gates["each_seed_candidate_macro_aupr_drop_vs_global_at_most"]
            for row in paired
        ),
        "classification_subset": all(
            row["candidate_subset_accuracy_delta_vs_global"]
            >= -gates["each_seed_candidate_subset_accuracy_drop_vs_global_at_most"]
            for row in paired
        ),
        "no_systematic_target_worsening": not systematic,
    }
    result = {
        "experiment": protocol["experiment"],
        "candidate": candidate_name,
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
            "mean_candidate_relative_nmae_reduction_vs_global": float(np.mean([
                row["candidate_relative_nmae_reduction_vs_global"] for row in paired
            ])),
            "mean_candidate_retention_of_full_route_absolute_gain": float(np.mean([
                row["candidate_retention_of_full_route_absolute_gain"] for row in paired
            ])),
            "mean_candidate_pearson_delta_vs_global": float(np.mean([
                row["candidate_pearson_delta_vs_global"] for row in paired
            ])),
            "mean_candidate_macro_aupr_delta_vs_global": float(np.mean([
                row["candidate_macro_aupr_delta_vs_global"] for row in paired
            ])),
            "mean_candidate_subset_accuracy_delta_vs_global": float(np.mean([
                row["candidate_subset_accuracy_delta_vs_global"] for row in paired
            ])),
        },
        "target_nmae_deltas_candidate_minus_global": target_deltas,
        "systematic_targets_worse_by_more_than_0_03": systematic,
        "sawfm_sweep_period_deltas": target_deltas["SawFM.sweep_period"],
        "gate_results": gate_results,
        "promotion_passed": bool(all(gate_results.values())),
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
