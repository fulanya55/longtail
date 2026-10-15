"""Aggregate the preregistered P6a-r1 C0/C1 development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, object] | list[object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    resolve = lambda value: Path(value) if Path(value).is_absolute() else project / value
    protocol = _load(resolve(args.preregister))
    audit = _load(resolve(args.audit))
    latency = _load(resolve(args.latency))
    run_root = resolve(args.run_root)
    rows = _load(run_root / "summary.json")
    if not isinstance(protocol, dict) or not isinstance(audit, dict) or not isinstance(latency, dict):
        raise TypeError("P6a aggregate inputs must be JSON objects")
    if not isinstance(rows, list):
        raise TypeError("summary.json must be a list")
    if not bool(audit["all_excluded_overlap_zero"]):
        raise RuntimeError("identity audit does not authorize P6a aggregate")
    if not bool(audit["all_targets_finite_and_variable"]):
        raise RuntimeError("target variability audit does not authorize P6a aggregate")
    if not bool(audit["alignment"]["all_valid_positive_finite"]) or not bool(audit["alignment"]["all_masks_label_aligned"]):
        raise RuntimeError("alignment audit does not authorize P6a aggregate")

    baseline_name = str(protocol["comparators"]["C0"])
    candidate_name = str(protocol["comparators"]["C1"])
    seeds = tuple(int(value) for value in protocol["training"]["model_seeds"])
    summary = {(str(row["model"]), int(row["seed"])): row for row in rows}
    expected = {(name, seed) for name in (baseline_name, candidate_name) for seed in seeds}
    if set(summary) != expected:
        raise RuntimeError(f"unexpected model/seed cells: expected={expected}, actual={set(summary)}")

    confirm: dict[tuple[str, int], dict[str, object]] = {}
    for model_name, seed in expected:
        receipt_path = run_root / f"{model_name}__cos_lr1e4_wd1e4__seed{seed}" / "receipt_parameter.json"
        receipt = _load(receipt_path)
        if not isinstance(receipt, dict):
            raise TypeError(f"invalid receipt: {receipt_path}")
        confirm[(model_name, seed)] = receipt["confirm_adaptive"]

    seed_rows, target_rows = [], []
    for seed in seeds:
        baseline, candidate = confirm[(baseline_name, seed)], confirm[(candidate_name, seed)]
        base_parameter = baseline["parameter"]
        candidate_parameter = candidate["parameter"]
        seed_rows.append({
            "seed": seed,
            "full12_oracle_macro_nmae_reduction": float(
                1.0 - float(candidate_parameter["oracle_macro_target_nmae"])
                / float(base_parameter["oracle_macro_target_nmae"])
            ),
            "full12_macro_pearson_delta": float(
                float(candidate_parameter["oracle_macro_pearson_r"])
                - float(base_parameter["oracle_macro_pearson_r"])
            ),
            "macro_aupr_delta": float(float(candidate["macro_aupr"]) - float(baseline["macro_aupr"])),
            "subset_accuracy_delta": float(float(candidate["subset_acc"]) - float(baseline["subset_acc"])),
        })
    for index, target in enumerate(confirm[(baseline_name, seeds[0])]["parameter"]["per_target"]):
        deltas = [
            float(confirm[(candidate_name, seed)]["parameter"]["per_target"][index]["normalized_mae"])
            - float(confirm[(baseline_name, seed)]["parameter"]["per_target"][index]["normalized_mae"])
            for seed in seeds
        ]
        target_rows.append({
            "target": f"{target['class_name']}.{target['name']}",
            "candidate_minus_baseline_nmae_by_seed": deltas,
            "mean_candidate_minus_baseline_nmae": float(np.mean(deltas)),
            "worse_in_all_three_by_more_than_0p03": bool(all(value > 0.03 for value in deltas)),
        })

    gates = protocol["development_promotion_gates"]
    parameter_ratio = float(summary[(candidate_name, seeds[0])]["parameters"]) / float(summary[(baseline_name, seeds[0])]["parameters"])
    latency_gate = latency["gates"]
    gate_result = {
        "each_seed_full12_reduction": bool(all(
            row["full12_oracle_macro_nmae_reduction"] >= float(gates["each_seed_full12_oracle_macro_nmae_reduction_vs_C0_at_least"])
            for row in seed_rows
        )),
        "mean_full12_reduction": bool(np.mean([row["full12_oracle_macro_nmae_reduction"] for row in seed_rows]) >= float(gates["mean_full12_oracle_macro_nmae_reduction_vs_C0_at_least"])),
        "each_seed_pearson": bool(all(row["full12_macro_pearson_delta"] >= 0.0 for row in seed_rows)),
        "no_systemic_target_harm": not any(row["worse_in_all_three_by_more_than_0p03"] for row in target_rows),
        "each_seed_macro_aupr": bool(all(row["macro_aupr_delta"] >= -float(gates["each_seed_macro_aupr_drop_vs_C0_at_most"]) for row in seed_rows)),
        "each_seed_subset_accuracy": bool(all(row["subset_accuracy_delta"] >= -float(gates["each_seed_subset_accuracy_drop_vs_C0_at_most"]) for row in seed_rows)),
        "parameter_count": parameter_ratio <= 1.0 + float(gates["candidate_parameter_count_relative_increase_vs_C0_at_most"]),
        "batch64_latency": bool(latency_gate["batch64_median_ratio_passed"]),
        "batch1_latency": bool(latency_gate["batch1_p95_ratio_passed"]),
    }
    result = {
        "experiment": protocol["experiment"], "preregister": str(resolve(args.preregister)),
        "audit": str(resolve(args.audit)), "latency": str(resolve(args.latency)),
        "run_root": str(run_root), "seeds": list(seeds), "seed_results": seed_rows,
        "target_results": target_rows, "parameter_count_ratio": parameter_ratio,
        "gates": gate_result, "development_passed": bool(all(gate_result.values())),
        "verdict": "promote_to_frozen_confirmation" if all(gate_result.values()) else "stop_no_confirmation_data",
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
