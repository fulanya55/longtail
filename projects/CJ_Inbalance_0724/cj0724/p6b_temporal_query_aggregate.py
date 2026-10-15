"""Aggregate preregistered P6b C0/C1/C2 development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path):
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
    protocol, audit, latency = (_read(resolve(value)) for value in (args.preregister, args.audit, args.latency))
    if not audit["all_excluded_overlap_zero"] or not audit["all_targets_finite_and_variable"]:
        raise RuntimeError("P6b fresh identity/target audit did not pass")
    if not audit["alignment"]["all_valid_positive_finite"] or not audit["alignment"]["all_masks_label_aligned"]:
        raise RuntimeError("P6b alignment audit did not pass")
    names = {key: str(value) for key, value in protocol["models"].items()}
    seeds = tuple(int(value) for value in protocol["training"]["model_seeds"])
    run_root = resolve(args.run_root)
    summary = {(str(row["model"]), int(row["seed"])): row for row in _read(run_root / "summary.json")}
    expected = {(model, seed) for model in names.values() for seed in seeds}
    if set(summary) != expected:
        raise RuntimeError(f"P6b model/seed cells are incomplete: {set(summary)}")
    metrics = {}
    for model_name, seed in expected:
        receipt = _read(run_root / f"{model_name}__cos_lr1e4_wd1e4__seed{seed}" / "receipt_parameter.json")
        metrics[(model_name, seed)] = receipt["confirm_adaptive"]

    c0, c1, c2 = (names[key] for key in ("C0_global", "C1_temporal_query", "C2_equal_compute_global_control"))
    seed_rows, target_rows = [], []
    for seed in seeds:
        m0, m1, m2 = (metrics[(model, seed)] for model in (c0, c1, c2))
        p0, p1, p2 = (value["parameter"] for value in (m0, m1, m2))
        seed_rows.append({
            "seed": seed,
            "C1_vs_C0_full12_reduction": float(1.0 - p1["oracle_macro_target_nmae"] / p0["oracle_macro_target_nmae"]),
            "C1_vs_C2_full12_reduction": float(1.0 - p1["oracle_macro_target_nmae"] / p2["oracle_macro_target_nmae"]),
            "C1_vs_C0_pearson_delta": float(p1["oracle_macro_pearson_r"] - p0["oracle_macro_pearson_r"]),
            "C1_vs_C2_pearson_delta": float(p1["oracle_macro_pearson_r"] - p2["oracle_macro_pearson_r"]),
            "C1_vs_C0_macro_aupr_delta": float(m1["macro_aupr"] - m0["macro_aupr"]),
            "C1_vs_C0_subset_delta": float(m1["subset_acc"] - m0["subset_acc"]),
        })
    for index, target in enumerate(metrics[(c0, seeds[0])]["parameter"]["per_target"]):
        deltas = [
            float(metrics[(c1, seed)]["parameter"]["per_target"][index]["normalized_mae"])
            - float(metrics[(c0, seed)]["parameter"]["per_target"][index]["normalized_mae"])
            for seed in seeds
        ]
        target_rows.append({"target": f"{target['class_name']}.{target['name']}", "C1_minus_C0_by_seed": deltas,
                            "mean_C1_minus_C0": float(np.mean(deltas)),
                            "systemic_harm": bool(all(value > 0.03 for value in deltas))})
    gates = protocol["promotion_gates"]
    c1_c2_parameters_equal = all(summary[(c1, seed)]["parameters"] == summary[(c2, seed)]["parameters"] for seed in seeds)
    latency_gates = latency["gates"]
    gate_result = {
        "each_seed_C1_vs_C0_full12": bool(all(row["C1_vs_C0_full12_reduction"] >= gates["each_seed_C1_full12_oracle_macro_nmae_reduction_vs_C0_at_least"] for row in seed_rows)),
        "mean_C1_vs_C0_full12": bool(np.mean([row["C1_vs_C0_full12_reduction"] for row in seed_rows]) >= gates["mean_C1_full12_oracle_macro_nmae_reduction_vs_C0_at_least"]),
        "each_seed_C1_vs_C2_full12": bool(all(row["C1_vs_C2_full12_reduction"] >= gates["each_seed_C1_full12_oracle_macro_nmae_reduction_vs_C2_at_least"] for row in seed_rows)),
        "mean_C1_vs_C2_full12": bool(np.mean([row["C1_vs_C2_full12_reduction"] for row in seed_rows]) >= gates["mean_C1_full12_oracle_macro_nmae_reduction_vs_C2_at_least"]),
        "each_seed_pearson": bool(all(row["C1_vs_C0_pearson_delta"] >= 0.0 and row["C1_vs_C2_pearson_delta"] >= 0.0 for row in seed_rows)),
        "no_systemic_target_harm": not any(row["systemic_harm"] for row in target_rows),
        "each_seed_macro_aupr": bool(all(row["C1_vs_C0_macro_aupr_delta"] >= -gates["each_seed_C1_macro_aupr_drop_vs_C0_at_most"] for row in seed_rows)),
        "each_seed_subset": bool(all(row["C1_vs_C0_subset_delta"] >= -gates["each_seed_C1_subset_accuracy_drop_vs_C0_at_most"] for row in seed_rows)),
        "C1_C2_parameter_equal": bool(c1_c2_parameters_equal),
        "C1_C2_batch64_latency": bool(latency_gates["C1_C2_batch64_median_passed"]),
        "C1_C0_batch64_latency": bool(latency_gates["C1_C0_batch64_median_passed"]),
        "C1_C0_batch1_latency": bool(latency_gates["C1_C0_batch1_p95_passed"]),
    }
    result = {"experiment": protocol["experiment"], "run_root": str(run_root), "seeds": list(seeds), "seed_results": seed_rows,
              "target_results": target_rows, "parameters": {name: int(summary[(name, seeds[0])]["parameters"]) for name in names.values()},
              "gates": gate_result, "development_passed": bool(all(gate_result.values())),
              "verdict": "promote_to_frozen_confirmation" if all(gate_result.values()) else "stop_no_confirmation_data",
              "test_generated": False, "test_opened": False, "locked_test_used": False}
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
