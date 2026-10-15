"""Recompute B4 aggregate and deployment-cost fields from frozen run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cj0724.dual_axis import POWER_LEVELS
from cj0724.route_b_mechanism_router import SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b4_matched_hard_negatives_20260725"))
    args = parser.parse_args()
    with (args.output_root / "power_runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["power_dB"] = int(row["power_dB"])
        for key in list(row):
            if key not in {"seed", "power_dB", "selected_MFT", "selected_ISRJ"}:
                row[key] = float(row[key])
    target_summary = {}
    target_gates = {}
    for target in ("MFT", "ISRJ"):
        fpr = np.asarray([row[f"{target}_fpr_reduction"] for row in rows])
        aupr = np.asarray([row[f"delta_{target}_aupr"] for row in rows])
        power_recall_drop = [
            np.mean([row[f"{target}_recall_drop"] for row in rows if row["power_dB"] == power])
            for power in POWER_LEVELS
        ]
        seed_rows = []
        for seed in SEEDS:
            selected = [row for row in rows if row["seed"] == seed]
            seed_rows.append({
                "seed": seed,
                "mean_fpr_reduction": float(np.mean([row[f"{target}_fpr_reduction"] for row in selected])),
                "mean_aupr_delta": float(np.mean([row[f"delta_{target}_aupr"] for row in selected])),
            })
        same_direction = sum(row["mean_fpr_reduction"] > 0 and row["mean_aupr_delta"] >= 0 for row in seed_rows)
        target_summary[target] = {
            "mean_fpr_reduction": float(fpr.mean()), "mean_aupr_delta": float(aupr.mean()),
            "max_power_mean_recall_drop": float(max(power_recall_drop)),
            "same_direction_seeds": int(same_direction), "seed_rows": seed_rows,
        }
        target_gates[target] = {
            "mean_fpr_reduction_at_least_0p05": target_summary[target]["mean_fpr_reduction"] >= 0.05,
            "mean_aupr_not_lower": target_summary[target]["mean_aupr_delta"] >= 0.0,
            "max_power_recall_drop_at_most_0p01": target_summary[target]["max_power_mean_recall_drop"] <= 0.01,
            "same_direction_at_least_4_seeds": same_direction >= 4,
        }
    deployment_bytes = []
    for seed in SEEDS:
        seed_root = args.output_root / f"seed_{seed}"
        selected = json.loads((seed_root / "selection.json").read_text(encoding="utf-8"))["selected"]
        deployment_bytes.append(sum(
            (seed_root / f"{target.lower()}_{variant}.joblib").stat().st_size
            for target, variant in selected.items() if variant != "B2-RC"
        ))
    original = json.loads((args.output_root / "result.json").read_text(encoding="utf-8"))
    result = {
        **original,
        "target_summary": target_summary,
        "target_gates": target_gates,
        "cost": {
            **original["cost"],
            "all_screened_model_bytes_per_seed": original["cost"].get(
                "all_screened_model_bytes_per_seed", original["cost"].get("mean_model_bytes_per_seed")
            ),
            "mean_deployment_incremental_bytes_per_seed": float(np.mean(deployment_bytes)),
            "max_deployment_incremental_bytes": int(max(deployment_bytes)),
        },
        "locked_test_used": False,
    }
    result["cost"].pop("mean_model_bytes_per_seed", None)
    result["B4_passed"] = bool(
        all(all(gate.values()) for gate in target_gates.values())
        and all(result["overall_gates"].values())
    )
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"target_summary": target_summary, "target_gates": target_gates,
                      "overall_gates": result["overall_gates"], "cost": result["cost"],
                      "B4_passed": result["B4_passed"]}, indent=2))


if __name__ == "__main__":
    main()
