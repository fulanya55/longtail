"""Aggregate the five-seed representative-power B3-R engineering confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cj0724.route_b_mechanism_router import SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b3r_gpu_unified_five_seed_20260725"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = {
        20260724: Path("outputs/b3r_gpu_unified_screen_20260725/result.json"),
        **{seed: Path(f"outputs/b3r_gpu_unified_seed{seed}_20260725/result.json") for seed in SEEDS[1:]},
    }
    runs = [(seed, json.loads(path.read_text(encoding="utf-8"))) for seed, path in sources.items()]
    rows = [dict(seed=seed, **row) for seed, run in runs for row in run["rows"]]
    mean_multiplier = np.asarray([row["compute_multiplier_vs_E5_compute"] for row in rows])
    end_to_end_multipliers = np.asarray([
        row["total_seconds"]
        / (row["read_seconds"] + row["normalize_seconds"] + row["E5_compute_only_seconds"])
        for row in rows
    ])
    result = {
        "seeds": list(SEEDS), "powers_dB": [10, 30, 50],
        "runs": [{"seed": seed, **run} for seed, run in runs],
        "summary": {
            "points": len(rows),
            "mean_compute_multiplier_vs_E5": float(mean_multiplier.mean()),
            "std_compute_multiplier_vs_E5": float(mean_multiplier.std(ddof=0)),
            "worst_compute_multiplier_vs_E5": float(mean_multiplier.max()),
            "mean_end_to_end_multiplier_vs_E5": float(end_to_end_multipliers.mean()),
            "worst_end_to_end_multiplier_vs_E5": float(end_to_end_multipliers.max()),
            "mean_total_seconds": float(np.mean([row["total_seconds"] for row in rows])),
            "max_abs_macro_aupr_difference": float(max(abs(row["macro_aupr_difference"]) for row in rows)),
            "max_abs_hamming_difference": float(max(abs(row["hamming_difference"]) for row in rows)),
            "seed_screens_passed": int(sum(run["B3R_screen_passed"] for _, run in runs)),
        },
        "gates": {}, "locked_test_used": False,
    }
    result["gates"] = {
        "five_of_five_seed_screens_passed": result["summary"]["seed_screens_passed"] == 5,
        "mean_compute_at_most_1p5x_E5": result["summary"]["mean_compute_multiplier_vs_E5"] <= 1.5,
        "worst_compute_at_most_1p6x_E5": result["summary"]["worst_compute_multiplier_vs_E5"] <= 1.6,
        "mean_end_to_end_at_most_1p5x_E5": result["summary"]["mean_end_to_end_multiplier_vs_E5"] <= 1.5,
        "worst_end_to_end_at_most_1p6x_E5": result["summary"]["worst_end_to_end_multiplier_vs_E5"] <= 1.6,
        "macro_aupr_equivalent": result["summary"]["max_abs_macro_aupr_difference"] <= 0.003,
        "hamming_equivalent": result["summary"]["max_abs_hamming_difference"] <= 0.003,
    }
    result["B3_confirmed"] = bool(all(result["gates"].values()))
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "gates": result["gates"],
                      "B3_confirmed": result["B3_confirmed"]}, indent=2))


if __name__ == "__main__":
    main()
