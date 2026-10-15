"""Aggregate completed dual-axis runs and calculate 2x2 effects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "primary_macro_aupr",
    "primary_target_aupr",
    "primary_macro_f1_calibrated",
    "primary_hamming_calibrated",
    "primary_weak_recall_calibrated",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/dual_axis_b1"))
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.run_root.glob("*/*/*/seed_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["summary_path"] = str(path.resolve())
        summaries.append(payload)
    if not summaries:
        raise FileNotFoundError(f"No completed summaries under {args.run_root}")

    run_rows = []
    for item in summaries:
        row = {
            "target_label": item["target_label"],
            "cell": item["cell"],
            "loss": item["loss"],
            "method": item.get("method", item["loss"]),
            "seed": item["seed"],
            "best_epoch": item["best_epoch"],
            "manifest_sha256": item["manifest_sha256"],
            "split_sha256": item["split_sha256"],
        }
        row.update({metric: item[metric] for metric in METRICS})
        run_rows.append(row)
    write_csv(args.run_root / "runs.csv", run_rows)

    aggregate_rows = []
    keys = sorted({(row["target_label"], row["cell"], row["method"]) for row in run_rows})
    for target, cell, method in keys:
        selected = [
            row for row in run_rows
            if row["target_label"] == target and row["cell"] == cell and row["method"] == method
        ]
        row = {"target_label": target, "cell": cell, "method": method, "runs": len(selected)}
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        aggregate_rows.append(row)
    write_csv(args.run_root / "cell_aggregates.csv", aggregate_rows)

    effect_rows = []
    targets = sorted({row["target_label"] for row in run_rows})
    methods = sorted({row["method"] for row in run_rows})
    for target in targets:
        for method in methods:
            seeds = sorted(
                set.intersection(
                    *[
                        {int(row["seed"]) for row in run_rows if row["target_label"] == target and row["method"] == method and row["cell"] == cell}
                        for cell in ("D00", "D10", "D01", "D11")
                    ]
                )
            )
            for seed in seeds:
                by_cell = {
                    row["cell"]: row
                    for row in run_rows
                    if row["target_label"] == target and row["method"] == method and int(row["seed"]) == seed
                }
                row = {"target_label": target, "method": method, "seed": seed}
                for metric in METRICS:
                    d00 = float(by_cell["D00"][metric])
                    d10 = float(by_cell["D10"][metric])
                    d01 = float(by_cell["D01"][metric])
                    d11 = float(by_cell["D11"][metric])
                    row[f"{metric}_frequency_effect"] = d10 - d00
                    row[f"{metric}_visibility_effect"] = d01 - d00
                    row[f"{metric}_joint_effect"] = d11 - d00
                    row[f"{metric}_interaction"] = d11 - d10 - d01 + d00
                effect_rows.append(row)
    write_csv(args.run_root / "paired_effects.csv", effect_rows)

    power_rows = []
    for item in summaries:
        path = Path(item["summary_path"]).parent / "power_metrics.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                power_rows.append(
                    {
                        "target_label": item["target_label"],
                        "cell": item["cell"],
                        "loss": item["loss"],
                        "method": item.get("method", item["loss"]),
                        "seed": int(item["seed"]),
                        **{key: raw[key] for key in raw},
                    }
                )
    write_csv(args.run_root / "power_runs.csv", power_rows)

    power_metric_names = (
        "macro_aupr",
        "target_aupr",
        "macro_f1_calibrated",
        "hamming_calibrated",
        "weak_recall_calibrated",
    )
    power_aggregate_rows = []
    power_keys = sorted(
        {
            (row["target_label"], row["cell"], row["method"], int(row["level"]))
            for row in power_rows
        }
    )
    for target, cell, method, level in power_keys:
        selected = [
            row
            for row in power_rows
            if row["target_label"] == target
            and row["cell"] == cell
            and row["method"] == method
            and int(row["level"]) == level
        ]
        aggregate = {
            "target_label": target,
            "cell": cell,
            "method": method,
            "level": level,
            "runs": len(selected),
        }
        for metric in power_metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            aggregate[f"{metric}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        power_aggregate_rows.append(aggregate)
    write_csv(args.run_root / "power_curve_aggregates.csv", power_aggregate_rows)

    matched_rows = []
    for target in sorted({row["target_label"] for row in power_rows}):
        weak_level = 50 if target in ("FMB", "SawFM", "SinFM") else 10
        for method in sorted({row["method"] for row in power_rows if row["target_label"] == target}):
            seeds = sorted({int(row["seed"]) for row in power_rows if row["target_label"] == target and row["method"] == method})
            for seed in seeds:
                selected = [
                    row
                    for row in power_rows
                    if row["target_label"] == target
                    and row["method"] == method
                    and int(row["seed"]) == seed
                ]
                lookup = {(row["cell"], int(row["level"])): row for row in selected}
                needed = [
                    (cell, level)
                    for cell in ("D00", "D10", "D01", "D11")
                    for level in (30, weak_level)
                ]
                if any(key not in lookup for key in needed):
                    continue
                result = {
                    "target_label": target,
                    "method": method,
                    "seed": seed,
                    "weak_level": weak_level,
                }
                for metric in ("macro_aupr", "target_aupr", "macro_f1_calibrated", "hamming_calibrated"):
                    value = lambda cell, level: float(lookup[(cell, level)][metric])
                    result[f"{metric}_D00_power_drop"] = value("D00", weak_level) - value("D00", 30)
                    result[f"{metric}_D10_power_drop"] = value("D10", weak_level) - value("D10", 30)
                    result[f"{metric}_balanced_exposure_gain"] = value("D01", weak_level) - value("D00", weak_level)
                    result[f"{metric}_longtail_exposure_gain"] = value("D11", weak_level) - value("D10", weak_level)
                    result[f"{metric}_frequency_effect_unexposed"] = value("D10", weak_level) - value("D00", weak_level)
                    result[f"{metric}_frequency_effect_exposed"] = value("D11", weak_level) - value("D01", weak_level)
                    result[f"{metric}_matched_interaction"] = (
                        value("D11", weak_level)
                        - value("D10", weak_level)
                        - value("D01", weak_level)
                        + value("D00", weak_level)
                    )
                matched_rows.append(result)
    write_csv(args.run_root / "matched_power_effects.csv", matched_rows)
    print(f"Aggregated {len(run_rows)} runs: {args.run_root.resolve()}")


if __name__ == "__main__":
    main()
