"""Aggregate a complete Route-B R1 development run without test access.

The script refuses to issue a hard-gate decision for an incomplete run.  It
reports per-variant mean/std/95% t-interval over seeds and paired deltas against
the preregistered references B0 (classification) and B1 (parameters).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


VARIANTS = ("B0", "B1", "B2", "B3", "B4")
METRICS = ("macro_aupr", "macro_f1", "parameter_macro_nmae", "parameter_micro_nmae")


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            for key in ("seed", "best_epoch", "parameters"):
                row[key] = int(row[key])
            for key in METRICS + ("total_loss", "classification_loss", "parameter_loss"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "ci_method": "deterministic_percentile_bootstrap"}
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else float("nan")
    rng = np.random.default_rng(20260826 + len(array))
    bootstrap = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {"n": int(len(array)), "mean": mean, "std": std, "ci95_low": float(low), "ci95_high": float(high), "ci_method": "deterministic_percentile_bootstrap"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("docs"))
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--max-aupr-drop", type=float, default=0.005)
    parser.add_argument("--max-f1-drop", type=float, default=0.01)
    parser.add_argument("--min-parameter-relative-improvement", type=float, default=0.05)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    rows_path = run_dir / "per_seed_summary.csv"
    protocol_path = run_dir / "protocol.json"
    if not rows_path.exists() or not protocol_path.exists():
        raise FileNotFoundError(f"R1 run is missing protocol or per_seed_summary: {run_dir}")
    rows = read_rows(rows_path)
    by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant in VARIANTS}
    seed_counts = {variant: len({row["seed"] for row in members}) for variant, members in by_variant.items()}
    complete = all(seed_counts[variant] >= args.min_seeds for variant in VARIANTS)
    locked_test_used = any(str(row.get("locked_test_used", "False")).lower() == "true" for row in rows)

    aggregate = {}
    for variant, members in by_variant.items():
        aggregate[variant] = {metric: summary([row[metric] for row in members if np.isfinite(row[metric])]) for metric in METRICS}

    gates = {"complete_seed_coverage": complete, "locked_test_unused": not locked_test_used}
    pairwise = {}
    if complete:
        b0 = {row["seed"]: row for row in by_variant["B0"]}
        b1 = {row["seed"]: row for row in by_variant["B1"]}
        for variant in ("B2", "B3", "B4"):
            current = {row["seed"]: row for row in by_variant[variant]}
            shared = sorted(set(b0) & set(current) & set(b1))
            aupr_delta = [current[seed]["macro_aupr"] - b0[seed]["macro_aupr"] for seed in shared]
            f1_delta = [current[seed]["macro_f1"] - b0[seed]["macro_f1"] for seed in shared]
            nmae_delta = [b1[seed]["parameter_macro_nmae"] - current[seed]["parameter_macro_nmae"] for seed in shared]
            relative = [delta / b1[seed]["parameter_macro_nmae"] for seed, delta in zip(shared, nmae_delta)]
            pairwise[variant] = {
                "seeds": shared,
                "classification_macro_aupr_delta": summary(aupr_delta),
                "classification_macro_f1_delta": summary(f1_delta),
                "parameter_macro_nmae_relative_improvement": summary(relative),
                "each_seed_classification_aupr_safe": bool(all(delta >= -args.max_aupr_drop for delta in aupr_delta)),
                "each_seed_classification_f1_safe": bool(all(delta >= -args.max_f1_drop for delta in f1_delta)),
                "mean_parameter_relative_improvement_pass": bool(np.mean(relative) >= args.min_parameter_relative_improvement),
            }
        gates["classification_safety"] = all(
            pairwise[v]["each_seed_classification_aupr_safe"] and pairwise[v]["each_seed_classification_f1_safe"]
            for v in pairwise
        )
        gates["parameter_improvement"] = all(pairwise[v]["mean_parameter_relative_improvement_pass"] for v in pairwise)
    else:
        gates.update({"classification_safety": None, "parameter_improvement": None})
    gates["hard_gate_pass"] = bool(complete and not locked_test_used and gates["classification_safety"] and gates["parameter_improvement"])

    output_root = args.output_root.resolve()
    results_dir = output_root / "results"
    receipts_dir = output_root / "receipts"
    reports_dir = output_root / "reports"
    for directory in (results_dir, receipts_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "R1",
        "status": "verified_development" if complete else "running_uncertain",
        "run_dir": str(run_dir),
        "seed_counts": seed_counts,
        "locked_test_used": locked_test_used,
        "thresholds": {"max_aupr_drop": args.max_aupr_drop, "max_f1_drop": args.max_f1_drop, "min_parameter_relative_improvement": args.min_parameter_relative_improvement},
        "aggregate": aggregate,
        "pairwise": pairwise,
        "gates": gates,
        "performance_claim": "development_only; no locked-test claim",
    }
    (results_dir / "route_b_r1_aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (receipts_dir / "route_b_r1_aggregate_receipt.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = ["# Route B R1 development aggregate", "", f"- Status: **{payload['status']}**", f"- Run: `{run_dir}`", f"- Locked test used: `{locked_test_used}`", "", "## Seed coverage", "", "| Variant | Seeds |", "|---|---:|"]
    report.extend(f"| {variant} | {seed_counts[variant]} |" for variant in VARIANTS)
    report += ["", "## Per-variant metrics (mean ± SD; deterministic 95% percentile bootstrap CI)", "", "| Variant | Macro AUPR | Macro F1 | Parameter macro NMAE |", "|---|---:|---:|---:|"]
    for variant in VARIANTS:
        values = aggregate[variant]
        def fmt(metric: str) -> str:
            item = values[metric]
            return "NA" if item["n"] == 0 else f"{item['mean']:.4f} ± {item['std']:.4f} [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]"
        report.append(f"| {variant} | {fmt('macro_aupr')} | {fmt('macro_f1')} | {fmt('parameter_macro_nmae')} |")
    report += ["", "## Gate decision", "", f"`{json.dumps(gates, ensure_ascii=False)}`", "", "Only a complete five-seed development run can receive a gate decision; no test-v2 or locked-test file is opened by this script."]
    (reports_dir / "route_b_r1_aggregate.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
