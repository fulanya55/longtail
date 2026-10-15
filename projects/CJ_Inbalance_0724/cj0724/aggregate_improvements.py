"""Aggregate the controlled VFT-D11 backbone/loss screen and apply promotion gates."""

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
    "primary_target_fpr_calibrated",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/vft_d11_backbone_loss_20260725"),
    )
    parser.add_argument("--baseline", default="b1_bce")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_root.glob("VFT/D11/*/seed_*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": item["method"],
                "model": item["model"],
                "loss": item["loss"],
                "seed": int(item["seed"]),
                "best_epoch": int(item["best_epoch"]),
                "parameter_count": int(item["parameter_count"]),
                "manifest_sha256": item["manifest_sha256"],
                "split_sha256": item["split_sha256"],
                **{metric: float(item[metric]) for metric in METRICS},
            }
        )
    if not rows:
        raise FileNotFoundError(f"No summaries found under {args.run_root}")
    write_csv(args.run_root / "improvement_runs.csv", rows)

    aggregate_rows = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        aggregate = {
            "method": method,
            "model": selected[0]["model"],
            "loss": selected[0]["loss"],
            "runs": len(selected),
            "parameter_count": selected[0]["parameter_count"],
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(np.nanmean(values))
            aggregate[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate)
    write_csv(args.run_root / "improvement_aggregates.csv", aggregate_rows)

    baseline_by_seed = {
        row["seed"]: row for row in rows if row["method"] == args.baseline
    }
    delta_rows = []
    decisions = []
    for method in sorted({row["method"] for row in rows if row["method"] != args.baseline}):
        candidate_by_seed = {row["seed"]: row for row in rows if row["method"] == method}
        seeds = sorted(set(baseline_by_seed) & set(candidate_by_seed))
        for seed in seeds:
            baseline = baseline_by_seed[seed]
            candidate = candidate_by_seed[seed]
            delta_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **{
                        f"delta_{metric}": candidate[metric] - baseline[metric]
                        for metric in METRICS
                    },
                }
            )
        selected = [row for row in delta_rows if row["method"] == method]
        means = {
            metric: float(np.mean([row[f"delta_{metric}"] for row in selected]))
            for metric in METRICS
        }
        target_positive_seeds = sum(
            row["delta_primary_target_aupr"] > 0.0 for row in selected
        )
        gates = {
            "three_matched_seeds": len(selected) == 3,
            "target_aupr_mean_at_least_plus_0p02": means["primary_target_aupr"] >= 0.02,
            "target_aupr_positive_at_least_2_of_3": target_positive_seeds >= 2,
            "macro_aupr_noninferior": means["primary_macro_aupr"] >= -0.005,
            "hamming_noninferior": means["primary_hamming_calibrated"] <= 0.005,
            "weak_recall_noninferior": means["primary_weak_recall_calibrated"] >= -0.05,
            "target_fpr_controlled": means["primary_target_fpr_calibrated"] <= 0.02,
        }
        decisions.append(
            {
                "method": method,
                "matched_seeds": seeds,
                "mean_deltas": means,
                "target_aupr_positive_seeds": target_positive_seeds,
                "gates": gates,
                "promote": all(gates.values()),
            }
        )
    write_csv(args.run_root / "improvement_deltas.csv", delta_rows)
    (args.run_root / "decision.json").write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "locked_test_used": False,
                "decisions": decisions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
