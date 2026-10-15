"""Compare E1/E2 hard-negative runs against the frozen E0 BCE baseline."""

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


def load_rows(root: Path, methods: set[str] | None = None) -> list[dict]:
    rows = []
    for path in sorted(root.glob("VFT/D11/*/seed_*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if methods is not None and item["method"] not in methods:
            continue
        rows.append(
            {
                "method": item["method"],
                "seed": int(item["seed"]),
                "best_epoch": int(item["best_epoch"]),
                "parameter_count": int(item["parameter_count"]),
                "manifest_sha256": item["manifest_sha256"],
                "split_sha256": item["split_sha256"],
                **{metric: float(item[metric]) for metric in METRICS},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/vft_d11_hard_negative_20260725"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/vft_d11_backbone_loss_20260725"),
    )
    args = parser.parse_args()

    rows = load_rows(args.baseline_root, {"b1_bce"}) + load_rows(args.run_root)
    if not rows:
        raise FileNotFoundError("No baseline or candidate summaries found")
    write_csv(args.run_root / "hard_negative_runs.csv", rows)

    aggregate_rows = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        aggregate = {
            "method": method,
            "runs": len(selected),
            "parameter_count": selected[0]["parameter_count"],
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate)
    write_csv(args.run_root / "hard_negative_aggregates.csv", aggregate_rows)

    baseline = {row["seed"]: row for row in rows if row["method"] == "b1_bce"}
    delta_rows = []
    decisions = []
    for method in sorted({row["method"] for row in rows if row["method"] != "b1_bce"}):
        candidate = {row["seed"]: row for row in rows if row["method"] == method}
        seeds = sorted(set(baseline) & set(candidate))
        selected = []
        for seed in seeds:
            row = {
                "method": method,
                "seed": seed,
                **{
                    f"delta_{metric}": candidate[seed][metric] - baseline[seed][metric]
                    for metric in METRICS
                },
            }
            selected.append(row)
            delta_rows.append(row)
        means = {
            metric: float(np.mean([row[f"delta_{metric}"] for row in selected]))
            for metric in METRICS
        }
        positive_target_seeds = sum(row["delta_primary_target_aupr"] > 0.0 for row in selected)
        gates = {
            "three_matched_seeds": len(selected) == 3,
            "target_aupr_mean_at_least_plus_0p02": means["primary_target_aupr"] >= 0.02,
            "target_aupr_positive_at_least_2_of_3": positive_target_seeds >= 2,
            "macro_aupr_noninferior": means["primary_macro_aupr"] >= -0.005,
            "hamming_noninferior": means["primary_hamming_calibrated"] <= 0.005,
            "weak_recall_noninferior": means["primary_weak_recall_calibrated"] >= -0.05,
            "target_fpr_noninferior": means["primary_target_fpr_calibrated"] <= 0.0,
        }
        decisions.append(
            {
                "method": method,
                "matched_seeds": seeds,
                "mean_deltas": means,
                "target_aupr_positive_seeds": positive_target_seeds,
                "gates": gates,
                "promote": all(gates.values()),
            }
        )
    write_csv(args.run_root / "hard_negative_deltas.csv", delta_rows)
    (args.run_root / "decision.json").write_text(
        json.dumps(
            {"baseline": "b1_bce", "locked_test_used": False, "decisions": decisions},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
