"""Aggregate power and VFT-negative-combination diagnostics for E0-E3."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LABEL_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGET_INDEX = 4


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def iter_runs(baseline_root: Path, candidate_root: Path):
    patterns = (
        (baseline_root, "VFT/D11/b1_bce/seed_*/summary.json"),
        (candidate_root, "VFT/D11/*/seed_*/summary.json"),
    )
    for root, pattern in patterns:
        for summary_path in sorted(root.glob(pattern)):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            yield summary["method"], int(summary["seed"]), summary_path.parent


def combination_name(labels: np.ndarray) -> str:
    active = [name for name, value in zip(LABEL_NAMES, labels) if value]
    return "+".join(active) if active else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/vft_d11_backbone_loss_20260725"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("outputs/vft_d11_hard_negative_20260725"),
    )
    args = parser.parse_args()

    power_values: dict[tuple[str, int], list[dict]] = defaultdict(list)
    combo_values: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    score_values: dict[str, list[dict]] = defaultdict(list)

    for method, seed, run_dir in iter_runs(args.baseline_root, args.candidate_root):
        with (run_dir / "power_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                level = int(row["level"])
                power_values[(method, level)].append(
                    {
                        "target_aupr": float(row["target_aupr"]),
                        "weak_recall": float(row["weak_recall_calibrated"]),
                        "target_fpr": float(row["target_fpr_calibrated"]),
                    }
                )

        thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))[
            "thresholds"
        ]
        threshold = float(thresholds[TARGET_INDEX])
        predictions = np.load(run_dir / "development_predictions.npz")
        labels = predictions["labels_10"].astype(bool)
        probabilities = predictions["probabilities_10"][:, TARGET_INDEX]
        target_labels = labels[:, TARGET_INDEX]

        positives = probabilities[target_labels]
        negatives = probabilities[~target_labels]
        score_values[method].append(
            {
                "seed": seed,
                "threshold": threshold,
                "positive_mean": float(positives.mean()),
                "positive_q10": float(np.quantile(positives, 0.10)),
                "negative_mean": float(negatives.mean()),
                "negative_q90": float(np.quantile(negatives, 0.90)),
                "positive_below_threshold": float(np.mean(positives < threshold)),
                "negative_above_threshold": float(np.mean(negatives >= threshold)),
            }
        )

        negative_rows = labels[~target_labels]
        negative_scores = probabilities[~target_labels]
        for bits in np.unique(negative_rows, axis=0):
            mask = np.all(negative_rows == bits, axis=1)
            support = int(mask.sum())
            fpr = float(np.mean(negative_scores[mask] >= threshold))
            combo_values[(method, combination_name(bits))].append((support, fpr, threshold))

    power_rows = []
    for (method, level), rows in sorted(power_values.items()):
        power_rows.append(
            {
                "method": method,
                "level": level,
                "runs": len(rows),
                **{
                    f"{metric}_mean": float(np.mean([row[metric] for row in rows]))
                    for metric in ("target_aupr", "weak_recall", "target_fpr")
                },
            }
        )

    score_rows = []
    for method, rows in sorted(score_values.items()):
        score_rows.append(
            {
                "method": method,
                "runs": len(rows),
                **{
                    f"{metric}_mean": float(np.mean([row[metric] for row in rows]))
                    for metric in (
                        "threshold",
                        "positive_mean",
                        "positive_q10",
                        "negative_mean",
                        "negative_q90",
                        "positive_below_threshold",
                        "negative_above_threshold",
                    )
                },
            }
        )

    combo_rows = []
    for (method, combination), rows in sorted(combo_values.items()):
        combo_rows.append(
            {
                "method": method,
                "negative_combination": combination,
                "support_sum": sum(row[0] for row in rows),
                "target_fpr_mean": float(np.mean([row[1] for row in rows])),
                "threshold_mean": float(np.mean([row[2] for row in rows])),
            }
        )
    combo_rows.sort(key=lambda row: (row["method"], -row["target_fpr_mean"], -row["support_sum"]))

    output_root = args.candidate_root
    write_csv(output_root / "hard_negative_power_diagnostics.csv", power_rows)
    write_csv(output_root / "hard_negative_score_diagnostics.csv", score_rows)
    write_csv(output_root / "hard_negative_combination_diagnostics.csv", combo_rows)
    report = {
        "locked_test_used": False,
        "power": power_rows,
        "scores_at_primary_level": score_rows,
        "top_negative_combinations": {
            method: [row for row in combo_rows if row["method"] == method][:8]
            for method in sorted(score_values)
        },
    }
    (output_root / "hard_negative_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
