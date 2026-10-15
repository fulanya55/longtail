"""Post-screen B2 attribution using saved predictions only; no model fitting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cj0724.b1_physics_sufficiency import metrics_from_probabilities
from cj0724.route_b_mechanism_router import SEEDS


POWER_LEVELS = (10, 15, 20, 25, 30, 35, 40, 45, 50)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2-root", type=Path, default=Path("outputs/b2_power_boundary_20260725"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b2_power_boundary_ablation_20260725"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    pure_isrj_rows = []
    for seed in SEEDS:
        for power in POWER_LEVELS:
            saved = np.load(args.b2_root / f"seed_{seed}/power_{power}dB.npz")
            labels = saved["labels"]
            weak = saved["weak_mask"]
            e5 = saved["probabilities_E5"].astype(np.float64)
            hard = saved["probabilities_B2-Hard"].astype(np.float64)
            e5_thresholds = saved["thresholds_E5"].astype(np.float64)
            hard_thresholds = saved["thresholds_B2-Hard"].astype(np.float64)
            variants = {
                "E5": (e5, e5_thresholds),
                "E5_plus_MFT": (e5.copy(), e5_thresholds.copy()),
                "E5_plus_ISRJ": (e5.copy(), e5_thresholds.copy()),
                "B2-Hard": (hard, hard_thresholds),
            }
            variants["E5_plus_MFT"][0][:, 3] = hard[:, 3]
            variants["E5_plus_MFT"][1][3] = hard_thresholds[3]
            variants["E5_plus_ISRJ"][0][:, 5] = hard[:, 5]
            variants["E5_plus_ISRJ"][1][5] = hard_thresholds[5]
            pure_isrj = np.all(labels == np.asarray([0, 0, 0, 0, 0, 1]), axis=1)
            for name, (probabilities, thresholds) in variants.items():
                metrics = metrics_from_probabilities(labels, probabilities, thresholds, weak)
                predicted = probabilities >= thresholds[None, :]
                rows.append({
                    "seed": seed, "power_dB": power, "variant": name,
                    "macro_aupr": float(metrics["macro_aupr"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "hamming": float(metrics["hamming_loss"]),
                    "subset_accuracy": float(np.all(predicted == labels.astype(bool), axis=1).mean()),
                    "weak_recall": float(metrics["weak_component_recall"]),
                    "MFT_aupr": float(metrics["per_label_aupr"][3]),
                    "MFT_fpr": float(metrics["per_label_fpr"][3]),
                    "ISRJ_aupr": float(metrics["per_label_aupr"][5]),
                    "ISRJ_fpr": float(metrics["per_label_fpr"][5]),
                })
                pure_isrj_rows.append({
                    "seed": seed, "power_dB": power, "variant": name,
                    "support": int(pure_isrj.sum()),
                    "exact_accuracy": float(np.all(predicted[pure_isrj] == labels[pure_isrj].astype(bool), axis=1).mean()),
                    "MFT_false_positive_rate": float(predicted[pure_isrj, 3].mean()),
                    "ISRJ_recall": float(predicted[pure_isrj, 5].mean()),
                })
    aggregates = []
    for name in ("E5", "E5_plus_MFT", "E5_plus_ISRJ", "B2-Hard"):
        selected = [row for row in rows if row["variant"] == name]
        aggregate = {"variant": name}
        for metric in ("macro_aupr", "macro_f1", "hamming", "subset_accuracy", "weak_recall"):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            aggregate[f"{metric}_mean_45_runs"] = float(np.nanmean(values))
        aggregates.append(aggregate)
    e5 = next(row for row in aggregates if row["variant"] == "E5")
    for aggregate in aggregates:
        for metric in ("macro_aupr", "macro_f1", "hamming", "subset_accuracy", "weak_recall"):
            aggregate[f"delta_{metric}_vs_E5"] = float(
                aggregate[f"{metric}_mean_45_runs"] - e5[f"{metric}_mean_45_runs"]
            )
    per_power = []
    for power in POWER_LEVELS:
        for name in ("E5_plus_MFT", "E5_plus_ISRJ", "B2-Hard"):
            candidate = [row for row in rows if row["power_dB"] == power and row["variant"] == name]
            reference = [row for row in rows if row["power_dB"] == power and row["variant"] == "E5"]
            per_power.append({
                "power_dB": power, "variant": name,
                "delta_macro_aupr": float(np.mean([row["macro_aupr"] for row in candidate]) -
                                           np.mean([row["macro_aupr"] for row in reference])),
                "delta_hamming": float(np.mean([row["hamming"] for row in candidate]) -
                                        np.mean([row["hamming"] for row in reference])),
                "delta_subset_accuracy": float(np.mean([row["subset_accuracy"] for row in candidate]) -
                                                np.mean([row["subset_accuracy"] for row in reference])),
            })
    conclusion = {
        "ISRJ_branch": "promotable physical evidence branch across power",
        "MFT_branch": "not power-robust; requires reliability protection before routing",
        "B3_allowed": False,
        "reason": "original preregistered B2 aggregate gate failed; do not optimize a non-promoted full router",
        "next_stage": "preregister B2-R with frozen ISRJ branch and a bounded MFT keep-E5 reliability rule",
    }
    result = {"aggregates": aggregates, "per_power": per_power, "conclusion": conclusion,
              "locked_test_used": False}
    _write(args.output_root / "runs.csv", rows)
    _write(args.output_root / "aggregates.csv", aggregates)
    _write(args.output_root / "per_power_deltas.csv", per_power)
    _write(args.output_root / "pure_isrj_combination.csv", pure_isrj_rows)
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
