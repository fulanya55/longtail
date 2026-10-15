"""Evaluate E5 physics replacement across all nine paired power views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from cj0724.dual_axis_train import metrics_from_probabilities
from cj0724.vft_physics_screen import extract_features


TARGET_INDEX = 4
LABEL_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")


def align_rows(values: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {int(index): position for position, index in enumerate(source)}
    return values[[lookup[int(index)] for index in target]]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def combination_name(label: np.ndarray) -> str:
    return "+".join(name for name, value in zip(LABEL_NAMES, label) if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce"),
    )
    parser.add_argument(
        "--physics-root",
        type=Path,
        default=Path("outputs/vft_physics_screen_20260725"),
    )
    parser.add_argument(
        "--fusion-root",
        type=Path,
        default=Path("outputs/vft_physics_fusion_20260725"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    frequencies = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)
    run_rows = []
    combination_rows = []
    label_rows = []
    exact_combination_rows = []

    for seed in (20260724, 20260725, 20260726):
        baseline_run = args.baseline_root / f"seed_{seed}"
        physics_run = args.physics_root if seed == 20260724 else args.physics_root / f"seed_{seed}"
        fusion_protocol = json.loads(
            (args.fusion_root / f"seed_{seed}" / "protocol.json").read_text(encoding="utf-8")
        )
        threshold = float(fusion_protocol["thresholds"][TARGET_INDEX])
        alpha = float(fusion_protocol["selected_physics_weight"])
        baseline_thresholds = np.asarray(
            json.loads((baseline_run / "thresholds.json").read_text(encoding="utf-8"))[
                "thresholds"
            ],
            dtype=np.float64,
        )
        audit = json.loads((baseline_run / "data_audit.json").read_text(encoding="utf-8"))
        manifest = json.loads((baseline_run / "longtail_manifest.json").read_text(encoding="utf-8"))
        train_indices = np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64))
        split = np.load(baseline_run / "split_indices.npz")
        development_indices = np.sort(split["development"].astype(np.int64))
        saved = np.load(physics_run / "features_and_predictions.npz")
        with h5py.File(Path(audit["levels"]["10"]["path"]), "r") as handle:
            labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        physics = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=seed,
        )
        physics.fit(saved["features_train"], labels[train_indices, TARGET_INDEX])
        baseline_predictions = np.load(baseline_run / "development_predictions.npz")

        for level in (10, 15, 20, 25, 30, 35, 40, 45, 50):
            mat_path = Path(audit["levels"][str(level)]["path"])
            if level == 10:
                features = align_rows(
                    saved["features_development"],
                    saved["development_indices"].astype(np.int64),
                    development_indices,
                )
            else:
                features = extract_features(
                    mat_path, development_indices, frequencies, device, args.batch_size
                )
            physics_probability = physics.predict_proba(features)[:, 1]
            source_indices = baseline_predictions[f"sample_indices_{level}"]
            neural = align_rows(
                baseline_predictions[f"probabilities_{level}"], source_indices, development_indices
            )
            true = align_rows(
                baseline_predictions[f"labels_{level}"], source_indices, development_indices
            )
            weak = align_rows(
                baseline_predictions[f"weak_mask_{level}"], source_indices, development_indices
            )
            fused = neural.copy()
            fused[:, TARGET_INDEX] = (
                (1.0 - alpha) * neural[:, TARGET_INDEX] + alpha * physics_probability
            )
            thresholds = np.asarray(fusion_protocol["thresholds"], dtype=np.float64)
            metrics = metrics_from_probabilities(true, fused, thresholds, weak)
            run_rows.append(
                {
                    "seed": seed,
                    "level": level,
                    "macro_aupr": float(metrics["macro_aupr"]),
                    "vft_aupr": float(metrics["per_label_aupr"][TARGET_INDEX]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "hamming": float(metrics["hamming_loss"]),
                    "weak_recall": float(metrics["weak_component_recall"]),
                    "vft_fpr": float(metrics["per_label_fpr"][TARGET_INDEX]),
                }
            )
            if level == 10:
                predicted_all = fused >= thresholds[None, :]
                baseline_predicted_all = neural >= baseline_thresholds[None, :]
                predicted = predicted_all[:, TARGET_INDEX]
                negative = true[:, TARGET_INDEX] == 0
                for label_index, label_name in enumerate(LABEL_NAMES):
                    label_true = true[:, label_index]
                    label_predicted = predicted_all[:, label_index]
                    baseline_label_predicted = baseline_predicted_all[:, label_index]
                    negatives = label_true == 0
                    fused_aupr = float(
                        average_precision_score(label_true, fused[:, label_index])
                    )
                    baseline_aupr = float(
                        average_precision_score(label_true, neural[:, label_index])
                    )
                    fused_f1 = float(
                        f1_score(label_true, label_predicted, zero_division=0)
                    )
                    baseline_f1 = float(
                        f1_score(label_true, baseline_label_predicted, zero_division=0)
                    )
                    label_rows.append(
                        {
                            "seed": seed,
                            "label": label_name,
                            "baseline_aupr": baseline_aupr,
                            "fused_aupr": fused_aupr,
                            "delta_aupr": fused_aupr - baseline_aupr,
                            "precision": float(
                                precision_score(label_true, label_predicted, zero_division=0)
                            ),
                            "recall": float(
                                recall_score(label_true, label_predicted, zero_division=0)
                            ),
                            "baseline_f1": baseline_f1,
                            "fused_f1": fused_f1,
                            "delta_f1": fused_f1 - baseline_f1,
                            "fpr": float(label_predicted[negatives].mean()),
                        }
                    )
                for bits in np.unique(true[negative], axis=0):
                    mask = negative & np.all(true == bits, axis=1)
                    combination_rows.append(
                        {
                            "seed": seed,
                            "combination": combination_name(bits),
                            "support": int(mask.sum()),
                            "vft_fpr": float(predicted[mask].mean()),
                        }
                    )
                for bits in np.unique(true, axis=0):
                    mask = np.all(true == bits, axis=1)
                    baseline_exact = float(
                        np.all(baseline_predicted_all[mask] == true[mask], axis=1).mean()
                    )
                    fused_exact = float(
                        np.all(predicted_all[mask] == true[mask], axis=1).mean()
                    )
                    exact_combination_rows.append(
                        {
                            "seed": seed,
                            "combination": combination_name(bits),
                            "support": int(mask.sum()),
                            "baseline_exact_accuracy": baseline_exact,
                            "fused_exact_accuracy": fused_exact,
                            "delta_exact_accuracy": fused_exact - baseline_exact,
                            "mean_label_accuracy": float(
                                (predicted_all[mask] == true[mask]).mean()
                            ),
                        }
                    )

    write_csv(args.fusion_root / "power_runs.csv", run_rows)
    aggregate_rows = []
    for level in (10, 15, 20, 25, 30, 35, 40, 45, 50):
        selected = [row for row in run_rows if row["level"] == level]
        def safe_mean(metric: str) -> float:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

        aggregate_rows.append(
            {
                "level": level,
                **{
                    f"{metric}_mean": safe_mean(metric)
                    for metric in ("macro_aupr", "vft_aupr", "macro_f1", "hamming", "weak_recall", "vft_fpr")
                },
            }
        )
    write_csv(args.fusion_root / "power_aggregates.csv", aggregate_rows)
    combo_aggregate = []
    for combination in sorted({row["combination"] for row in combination_rows}):
        selected = [row for row in combination_rows if row["combination"] == combination]
        combo_aggregate.append(
            {
                "combination": combination,
                "support_sum": sum(row["support"] for row in selected),
                "vft_fpr_mean": float(np.mean([row["vft_fpr"] for row in selected])),
            }
        )
    combo_aggregate.sort(key=lambda row: (-row["vft_fpr_mean"], -row["support_sum"]))
    write_csv(args.fusion_root / "negative_combination_fpr.csv", combo_aggregate)
    label_aggregate = []
    for label in LABEL_NAMES:
        selected = [row for row in label_rows if row["label"] == label]
        label_aggregate.append(
            {
                "label": label,
                **{
                    f"{metric}_mean": float(np.mean([row[metric] for row in selected]))
                    for metric in (
                        "baseline_aupr",
                        "fused_aupr",
                        "delta_aupr",
                        "precision",
                        "recall",
                        "baseline_f1",
                        "fused_f1",
                        "delta_f1",
                        "fpr",
                    )
                },
            }
        )
    write_csv(args.fusion_root / "per_label_10db.csv", label_aggregate)
    exact_aggregate = []
    for combination in sorted({row["combination"] for row in exact_combination_rows}):
        selected = [row for row in exact_combination_rows if row["combination"] == combination]
        exact_aggregate.append(
            {
                "combination": combination,
                "support_sum": sum(row["support"] for row in selected),
                "baseline_exact_accuracy_mean": float(
                    np.mean([row["baseline_exact_accuracy"] for row in selected])
                ),
                "fused_exact_accuracy_mean": float(
                    np.mean([row["fused_exact_accuracy"] for row in selected])
                ),
                "delta_exact_accuracy_mean": float(
                    np.mean([row["delta_exact_accuracy"] for row in selected])
                ),
                "mean_label_accuracy_mean": float(
                    np.mean([row["mean_label_accuracy"] for row in selected])
                ),
            }
        )
    exact_aggregate.sort(
        key=lambda row: (row["fused_exact_accuracy_mean"], row["combination"])
    )
    write_csv(args.fusion_root / "per_combination_10db.csv", exact_aggregate)
    result = {
        "locked_test_used": False,
        "power_aggregates": aggregate_rows,
        "per_label_10db": label_aggregate,
        "top_negative_combinations": combo_aggregate[:10],
        "lowest_exact_accuracy_combinations": exact_aggregate[:10],
    }
    (args.fusion_root / "power_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
