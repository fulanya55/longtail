"""Fuse the frozen E0 VFT output with the validated matched-filter branch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

from cj0724.dual_axis import PowerViewDataset
from cj0724.dual_axis_train import make_loader, metrics_from_probabilities, predict
from cj0724.models import build_benchmark_model
from cj0724.vft_physics_screen import extract_features


TARGET_INDEX = 4


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best = (-1.0, -1.0, 0.5)
    for threshold in np.linspace(0.01, 0.99, 99):
        score = float(f1_score(labels, probabilities >= threshold, zero_division=0))
        candidate = (score, -abs(float(threshold) - 0.5), float(threshold))
        if candidate[:2] > best[:2]:
            best = candidate
    return best[2]


def align_rows(
    values: np.ndarray, source_indices: np.ndarray, target_indices: np.ndarray
) -> np.ndarray:
    lookup = {int(index): position for position, index in enumerate(source_indices)}
    return values[[lookup[int(index)] for index in target_indices]]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        "--output-root",
        type=Path,
        default=Path("outputs/vft_physics_fusion_20260725"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260724, 20260725, 20260726])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    frequencies = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)
    rows = []

    for seed in args.seeds:
        baseline_run = args.baseline_root / f"seed_{seed}"
        physics_run = (
            args.physics_root
            if seed == 20260724
            else args.physics_root / f"seed_{seed}"
        )
        audit = json.loads((baseline_run / "data_audit.json").read_text(encoding="utf-8"))
        mat_path = Path(audit["levels"]["10"]["path"])
        split = np.load(baseline_run / "split_indices.npz")
        manifest = json.loads((baseline_run / "longtail_manifest.json").read_text(encoding="utf-8"))
        train_indices = np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64))
        calibration_indices = np.sort(split["calibration"].astype(np.int64))
        development_indices = np.sort(split["development"].astype(np.int64))
        with h5py.File(mat_path, "r") as handle:
            labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T

        saved = np.load(physics_run / "features_and_predictions.npz")
        saved_train_indices = saved["train_indices"].astype(np.int64)
        if not np.array_equal(saved_train_indices, train_indices):
            raise ValueError(f"Physics train indices do not match baseline seed {seed}")
        train_features = saved["features_train"]
        development_features = align_rows(
            saved["features_development"],
            saved["development_indices"].astype(np.int64),
            development_indices,
        )
        calibration_features = extract_features(
            mat_path, calibration_indices, frequencies, device, args.batch_size
        )
        physics = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=seed,
        )
        physics.fit(train_features, labels[train_indices, TARGET_INDEX])
        physics_calibration = physics.predict_proba(calibration_features)[:, 1]
        physics_development = physics.predict_proba(development_features)[:, 1]

        model = build_benchmark_model(
            "efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256
        ).to(device)
        checkpoint = torch.load(baseline_run / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        calibration_dataset = PowerViewDataset(
            {10: mat_path}, labels, calibration_indices, (10,), seed
        )
        calibration_loader = make_loader(
            calibration_dataset, args.batch_size, False, seed
        )
        calibration_true, neural_calibration, _, calibration_sample_indices, _ = predict(
            model, calibration_loader, device
        )
        calibration_dataset.close()
        neural_calibration = align_rows(
            neural_calibration, calibration_sample_indices, calibration_indices
        )
        calibration_true = align_rows(
            calibration_true, calibration_sample_indices, calibration_indices
        )

        development_payload = np.load(baseline_run / "development_predictions.npz")
        neural_development = align_rows(
            development_payload["probabilities_10"],
            development_payload["sample_indices_10"],
            development_indices,
        )
        development_true = align_rows(
            development_payload["labels_10"],
            development_payload["sample_indices_10"],
            development_indices,
        )
        development_weak = align_rows(
            development_payload["weak_mask_10"],
            development_payload["sample_indices_10"],
            development_indices,
        )

        alpha_candidates = []
        for alpha in np.linspace(0.0, 1.0, 21):
            fused = (
                (1.0 - alpha) * neural_calibration[:, TARGET_INDEX]
                + alpha * physics_calibration
            )
            alpha_candidates.append(
                {
                    "alpha": float(alpha),
                    "calibration_aupr": float(
                        average_precision_score(calibration_true[:, TARGET_INDEX], fused)
                    ),
                }
            )
        selected_alpha = max(
            alpha_candidates,
            key=lambda item: (item["calibration_aupr"], -abs(item["alpha"] - 0.5)),
        )["alpha"]
        fused_calibration_target = (
            (1.0 - selected_alpha) * neural_calibration[:, TARGET_INDEX]
            + selected_alpha * physics_calibration
        )
        thresholds = np.asarray(
            json.loads((baseline_run / "thresholds.json").read_text(encoding="utf-8"))[
                "thresholds"
            ],
            dtype=np.float64,
        )
        thresholds[TARGET_INDEX] = best_f1_threshold(
            calibration_true[:, TARGET_INDEX], fused_calibration_target
        )
        fused_development = neural_development.copy()
        fused_development[:, TARGET_INDEX] = (
            (1.0 - selected_alpha) * neural_development[:, TARGET_INDEX]
            + selected_alpha * physics_development
        )
        metrics = metrics_from_probabilities(
            development_true, fused_development, thresholds, development_weak
        )
        summary = json.loads((baseline_run / "summary.json").read_text(encoding="utf-8"))
        target_labels = development_true[:, TARGET_INDEX].astype(bool)
        row = {
            "seed": seed,
            "selected_physics_weight": selected_alpha,
            "calibration_fused_vft_aupr": max(
                item["calibration_aupr"] for item in alpha_candidates
            ),
            "vft_threshold": float(thresholds[TARGET_INDEX]),
            "macro_aupr": float(metrics["macro_aupr"]),
            "vft_aupr": float(metrics["per_label_aupr"][TARGET_INDEX]),
            "macro_f1": float(metrics["macro_f1"]),
            "hamming": float(metrics["hamming_loss"]),
            "weak_recall": float(metrics["weak_component_recall"]),
            "vft_fpr": float(metrics["per_label_fpr"][TARGET_INDEX]),
            "vft_positive_mean": float(
                fused_development[target_labels, TARGET_INDEX].mean()
            ),
            "vft_negative_mean": float(
                fused_development[~target_labels, TARGET_INDEX].mean()
            ),
            "delta_macro_aupr": float(metrics["macro_aupr"] - summary["primary_macro_aupr"]),
            "delta_vft_aupr": float(
                metrics["per_label_aupr"][TARGET_INDEX] - summary["primary_target_aupr"]
            ),
            "delta_macro_f1": float(
                metrics["macro_f1"] - summary["primary_macro_f1_calibrated"]
            ),
            "delta_hamming": float(
                metrics["hamming_loss"] - summary["primary_hamming_calibrated"]
            ),
            "delta_weak_recall": float(
                metrics["weak_component_recall"] - summary["primary_weak_recall_calibrated"]
            ),
            "delta_vft_fpr": float(
                metrics["per_label_fpr"][TARGET_INDEX] - summary["primary_target_fpr_calibrated"]
            ),
            "manifest_sha256": manifest["manifest_sha256"],
            "split_sha256": summary["split_sha256"],
            "locked_test_used": False,
        }
        rows.append(row)
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            seed_dir / "development_predictions.npz",
            labels=development_true,
            probabilities=fused_development,
            weak_mask=development_weak,
            sample_indices=development_indices,
            physics_probabilities=physics_development,
            neural_probabilities=neural_development,
        )
        (seed_dir / "protocol.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "selection_role": "calibration",
                    "alpha_candidates": alpha_candidates,
                    "selected_physics_weight": selected_alpha,
                    "thresholds": thresholds.tolist(),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "split_sha256": summary["split_sha256"],
                    "locked_test_used": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    write_csv(args.output_root / "runs.csv", rows)
    numeric = [key for key in rows[0] if key not in ("seed", "manifest_sha256", "split_sha256", "locked_test_used")]
    means = {key: float(np.mean([row[key] for row in rows])) for key in numeric}
    expected_runs = len(args.seeds)
    gates = {
        "all_requested_seeds_completed": len(rows) == expected_runs,
        "vft_aupr_mean_at_least_plus_0p02": means["delta_vft_aupr"] >= 0.02,
        "vft_aupr_positive_all_requested_seeds": sum(
            row["delta_vft_aupr"] > 0 for row in rows
        )
        == expected_runs,
        "macro_aupr_noninferior": means["delta_macro_aupr"] >= -0.005,
        "hamming_noninferior": means["delta_hamming"] <= 0.005,
        "weak_recall_noninferior": means["delta_weak_recall"] >= -0.05,
        "vft_fpr_noninferior": means["delta_vft_fpr"] <= 0.0,
    }
    result = {
        "method": "e5_e0_plus_vft_matched_filter",
        "means": means,
        "gates": gates,
        "promote": all(gates.values()),
        "locked_test_used": False,
        "runs": rows,
    }
    (args.output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
