"""Replay frozen E5 classifiers and benchmark one real development split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

from cj0724.dual_axis import PowerViewDataset
from cj0724.dual_axis_train import make_loader, predict
from cj0724.models import build_benchmark_model
from cj0724.vft_physics_screen import extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("outputs/vft_physics_e5_frozen_candidate_20260725"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vft_physics_e5_frozen_candidate_20260725/audit.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(args.device)
    replay_rows = []
    seeds = (20260724, 20260725, 20260726, 271828, 314159)

    for seed in seeds:
        if seed in (20260724, 20260725, 20260726):
            physics_root = (
                Path("outputs/vft_physics_screen_20260725")
                if seed == 20260724
                else Path(f"outputs/vft_physics_screen_20260725/seed_{seed}")
            )
            fusion_prediction = Path(
                f"outputs/vft_physics_fusion_20260725/seed_{seed}/development_predictions.npz"
            )
        else:
            physics_root = Path(
                f"outputs/vft_physics_confirmation_20260725/seed_{seed}"
            )
            fusion_prediction = Path(
                "outputs/vft_physics_confirmation_fusion_20260725/"
                f"seed_{seed}/development_predictions.npz"
            )
        classifier = joblib.load(
            args.bundle_root / f"seed_{seed}" / "vft_physics_classifier.joblib"
        )
        feature_payload = np.load(physics_root / "features_and_predictions.npz")
        current = classifier.predict_proba(feature_payload["features_development"])[:, 1]
        reported_payload = np.load(fusion_prediction)
        reported = reported_payload["physics_probabilities"]
        reported_indices = reported_payload["sample_indices"]
        source_indices = feature_payload["development_indices"]
        lookup = {int(index): position for position, index in enumerate(source_indices)}
        aligned = current[[lookup[int(index)] for index in reported_indices]]
        replay_rows.append(
            {
                "seed": seed,
                "samples": int(len(aligned)),
                "max_abs_probability_difference": float(np.max(np.abs(aligned - reported))),
                "exact_match": bool(np.array_equal(aligned, reported)),
            }
        )

    seed = 20260724
    baseline = Path(
        "outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce/seed_20260724"
    )
    audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
    mat_path = Path(audit["levels"]["10"]["path"])
    split = np.load(baseline / "split_indices.npz")
    development_indices = np.sort(split["development"].astype(np.int64))
    with h5py.File(mat_path, "r") as handle:
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
    model = build_benchmark_model(
        "efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256
    ).to(device)
    checkpoint = torch.load(baseline / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    dataset = PowerViewDataset({10: mat_path}, labels, development_indices, (10,), seed)
    loader = make_loader(dataset, args.batch_size, False, seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    predict(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    neural_seconds = time.perf_counter() - started
    dataset.close()

    frequencies = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    features = extract_features(
        mat_path, development_indices, frequencies, device, args.batch_size
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    physics_feature_seconds = time.perf_counter() - started
    classifier = joblib.load(
        args.bundle_root / "seed_20260724" / "vft_physics_classifier.joblib"
    )
    started = time.perf_counter()
    classifier.predict_proba(features)
    classifier_seconds = time.perf_counter() - started
    samples = int(len(development_indices))
    benchmark = {
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "samples": samples,
        "batch_size": args.batch_size,
        "neural_end_to_end_seconds": neural_seconds,
        "physics_feature_end_to_end_seconds": physics_feature_seconds,
        "physics_classifier_seconds": classifier_seconds,
        "e5_total_seconds": neural_seconds + physics_feature_seconds + classifier_seconds,
        "neural_samples_per_second": samples / neural_seconds,
        "physics_feature_samples_per_second": samples / physics_feature_seconds,
        "e5_samples_per_second": samples
        / (neural_seconds + physics_feature_seconds + classifier_seconds),
    }
    result = {
        "locked_test_used": False,
        "frozen_replay": replay_rows,
        "all_replays_exact": all(row["exact_match"] for row in replay_rows),
        "benchmark": benchmark,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
