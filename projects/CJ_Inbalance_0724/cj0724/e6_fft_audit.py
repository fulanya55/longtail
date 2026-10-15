"""E6 audit: FFT equivalence, speed, and frozen-E5 metric preservation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from cj0724.b1_physics_sufficiency import metrics_from_probabilities
from cj0724.vft_fft import extract_features_fft
from cj0724.vft_physics_screen import extract_features


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/e6_fft_audit_20260725"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    physics_root = Path("outputs/vft_physics_screen_20260725")
    frozen_root = Path("outputs/vft_physics_e5_frozen_candidate_20260725")
    fusion_root = Path("outputs/vft_physics_fusion_20260725/seed_20260724")
    baseline_root = Path("outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce/seed_20260724")
    saved = np.load(physics_root / "features_and_predictions.npz")
    frequencies = saved["frequencies_hz"].astype(np.float32)
    indices = saved["development_indices"].astype(np.int64)
    audit = json.loads((baseline_root / "data_audit.json").read_text(encoding="utf-8"))
    mat_path = Path(audit["levels"]["10"]["path"])

    _sync(device)
    started = time.perf_counter()
    direct = extract_features(mat_path, indices, frequencies, device, args.batch_size)
    _sync(device)
    direct_seconds = time.perf_counter() - started
    _sync(device)
    started = time.perf_counter()
    fft = extract_features_fft(mat_path, indices, frequencies, device, args.batch_size)
    _sync(device)
    fft_seconds = time.perf_counter() - started

    classifier = joblib.load(frozen_root / "seed_20260724" / "vft_physics_classifier.joblib")
    direct_probability = classifier.predict_proba(direct)[:, 1]
    fft_probability = classifier.predict_proba(fft)[:, 1]
    fusion = np.load(fusion_root / "development_predictions.npz")
    protocol = json.loads((fusion_root / "protocol.json").read_text(encoding="utf-8"))
    probabilities = fusion["probabilities"].copy()
    lookup = {int(index): position for position, index in enumerate(indices)}
    order = np.asarray([lookup[int(index)] for index in fusion["sample_indices"]], dtype=np.int64)
    probabilities[:, 4] = fft_probability[order]
    thresholds = np.asarray(protocol["thresholds"], dtype=np.float64)
    metrics = metrics_from_probabilities(fusion["labels"], probabilities, thresholds, fusion["weak_mask"])
    reference_metrics = metrics_from_probabilities(
        fusion["labels"], fusion["probabilities"], thresholds, fusion["weak_mask"]
    )
    max_feature_difference = float(np.max(np.abs(direct - fft)))
    max_saved_difference = float(np.max(np.abs(saved["features_development"] - fft)))
    max_probability_difference = float(np.max(np.abs(direct_probability - fft_probability)))
    result = {
        "protocol": {
            "seed": 20260724, "samples": int(len(indices)), "batch_size": args.batch_size,
            "device": str(device), "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "mat_path": str(mat_path), "locked_test_used": False,
        },
        "equivalence": {
            "max_abs_feature_difference_direct_vs_fft": max_feature_difference,
            "max_abs_feature_difference_saved_e5_vs_fft": max_saved_difference,
            "max_abs_probability_difference": max_probability_difference,
            "positions_and_peaks_unit_test": "tests/test_vft_fft.py",
        },
        "benchmark": {
            "direct_seconds": direct_seconds, "fft_seconds": fft_seconds,
            "direct_samples_per_second": len(indices) / direct_seconds,
            "fft_samples_per_second": len(indices) / fft_seconds,
            "speedup": direct_seconds / fft_seconds,
        },
        "reference_metrics": reference_metrics,
        "e6_metrics": metrics,
        "deltas": {
            "macro_aupr": float(metrics["macro_aupr"] - reference_metrics["macro_aupr"]),
            "vft_aupr": float(metrics["per_label_aupr"][4] - reference_metrics["per_label_aupr"][4]),
            "hamming": float(metrics["hamming_loss"] - reference_metrics["hamming_loss"]),
            "weak_recall": float(metrics["weak_component_recall"] - reference_metrics["weak_component_recall"]),
            "vft_fpr": float(metrics["per_label_fpr"][4] - reference_metrics["per_label_fpr"][4]),
        },
        "gates": {},
        "locked_test_used": False,
    }
    result["gates"] = {
        "matched_filter_numerically_equivalent": max_feature_difference <= 2e-4,
        "throughput_at_least_2x": result["benchmark"]["speedup"] >= 2.0,
        "vft_aupr_drop_within_0p005": result["deltas"]["vft_aupr"] >= -0.005,
        "weak_recall_drop_within_0p02": result["deltas"]["weak_recall"] >= -0.02,
        "vft_fpr_increase_within_0p02": result["deltas"]["vft_fpr"] <= 0.02,
    }
    result["e6_promoted"] = bool(all(result["gates"].values()))
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
