"""B1 physical-feature sufficiency screen for MFT and ISRJ presence."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

TARGETS = {"MFT": 3, "ISRJ": 5}
DEFAULT_RUN = Path("outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce/seed_20260724")


def _pool(sequence: np.ndarray, bins: int) -> np.ndarray:
    usable = sequence.shape[1] // bins * bins
    return sequence[:, :usable].reshape(len(sequence), bins, -1).max(axis=2)


def physics_features_from_normalized_iq(iq: np.ndarray) -> dict[str, np.ndarray]:
    """Build the frozen features from a pre-normalized float32 [B,2,3000] batch."""
    raw = np.asarray(iq, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (2, 3000):
        raise ValueError("iq must have shape [B,2,3000]")
    signal = raw[:, 0].astype(np.float64) + 1j * raw[:, 1].astype(np.float64)
    fft = np.fft.fft(signal, n=8192, axis=1)
    complex_acf = np.abs(np.fft.ifft(np.abs(fft) ** 2, axis=1)[:, 1:1201])
    complex_acf /= np.maximum(complex_acf[:, :1], 1e-8)
    envelope = np.abs(signal)
    envelope -= envelope.mean(axis=1, keepdims=True)
    envelope_fft = np.fft.fft(envelope, n=8192, axis=1)
    envelope_acf = np.abs(np.fft.ifft(np.abs(envelope_fft) ** 2, axis=1)[:, 1:1201])
    envelope_acf /= np.maximum(envelope_acf[:, :1], 1e-8)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal, axis=1), axes=1))
    spectrum /= np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-8)
    complex_bins = _pool(complex_acf, 120)
    envelope_bins = _pool(envelope_acf, 120)
    spectrum_bins = spectrum.reshape(len(signal), 60, 50).sum(axis=2)
    delay = complex_acf[:, 249:1000]
    periodic = envelope_acf[:, 39:600]
    delay_peak = delay.max(axis=1, keepdims=True)
    delay_position = delay.argmax(axis=1, keepdims=True) / delay.shape[1]
    periodic_peak = periodic.max(axis=1, keepdims=True)
    periodic_position = periodic.argmax(axis=1, keepdims=True) / periodic.shape[1]
    complex_top = np.sort(delay, axis=1)[:, -8:]
    envelope_top = np.sort(periodic, axis=1)[:, -8:]
    mft = np.concatenate((complex_bins, spectrum_bins, delay_peak, delay_position, complex_top), axis=1)
    isrj = np.concatenate((complex_bins, envelope_bins, spectrum_bins, periodic_peak, periodic_position, envelope_top), axis=1)
    combined = np.concatenate((complex_bins, envelope_bins, spectrum_bins, delay_peak, delay_position,
                               periodic_peak, periodic_position, complex_top, envelope_top), axis=1)
    return {"mft_delay": mft.astype(np.float32), "isrj_periodicity": isrj.astype(np.float32),
            "combined": combined.astype(np.float32)}


def extract_physics_features(mat_path: Path, indices: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    output = {"mft_delay": [], "isrj_periodicity": [], "combined": []}
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
    with h5py.File(mat_path, "r") as handle:
        for start in range(0, len(sorted_indices), batch_size):
            selected = sorted_indices[start:start + batch_size]
            raw = np.asarray(handle["data_legacy"][:, selected], dtype=np.float32).T.reshape(-1, 2, 3000)
            rms = np.sqrt(np.mean(np.square(raw).sum(axis=1), axis=1, keepdims=True))
            normalized = raw / np.maximum(rms[:, None], 1e-8)
            batch = physics_features_from_normalized_iq(normalized)
            for key in output:
                output[key].append(batch[key])
    return {key: np.concatenate(value) for key, value in output.items()}


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 99):
        candidates.append((float(f1_score(labels, probabilities >= threshold, zero_division=0)),
                           -abs(float(threshold) - 0.5), float(threshold)))
    return max(candidates)[2]


def metrics_from_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    weak_mask: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= np.asarray(thresholds)[None, :]
    target = labels.astype(bool)
    per_label_aupr = [
        float(average_precision_score(labels[:, index], probabilities[:, index]))
        for index in range(labels.shape[1])
    ]
    per_label_fpr = []
    per_label_f1 = []
    for index in range(labels.shape[1]):
        tp = int((predicted[:, index] & target[:, index]).sum())
        fp = int((predicted[:, index] & ~target[:, index]).sum())
        fn = int((~predicted[:, index] & target[:, index]).sum())
        tn = int((~predicted[:, index] & ~target[:, index]).sum())
        per_label_fpr.append(float(fp / max(fp + tn, 1)))
        per_label_f1.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    weak_positive = weak_mask.astype(bool) & target
    weak_recall = float(
        predicted[weak_positive].mean()
    ) if np.any(weak_positive) else float("nan")
    return {
        "macro_aupr": float(np.mean(per_label_aupr)),
        "per_label_aupr": per_label_aupr,
        "per_label_fpr": per_label_fpr,
        "macro_f1": float(np.mean(per_label_f1)),
        "hamming_loss": float((predicted != target).mean()),
        "weak_component_recall": weak_recall,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b1_physics_sufficiency_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.baseline_run / "summary.json").read_text(encoding="utf-8"))
    seed = int(summary["seed"])
    audit = json.loads((args.baseline_run / "data_audit.json").read_text(encoding="utf-8"))
    mat_path = Path(audit["levels"]["10"]["path"])
    split = np.load(args.baseline_run / "split_indices.npz")
    manifest = json.loads((args.baseline_run / "longtail_manifest.json").read_text(encoding="utf-8"))
    roles = {
        "train": np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64)),
        "early": np.sort(split["early_stop"].astype(np.int64)),
        "development": np.sort(split["development"].astype(np.int64)),
    }
    with h5py.File(mat_path, "r") as handle:
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
    started = time.perf_counter()
    matrices = {role: extract_physics_features(mat_path, indices, args.batch_size) for role, indices in roles.items()}
    payload = np.load(args.baseline_run / "development_predictions.npz")
    lookup = {int(index): position for position, index in enumerate(payload["sample_indices_10"])}
    order = [lookup[int(index)] for index in roles["development"]]
    baseline_probability = payload["probabilities_10"][order]
    development_labels = payload["labels_10"][order]
    development_weak = payload["weak_mask_10"][order]
    baseline_thresholds = np.asarray(json.loads((args.baseline_run / "thresholds.json").read_text(encoding="utf-8"))["thresholds"])
    baseline_metrics = metrics_from_probabilities(
        development_labels, baseline_probability, baseline_thresholds, development_weak
    )
    rows = []
    candidates = {"MFT": ("mft_delay", "combined"), "ISRJ": ("isrj_periodicity", "combined")}
    for target_name, target_index in TARGETS.items():
        baseline_aupr = float(baseline_metrics["per_label_aupr"][target_index])
        baseline_fpr = float(baseline_metrics["per_label_fpr"][target_index])
        for feature_name in candidates[target_name]:
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                l2_regularization=1.0, random_state=seed,
            )
            model.fit(matrices["train"][feature_name], labels[roles["train"], target_index])
            early_probability = model.predict_proba(matrices["early"][feature_name])[:, 1]
            development_probability = model.predict_proba(matrices["development"][feature_name])[:, 1]
            threshold = best_threshold(labels[roles["early"], target_index], early_probability)
            fused = baseline_probability.copy()
            fused[:, target_index] = development_probability
            thresholds = baseline_thresholds.copy()
            thresholds[target_index] = threshold
            metrics = metrics_from_probabilities(development_labels, fused, thresholds, development_weak)
            target_aupr = float(metrics["per_label_aupr"][target_index])
            target_fpr = float(metrics["per_label_fpr"][target_index])
            row = {
                "target": target_name, "feature": feature_name,
                "feature_dimension": int(matrices["train"][feature_name].shape[1]),
                "early_aupr": float(average_precision_score(labels[roles["early"], target_index], early_probability)),
                "development_aupr": target_aupr,
                "baseline_aupr": baseline_aupr,
                "delta_aupr": target_aupr - baseline_aupr,
                "threshold": threshold,
                "development_fpr": target_fpr,
                "baseline_fpr": baseline_fpr,
                "delta_fpr": target_fpr - baseline_fpr,
                "macro_aupr": float(metrics["macro_aupr"]),
                "delta_macro_aupr": float(metrics["macro_aupr"] - baseline_metrics["macro_aupr"]),
                "hamming": float(metrics["hamming_loss"]),
                "delta_hamming": float(metrics["hamming_loss"] - baseline_metrics["hamming_loss"]),
            }
            row["screen_gate_passed"] = bool(
                (row["delta_aupr"] >= 0.05 or row["delta_fpr"] <= -0.10)
                and row["delta_macro_aupr"] >= -0.01
                and row["delta_hamming"] <= 0.005
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "protocol": {"seed": seed, "mat_path": str(mat_path), "roles": {k: int(len(v)) for k, v in roles.items()},
                     "classifier": "fixed HistGradientBoosting leaf7", "locked_test_used": False},
        "baseline": {"macro_aupr": float(baseline_metrics["macro_aupr"]),
                     "hamming": float(baseline_metrics["hamming_loss"])},
        "runs": rows,
        "promoted_targets": sorted({row["target"] for row in rows if row["screen_gate_passed"]}),
        "seconds": time.perf_counter() - started,
        "locked_test_used": False,
    }
    _write_csv(args.output_root / "screen.csv", rows)
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
