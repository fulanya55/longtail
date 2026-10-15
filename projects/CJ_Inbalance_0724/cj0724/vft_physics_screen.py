"""Screen deployable VFT matched-filter features on the frozen D11 split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FS_HZ = 10.0e6
PULSE_SAMPLES = 1000
CHIRP_RATE_HZ_PER_S = 5.0e10
TARGET_INDEX = 4


def build_vft_filter_bank(
    frequencies_hz: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(PULSE_SAMPLES, device=device, dtype=torch.float32) / FS_HZ
    frequencies = torch.as_tensor(frequencies_hz, device=device, dtype=torch.float32)
    phase = (
        torch.pi * CHIRP_RATE_HZ_PER_S * time[None, :].square()
        + 2.0 * torch.pi * frequencies[:, None] * time[None, :]
    )
    real = torch.cos(phase)
    imag = torch.sin(phase)
    real_weights = torch.stack((real, imag), dim=1)
    imag_weights = torch.stack((-imag, real), dim=1)
    return torch.cat((real_weights, imag_weights), dim=0), frequencies


def matched_filter_peaks(
    iq: torch.Tensor, frequencies_hz: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized correlation peaks and their valid-window positions."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 3000):
        raise ValueError("iq must have shape [B, 2, 3000]")
    weights, _ = build_vft_filter_bank(frequencies_hz, iq.device)
    count = len(frequencies_hz)
    correlation = F.conv1d(iq, weights)
    real, imag = correlation[:, :count], correlation[:, count:]
    local_energy = F.conv1d(
        iq.square().sum(dim=1, keepdim=True),
        torch.ones(1, 1, PULSE_SAMPLES, device=iq.device),
    ).clamp_min(1.0e-8)
    normalized = torch.sqrt(real.square() + imag.square() + 1.0e-12) / torch.sqrt(
        local_energy * float(PULSE_SAMPLES)
    )
    return normalized.max(dim=2)


def feature_matrix(peaks: np.ndarray, positions: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    zero_index = int(np.argmin(np.abs(frequencies)))
    zero = peaks[:, zero_index : zero_index + 1]
    peak_sum = peaks.sum(axis=1, keepdims=True).clip(min=1.0e-8)
    distribution = peaks / peak_sum
    entropy = -(distribution * np.log(distribution.clip(min=1.0e-8))).sum(axis=1, keepdims=True)
    best = peaks.argmax(axis=1)
    summary = np.column_stack(
        (
            peaks.max(axis=1),
            peaks[:, frequencies > 0].max(axis=1),
            peaks[:, frequencies < 0].max(axis=1),
            frequencies[best] / 1000.0,
            positions[np.arange(len(positions)), best] / 2000.0,
            entropy[:, 0],
        )
    )
    return np.concatenate((peaks, peaks - zero, summary), axis=1).astype(np.float32)


def extract_features(
    mat_path: Path,
    indices: np.ndarray,
    frequencies: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
    features = []
    with h5py.File(mat_path, "r") as handle:
        for start in range(0, len(sorted_indices), batch_size):
            selected = sorted_indices[start : start + batch_size]
            raw = np.asarray(handle["data_legacy"][:, selected], dtype=np.float32).T
            iq = raw.reshape(-1, 2, 3000)
            rms = np.sqrt(np.mean(np.square(iq).sum(axis=1), axis=1, keepdims=True))
            iq = iq / np.maximum(rms[:, None], 1.0e-8)
            with torch.no_grad():
                peaks, positions = matched_filter_peaks(
                    torch.from_numpy(iq).to(device), frequencies
                )
            features.append(
                feature_matrix(
                    peaks.cpu().numpy(), positions.cpu().numpy(), frequencies
                )
            )
    return np.concatenate(features)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=Path(
            "outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce/seed_20260724"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vft_physics_screen_20260725"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    baseline_summary = json.loads(
        (args.baseline_run / "summary.json").read_text(encoding="utf-8")
    )
    seed = int(baseline_summary["seed"])
    audit = json.loads((args.baseline_run / "data_audit.json").read_text(encoding="utf-8"))
    mat_path = Path(audit["levels"]["10"]["path"])
    split = np.load(args.baseline_run / "split_indices.npz")
    manifest = json.loads((args.baseline_run / "longtail_manifest.json").read_text(encoding="utf-8"))
    train_indices = np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64))
    early_indices = np.sort(split["early_stop"].astype(np.int64))
    development_indices = np.sort(split["development"].astype(np.int64))
    with h5py.File(mat_path, "r") as handle:
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T

    frequencies = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)
    device = torch.device(args.device)
    matrices = {
        "train": extract_features(mat_path, train_indices, frequencies, device, args.batch_size),
        "early": extract_features(mat_path, early_indices, frequencies, device, args.batch_size),
        "development": extract_features(
            mat_path, development_indices, frequencies, device, args.batch_size
        ),
    }
    role_indices = {
        "train": train_indices,
        "early": early_indices,
        "development": development_indices,
    }
    targets = {
        role: labels[indices, TARGET_INDEX] for role, indices in role_indices.items()
    }

    candidates = {
        "logistic_c0p1": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
        ),
        "logistic_c1": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        ),
        "hist_leaf7": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0,
            random_state=seed,
        ),
        "hist_leaf15": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=seed,
        ),
        "forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
    }
    rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(matrices["train"], targets["train"])
        fitted[name] = model
        row = {"model": name}
        for role in ("early", "development"):
            probability = model.predict_proba(matrices[role])[:, 1]
            row[f"{role}_aupr"] = float(average_precision_score(targets[role], probability))
            row[f"{role}_auroc"] = float(roc_auc_score(targets[role], probability))
            row[f"{role}_positive_mean"] = float(probability[targets[role] == 1].mean())
            row[f"{role}_negative_mean"] = float(probability[targets[role] == 0].mean())
        rows.append(row)
    rows.sort(key=lambda row: row["early_aupr"], reverse=True)
    selected = rows[0]

    baseline_predictions = np.load(args.baseline_run / "development_predictions.npz")
    baseline_probability = baseline_predictions["probabilities_10"][:, TARGET_INDEX]
    baseline_target = baseline_predictions["labels_10"][:, TARGET_INDEX]
    baseline_aupr = float(average_precision_score(baseline_target, baseline_probability))
    selected_probability = fitted[selected["model"]].predict_proba(matrices["development"])[:, 1]
    np.savez_compressed(
        args.output_root / "features_and_predictions.npz",
        frequencies_hz=frequencies,
        train_indices=train_indices,
        early_indices=early_indices,
        development_indices=development_indices,
        development_labels=targets["development"],
        development_probabilities=selected_probability,
        **{f"features_{role}": matrix for role, matrix in matrices.items()},
    )
    write_csv(args.output_root / "candidate_metrics.csv", rows)
    result = {
        "protocol": {
            "seed": seed,
            "level_dB": 10,
            "train_identities": int(len(train_indices)),
            "early_identities": int(len(early_indices)),
            "development_identities": int(len(development_indices)),
            "frequency_grid_hz": frequencies.tolist(),
            "feature_dimension": int(matrices["train"].shape[1]),
            "locked_test_used": False,
            "manifest_sha256": manifest["manifest_sha256"],
            "split_sha256": json.loads(
                (args.baseline_run / "split_receipt.json").read_text(encoding="utf-8")
            )["split_sha256"],
        },
        "selection_role": "early_stop",
        "selected_model": selected,
        "development_baseline_vft_aupr": baseline_aupr,
        "development_delta_vft_aupr": selected["development_aupr"] - baseline_aupr,
        "promote_to_neural_integration": (
            selected["development_aupr"] >= baseline_aupr + 0.02
        ),
        "candidates": rows,
    }
    (args.output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
