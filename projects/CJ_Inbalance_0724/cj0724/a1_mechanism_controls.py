"""A1 mechanism controls for the frozen E5 VFT matched-filter branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from cj0724.vft_physics_screen import FS_HZ, feature_matrix


TARGET_INDEX = 4
DEFAULT_BASELINE = Path("outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce/seed_20260724")


def _filter_bank(
    frequencies_hz: np.ndarray,
    pulse_samples: int,
    chirp_rate_hz_per_s: float,
    device: torch.device,
    random_phase_seed: int | None = None,
) -> torch.Tensor:
    time_axis = torch.arange(pulse_samples, device=device, dtype=torch.float32) / FS_HZ
    frequencies = torch.as_tensor(frequencies_hz, device=device, dtype=torch.float32)
    if random_phase_seed is None:
        base_phase = torch.pi * float(chirp_rate_hz_per_s) * time_axis.square()
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(random_phase_seed))
        base_phase = torch.rand(pulse_samples, generator=generator, dtype=torch.float32).to(device) * (2.0 * torch.pi)
    phase = base_phase[None, :] + 2.0 * torch.pi * frequencies[:, None] * time_axis[None, :]
    real, imag = torch.cos(phase), torch.sin(phase)
    real_weights = torch.stack((real, imag), dim=1)
    imag_weights = torch.stack((-imag, real), dim=1)
    return torch.cat((real_weights, imag_weights), dim=0)


def _template_features(
    mat_path: Path,
    indices: np.ndarray,
    frequencies: np.ndarray,
    device: torch.device,
    batch_size: int,
    pulse_samples: int,
    chirp_rate: float,
    random_phase_seed: int | None = None,
) -> np.ndarray:
    weights = _filter_bank(frequencies, pulse_samples, chirp_rate, device, random_phase_seed)
    energy_kernel = torch.ones(1, 1, pulse_samples, device=device)
    count = len(frequencies)
    features = []
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
    with h5py.File(mat_path, "r") as handle:
        for start in range(0, len(sorted_indices), batch_size):
            selected = sorted_indices[start:start + batch_size]
            raw = np.asarray(handle["data_legacy"][:, selected], dtype=np.float32).T
            iq = raw.reshape(-1, 2, 3000)
            rms = np.sqrt(np.mean(np.square(iq).sum(axis=1), axis=1, keepdims=True))
            tensor = torch.from_numpy(iq / np.maximum(rms[:, None], 1.0e-8)).to(device)
            with torch.inference_mode():
                correlation = F.conv1d(tensor, weights)
                real, imag = correlation[:, :count], correlation[:, count:]
                local_energy = F.conv1d(
                    tensor.square().sum(dim=1, keepdim=True), energy_kernel
                ).clamp_min(1.0e-8)
                normalized = torch.sqrt(real.square() + imag.square() + 1.0e-12) / torch.sqrt(
                    local_energy * float(pulse_samples)
                )
                peaks, positions = normalized.max(dim=2)
            features.append(feature_matrix(
                peaks.cpu().numpy(), positions.cpu().numpy(), frequencies
            ))
    return np.concatenate(features)


def _spectral_features(mat_path: Path, indices: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
    with h5py.File(mat_path, "r") as handle:
        for start in range(0, len(sorted_indices), batch_size):
            selected = sorted_indices[start:start + batch_size]
            raw = np.asarray(handle["data_legacy"][:, selected], dtype=np.float32).T.reshape(-1, 2, 3000)
            signal = raw[:, 0] + 1j * raw[:, 1]
            signal /= np.maximum(np.sqrt(np.mean(np.abs(signal) ** 2, axis=1, keepdims=True)), 1.0e-8)
            spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal, axis=1), axes=1)).astype(np.float32)
            spectrum /= np.maximum(spectrum.sum(axis=1, keepdims=True), 1.0e-8)
            bins = spectrum.reshape(len(signal), 60, 50).sum(axis=2)
            quantiles = np.quantile(spectrum, [0.25, 0.5, 0.75, 0.9, 0.99], axis=1).T
            entropy = -(spectrum * np.log(np.maximum(spectrum, 1.0e-12))).sum(axis=1, keepdims=True)
            peak_indices = np.argpartition(spectrum, -3, axis=1)[:, -3:]
            peak_values = np.take_along_axis(spectrum, peak_indices, axis=1)
            peak_positions = (peak_indices.astype(np.float32) - 1500.0) / 1500.0
            outputs.append(np.concatenate((bins, quantiles, entropy, peak_values, peak_positions), axis=1).astype(np.float32))
    return np.concatenate(outputs)


def _fit_and_score(
    matrices: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    seed: int,
    shuffled: bool = False,
) -> dict[str, float]:
    train_target = targets["train"].copy()
    if shuffled:
        train_target = np.random.default_rng(seed + 991).permutation(train_target)
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(matrices["train"], train_target)
    result = {}
    for role in ("early", "development"):
        probability = model.predict_proba(matrices[role])[:, 1]
        result[f"{role}_aupr"] = float(average_precision_score(targets[role], probability))
        result[f"{role}_auroc"] = float(roc_auc_score(targets[role], probability))
    return result


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
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/a1_vft_mechanism_controls_20260725"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    targets = {role: labels[index, TARGET_INDEX] for role, index in roles.items()}
    prevalence = float(targets["development"].mean())
    frequencies = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)
    device = torch.device(args.device)
    controls = {
        "correct_lfm": {"pulse_samples": 1000, "chirp_rate": 5.0e10},
        "tone_zero_chirp": {"pulse_samples": 1000, "chirp_rate": 0.0},
        "wrong_chirp_0p8": {"pulse_samples": 1000, "chirp_rate": 4.0e10},
        "wrong_chirp_1p2": {"pulse_samples": 1000, "chirp_rate": 6.0e10},
        "wrong_pulse_0p8": {"pulse_samples": 800, "chirp_rate": 5.0e10},
        "wrong_pulse_1p2": {"pulse_samples": 1200, "chirp_rate": 5.0e10},
        "random_phase": {"pulse_samples": 1000, "chirp_rate": 0.0, "random_phase_seed": 20260725},
    }
    rows = []
    feature_hashes = {}
    for name, config in controls.items():
        started = time.perf_counter()
        matrices = {
            role: _template_features(
                mat_path, index, frequencies, device, args.batch_size,
                config["pulse_samples"], config["chirp_rate"], config.get("random_phase_seed"),
            )
            for role, index in roles.items()
        }
        score = _fit_and_score(matrices, targets, seed)
        shuffled_score = _fit_and_score(matrices, targets, seed, shuffled=True) if name == "correct_lfm" else {}
        row = {"control": name, "feature_dimension": int(matrices["train"].shape[1]), **config, **score,
               **{f"shuffled_{key}": value for key, value in shuffled_score.items()},
               "seconds": time.perf_counter() - started}
        rows.append(row)
        feature_hashes[name] = hashlib.sha256(matrices["development"].tobytes()).hexdigest()
        print(json.dumps(row), flush=True)
    started = time.perf_counter()
    spectral = {role: _spectral_features(mat_path, index, args.batch_size) for role, index in roles.items()}
    spectral_score = _fit_and_score(spectral, targets, seed)
    rows.append({"control": "generic_spectrum", "feature_dimension": int(spectral["train"].shape[1]),
                 **spectral_score, "seconds": time.perf_counter() - started})
    feature_hashes["generic_spectrum"] = hashlib.sha256(spectral["development"].tobytes()).hexdigest()
    rows.sort(key=lambda row: row["development_aupr"], reverse=True)
    correct = next(row for row in rows if row["control"] == "correct_lfm")
    strongest_wrong = max(row["development_aupr"] for row in rows if row["control"] != "correct_lfm")
    result = {
        "protocol": {
            "seed": seed, "mat_path": str(mat_path), "roles": {key: int(len(value)) for key, value in roles.items()},
            "frequency_grid_hz": frequencies.tolist(), "classifier": "fixed HistGradientBoosting leaf7",
            "selection_role": "none; all controls predeclared", "locked_test_used": False,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "development_prevalence": prevalence,
        "correct_template_development_aupr": correct["development_aupr"],
        "strongest_wrong_control_development_aupr": strongest_wrong,
        "correct_margin": float(correct["development_aupr"] - strongest_wrong),
        "shuffled_development_aupr": correct["shuffled_development_aupr"],
        "mechanism_gate_passed": bool(
            correct["development_aupr"] > strongest_wrong
            and abs(correct["shuffled_development_aupr"] - prevalence) <= 0.05
        ),
        "feature_hashes": feature_hashes,
        "controls": rows,
        "locked_test_used": False,
    }
    _write_csv(args.output_root / "controls.csv", rows)
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
