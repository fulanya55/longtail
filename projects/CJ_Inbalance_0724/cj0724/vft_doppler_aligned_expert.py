"""P1h reference-aligned VFT Doppler expert on fresh compound waveforms.

The expert uses the transmitted LFM reference, searches the physically valid
arrival-delay interval, dechirps each candidate, and estimates Doppler from a
local ambiguity grid.  No test labels are used for fitting or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import SIGNAL_LENGTH, V73IQDataset
from cj0724.dual_axis import read_identity_metadata


FS_HZ = 10.0e6
PULSE_SAMPLES = 1000
CHIRP_RATE_HZ_PER_S = 5.0e10
VFT_CLASS_INDEX = 4
VFT_DOPPLER_COLUMN = 49
POWERS = (20, 25)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def lfm_reference(
    device: torch.device,
    chirp_rate_hz_per_s: float = CHIRP_RATE_HZ_PER_S,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    time_s = torch.arange(PULSE_SAMPLES, device=device, dtype=torch.float32) / FS_HZ
    pulse = torch.polar(
        torch.ones_like(time_s), torch.pi * float(chirp_rate_hz_per_s) * time_s.square()
    ).to(dtype)
    reference = torch.zeros(SIGNAL_LENGTH, device=device, dtype=dtype)
    reference[:PULSE_SAMPLES] = pulse
    return reference


@torch.inference_mode()
def estimate_aligned_doppler(
    iq: torch.Tensor,
    chirp_rate_hz_per_s: float = CHIRP_RATE_HZ_PER_S,
    delay_min_samples: int = 100,
    delay_max_samples: int = 500,
    delay_candidates: int = 8,
    doppler_min_hz: float = 1000.0,
    doppler_max_hz: float = 7000.0,
    doppler_step_hz: float = 100.0,
    search_delay: bool = True,
) -> dict[str, torch.Tensor]:
    """Estimate VFT Doppler from the reference-aligned delay-Doppler peak."""
    if iq.ndim != 3 or tuple(iq.shape[1:]) != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [B,2,{SIGNAL_LENGTH}]")
    if not 0 <= delay_min_samples <= delay_max_samples < SIGNAL_LENGTH:
        raise ValueError("invalid delay search interval")
    if doppler_step_hz <= 0 or doppler_max_hz < doppler_min_hz:
        raise ValueError("invalid Doppler grid")

    signal = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    signal = signal / signal.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    reference = lfm_reference(signal.device, chirp_rate_hz_per_s)
    pulse = reference[:PULSE_SAMPLES]
    pulse_index = torch.arange(PULSE_SAMPLES, device=signal.device)

    if search_delay:
        correlation = torch.fft.ifft(
            torch.fft.fft(signal) * torch.fft.fft(reference).conj()
        ).abs()[:, delay_min_samples : delay_max_samples + 1]
        candidate_count = min(int(delay_candidates), int(correlation.shape[1]))
        candidate_delay = (
            torch.topk(correlation, candidate_count, dim=1).indices + delay_min_samples
        )
    else:
        candidate_delay = torch.zeros(
            (len(signal), 1), dtype=torch.long, device=signal.device
        )

    gather_index = (candidate_delay[:, :, None] + pulse_index[None, None, :]) % SIGNAL_LENGTH
    expanded = signal[:, None, :].expand(-1, candidate_delay.shape[1], -1)
    dechirped = torch.gather(expanded, 2, gather_index) * pulse.conj()[None, None, :]

    count = int(round((doppler_max_hz - doppler_min_hz) / doppler_step_hz)) + 1
    frequency = torch.linspace(
        doppler_min_hz, doppler_max_hz, count, device=signal.device, dtype=torch.float32
    )
    basis = torch.exp(
        -2j * torch.pi * frequency[:, None] * (pulse_index[None, :] / FS_HZ)
    )
    ambiguity = torch.matmul(dechirped, basis.T).abs()
    ambiguity = ambiguity / dechirped.abs().sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    flattened = ambiguity.flatten(1)
    best = flattened.argmax(dim=1)
    frequency_index = best % len(frequency)
    delay_slot = torch.div(best, len(frequency), rounding_mode="floor")
    batch_index = torch.arange(len(signal), device=signal.device)
    return {
        "prediction_hz": frequency[frequency_index].float(),
        "peak_score": flattened[batch_index, best].float(),
        "delay_samples": candidate_delay[batch_index, delay_slot].float(),
    }


def transition_file(root: Path, split: str, power: int) -> Path:
    canonical = list(range(10, 51, 5))
    index = canonical.index(int(power)) + 1
    path = root / f"level_{index:02d}__jnr_{power}dB" / f"closedset_6class_{split}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_vft_metadata(path: Path) -> dict[str, np.ndarray]:
    identities, labels = read_identity_metadata(path)
    with h5py.File(path, "r") as handle:
        target = np.asarray(handle["parameter_values"][VFT_DOPPLER_COLUMN, :], dtype=np.float64)
        mask = np.asarray(handle["parameter_mask"][VFT_DOPPLER_COLUMN, :], dtype=np.uint8).astype(bool)
    positive = (labels[:, VFT_CLASS_INDEX] > 0) & mask & np.isfinite(target)
    return {
        "identity": identities,
        "labels": labels,
        "target": target,
        "positive": positive,
    }


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(target) < 2 or np.std(prediction) <= 0 or np.std(target) <= 0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def regression_metrics(
    prediction: np.ndarray, target: np.ndarray, baseline_median: float
) -> dict[str, float]:
    mae = float(np.mean(np.abs(prediction - target)))
    baseline_mae = float(np.mean(np.abs(float(baseline_median) - target)))
    return {
        "sample_count": int(len(target)),
        "mae_hz": mae,
        "constant_median_mae_hz": baseline_mae,
        "relative_mae_reduction_vs_constant": float(1.0 - mae / max(baseline_mae, 1.0e-12)),
        "pearson_correlation": _pearson(prediction, target),
        "prediction_std_hz": float(np.std(prediction)),
        "target_std_hz": float(np.std(target)),
    }


@torch.inference_mode()
def predict_file(
    path: Path,
    metadata: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    estimator_kwargs: dict[str, object],
) -> dict[str, np.ndarray]:
    indices = np.flatnonzero(metadata["positive"]).astype(np.int64)
    dataset = V73IQDataset(path, indices, [VFT_DOPPLER_COLUMN])
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = {"sample_index": [], "prediction": [], "peak_score": [], "delay_samples": []}
    for batch in loader:
        estimate = estimate_aligned_doppler(
            batch["iq"].to(device, non_blocking=True), **estimator_kwargs
        )
        output["sample_index"].append(batch["sample_index"].numpy())
        output["prediction"].append(estimate["prediction_hz"].cpu().numpy())
        output["peak_score"].append(estimate["peak_score"].cpu().numpy())
        output["delay_samples"].append(estimate["delay_samples"].cpu().numpy())
    dataset.close()
    return {name: np.concatenate(parts) for name, parts in output.items()}


def evaluate_seed(
    root: Path,
    baseline_median: float,
    batch_size: int,
    device: torch.device,
    protocol: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    variants = {
        "correct_reference": {},
        "wrong_reference_0p8": {"chirp_rate_hz_per_s": 0.8 * CHIRP_RATE_HZ_PER_S},
        "wrong_reference_1p2": {"chirp_rate_hz_per_s": 1.2 * CHIRP_RATE_HZ_PER_S},
        "no_delay_alignment": {"search_delay": False},
    }
    saved: dict[str, list[np.ndarray]] = {
        "identity": [], "power_db": [], "target_hz": [], "combination_size": []
    }
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    scores: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    delays: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    identity_reference = None
    for power in POWERS:
        path = transition_file(root, "test", power)
        metadata = read_vft_metadata(path)
        if identity_reference is None:
            identity_reference = metadata["identity"]
        elif not np.array_equal(identity_reference, metadata["identity"]):
            raise RuntimeError("power views do not share identity order")
        selected = np.flatnonzero(metadata["positive"])
        saved["identity"].append(metadata["identity"][selected])
        saved["power_db"].append(np.full(len(selected), power, dtype=np.int16))
        saved["target_hz"].append(metadata["target"][selected])
        saved["combination_size"].append(metadata["labels"][selected].sum(axis=1).astype(np.int8))
        for name, override in variants.items():
            kwargs = dict(protocol["estimator"])
            kwargs.update(override)
            result = predict_file(path, metadata, batch_size, device, kwargs)
            if not np.array_equal(result["sample_index"], selected):
                raise RuntimeError("prediction order differs from positive metadata")
            predictions[name].append(result["prediction"])
            scores[name].append(result["peak_score"])
            delays[name].append(result["delay_samples"])

    arrays = {name: np.concatenate(parts) for name, parts in saved.items()}
    for name in variants:
        arrays[f"{name}_prediction_hz"] = np.concatenate(predictions[name])
        arrays[f"{name}_peak_score"] = np.concatenate(scores[name])
        arrays[f"{name}_delay_samples"] = np.concatenate(delays[name])
    np.savez_compressed(output_dir / "test_per_sample.npz", **arrays)

    metrics = {}
    for name in variants:
        prediction = arrays[f"{name}_prediction_hz"]
        metrics[name] = {
            "overall": regression_metrics(prediction, arrays["target_hz"], baseline_median),
            "compound_only": regression_metrics(
                prediction[arrays["combination_size"] >= 2],
                arrays["target_hz"][arrays["combination_size"] >= 2], baseline_median,
            ),
            "by_power": {
                str(power): regression_metrics(
                    prediction[arrays["power_db"] == power],
                    arrays["target_hz"][arrays["power_db"] == power], baseline_median,
                )
                for power in POWERS
            },
        }
    return {"root": str(root), "baseline_median_hz": baseline_median, "variants": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--output-root", default="outputs/p1h_vft_doppler_aligned_confirmation_20260731")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    roots = [project / value for value in args.data_roots]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = {
        "experiment": "P1h VFT reference-aligned dechirped Doppler expert",
        "estimator": {
            "chirp_rate_hz_per_s": CHIRP_RATE_HZ_PER_S,
            "delay_min_samples": 100,
            "delay_max_samples": 500,
            "delay_candidates": 8,
            "doppler_min_hz": 1000.0,
            "doppler_max_hz": 7000.0,
            "doppler_step_hz": 100.0,
            "search_delay": True,
        },
        "target": {"class": "VFT", "parameter": "doppler_Hz", "column": VFT_DOPPLER_COLUMN},
        "training": "deterministic physical expert; train role is used only for the constant-median baseline",
        "controls": ["wrong_reference_0p8", "wrong_reference_1p2", "no_delay_alignment"],
        "promotion_gates": {
            "mean_mae_reduction_vs_constant_min": 0.10,
            "mean_pearson_min": 0.30,
            "three_of_three_positive_mae_and_correlation": True,
            "compound_mean_mae_reduction_min": 0.10,
            "compound_mean_pearson_min": 0.30,
            "correct_mae_advantage_vs_best_control_min": 0.20,
            "correct_beats_controls_three_of_three": True,
        },
        "test_used_at_freeze": False,
        "test_fit_or_selection": False,
        "confirm_used": False,
        "locked_test_used": False,
    }

    train_receipts = []
    for root in roots:
        metadata_by_power = {
            power: read_vft_metadata(transition_file(root, "train", power)) for power in POWERS
        }
        first = metadata_by_power[POWERS[0]]
        for power in POWERS[1:]:
            current = metadata_by_power[power]
            if not np.array_equal(first["identity"], current["identity"]):
                raise RuntimeError("train power identities are not aligned")
            if not np.array_equal(first["labels"], current["labels"]):
                raise RuntimeError("train power labels are not aligned")
        selected = first["positive"]
        baseline_median = float(np.median(first["target"][selected]))
        train_receipts.append({
            "root": str(root),
            "identity_count": int(len(first["identity"])),
            "identity_sha256": _hash_int64(first["identity"]),
            "vft_positive_count": int(selected.sum()),
            "baseline_median_hz": baseline_median,
        })
    protocol["train_receipts"] = train_receipts
    freeze_path = output_root / "method_freeze.json"
    freeze_path.write_text(json.dumps(_jsonable(protocol), indent=2), encoding="utf-8")
    freeze_hash = _sha256(freeze_path)
    (output_root / "test_access_started.json").write_text(json.dumps({
        "method_freeze_sha256": freeze_hash,
        "purpose": "one-time fresh P1h test evaluation after deterministic method freeze",
        "started_unix": time.time(),
    }, indent=2), encoding="utf-8")

    results = []
    for index, (root, train_receipt) in enumerate(zip(roots, train_receipts), start=1):
        seed_dir = output_root / f"data_seed_{index}"
        seed_dir.mkdir()
        result = evaluate_seed(
            root, train_receipt["baseline_median_hz"], args.batch_size,
            device, protocol, seed_dir,
        )
        result.update({
            "data_seed_index": index,
            "method_freeze_sha256": freeze_hash,
            "test_used_once": True,
            "test_fit_or_selection": False,
            "confirm_used": False,
            "locked_test_used": False,
        })
        (seed_dir / "test_result.json").write_text(
            json.dumps(_jsonable(result), indent=2), encoding="utf-8"
        )
        results.append(result)

    correct = [row["variants"]["correct_reference"]["overall"] for row in results]
    compound = [row["variants"]["correct_reference"]["compound_only"] for row in results]
    control_mae = []
    per_seed_control_win = []
    for row in results:
        correct_mae = row["variants"]["correct_reference"]["overall"]["mae_hz"]
        best_control = min(
            row["variants"][name]["overall"]["mae_hz"]
            for name in ("wrong_reference_0p8", "wrong_reference_1p2", "no_delay_alignment")
        )
        control_mae.append(best_control)
        per_seed_control_win.append(correct_mae < best_control)
    mean_correct_mae = float(np.mean([row["mae_hz"] for row in correct]))
    mean_best_control_mae = float(np.mean(control_mae))
    control_advantage = float(1.0 - mean_correct_mae / mean_best_control_mae)
    gates = {
        "mean_mae_reduction_at_least_0p10": bool(np.mean([
            row["relative_mae_reduction_vs_constant"] for row in correct
        ]) >= 0.10),
        "mean_pearson_at_least_0p30": bool(np.mean([
            row["pearson_correlation"] for row in correct
        ]) >= 0.30),
        "three_of_three_positive_mae_and_correlation": bool(all(
            row["relative_mae_reduction_vs_constant"] > 0 and row["pearson_correlation"] > 0
            for row in correct
        )),
        "compound_mean_mae_reduction_at_least_0p10": bool(np.mean([
            row["relative_mae_reduction_vs_constant"] for row in compound
        ]) >= 0.10),
        "compound_mean_pearson_at_least_0p30": bool(np.mean([
            row["pearson_correlation"] for row in compound
        ]) >= 0.30),
        "correct_mae_advantage_vs_best_control_at_least_0p20": bool(control_advantage >= 0.20),
        "correct_beats_controls_three_of_three": bool(all(per_seed_control_win)),
    }
    summary = {
        "experiment": protocol["experiment"],
        "correct_reference": {
            "mean_mae_hz": mean_correct_mae,
            "mean_relative_mae_reduction_vs_constant": float(np.mean([
                row["relative_mae_reduction_vs_constant"] for row in correct
            ])),
            "mean_pearson_correlation": float(np.mean([
                row["pearson_correlation"] for row in correct
            ])),
            "per_seed_mae_reduction": [
                row["relative_mae_reduction_vs_constant"] for row in correct
            ],
            "per_seed_pearson": [row["pearson_correlation"] for row in correct],
            "compound_mean_mae_reduction": float(np.mean([
                row["relative_mae_reduction_vs_constant"] for row in compound
            ])),
            "compound_mean_pearson": float(np.mean([
                row["pearson_correlation"] for row in compound
            ])),
        },
        "best_control_mean_mae_hz": mean_best_control_mae,
        "correct_mae_advantage_vs_best_control": control_advantage,
        "gates": gates,
        "innovation_validated": bool(all(gates.values())),
        "method_freeze_sha256": freeze_hash,
        "elapsed_seconds": float(time.perf_counter() - started),
        "test_used_once": True,
        "test_fit_or_selection": False,
        "confirm_used": False,
        "locked_test_used": False,
    }
    (output_root / "results.json").write_text(
        json.dumps(_jsonable(results), indent=2), encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
