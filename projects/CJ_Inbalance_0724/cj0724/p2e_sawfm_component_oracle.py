"""P2e exact SawFM-component oracle sufficiency for the frozen period expert."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cj0724.data import V73IQDataset, complex_rms_normalize
from cj0724.dual_axis import read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1k_joint_level3_system import SAW_PARAMETER_COLUMN, _hash_int64
from cj0724.physics_parameter_specialists import build_physics_specialist


SAMPLE_RATE_HZ = 10.0e6
SIGNAL_LENGTH = 3000
SAW_CLASS_INDEX = 1
SAW_BANDWIDTH_COLUMN = 12
SAW_PERIOD_COLUMN = 14
SAW_DELAY_COLUMN = 15
SEEDS = (2041, 2042, 2043)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_sawfm_component(
    bandwidth_hz: np.ndarray,
    period_s: np.ndarray,
    delay_s: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    signal_length: int = SIGNAL_LENGTH,
) -> np.ndarray:
    bandwidth_hz = np.asarray(bandwidth_hz, dtype=np.float64).reshape(-1)
    period_s = np.asarray(period_s, dtype=np.float64).reshape(-1)
    delay_s = np.asarray(delay_s, dtype=np.float64).reshape(-1)
    if not (len(bandwidth_hz) == len(period_s) == len(delay_s)):
        raise ValueError("SawFM parameter arrays must have equal length")
    if np.any(bandwidth_hz <= 0.0) or np.any(period_s <= 0.0) or np.any(delay_s < 0.0):
        raise ValueError("SawFM oracle parameters are outside the physical domain")
    time_s = np.arange(int(signal_length), dtype=np.float64) / float(sample_rate_hz)
    instantaneous_frequency = (
        -bandwidth_hz[:, None] / 2.0
        + bandwidth_hz[:, None] * np.mod(time_s[None, :], period_s[:, None]) / period_s[:, None]
    )
    phase = 2.0 * np.pi * np.cumsum(instantaneous_frequency, axis=1) / float(sample_rate_hz)
    signal = np.exp(1j * phase)
    delays = np.rint(delay_s * float(sample_rate_hz)).astype(np.int64)
    for row, delay in enumerate(delays):
        signal[row] = np.roll(signal[row], int(delay))
    iq = np.stack((signal.real, signal.imag), axis=1).astype(np.float32)
    normalized = np.empty_like(iq)
    for row in range(len(iq)):
        normalized[row] = complex_rms_normalize(iq[row])
    return normalized


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = np.abs(prediction - target)
    correlation = 0.0
    if np.std(prediction) > 0.0 and np.std(target) > 0.0:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {"mae": float(error.mean()), "mae_us": float(error.mean() * 1.0e6), "pearson": correlation}


def reduction(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    return float(1.0 - candidate["mae"] / baseline["mae"])


@torch.inference_mode()
def predict_tensor(
    model: torch.nn.Module,
    iq: np.ndarray,
    device: torch.device,
    mean: float,
    scale: float,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(iq)), batch_size=batch_size, shuffle=False)
    output = []
    model.eval()
    for (batch,) in loader:
        prediction = model(batch.to(device))["prediction"] * float(scale) + float(mean)
        output.append(prediction.cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def predict_raw(
    model: torch.nn.Module,
    path: Path,
    indices: np.ndarray,
    device: torch.device,
    mean: float,
    scale: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = V73IQDataset(path, indices, [SAW_PARAMETER_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    predictions, targets = [], []
    for batch in loader:
        prediction = model(batch["iq"].to(device))["prediction"] * float(scale) + float(mean)
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["parameter_values"][:, 0].numpy())
    dataset.close()
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="generated_data/p2e_sawfm_component_oracle_seed1600000000_rechunked/level_09__jnr_50dB/closedset_6class_train.mat")
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--preregister", default="configs/p2e_sawfm_component_oracle_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2e_sawfm_component_oracle_sufficiency_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data_file = project / args.data_file
    reference_root = project / args.reference_freeze_root
    specialist_root = project / args.specialist_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)

    identities, labels = read_identity_metadata(data_file)
    exclusions = []
    for raw_path in args.excluded_identity_files:
        path = Path(raw_path)
        prior, _ = read_identity_metadata(path)
        exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
    if any(row["overlap_count"] != 0 for row in exclusions):
        raise RuntimeError("P2e identities overlap a prior split")
    with h5py.File(data_file, "r") as handle:
        parameter_values = np.asarray(handle["parameter_values"], dtype=np.float64)
        parameter_mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    selected = np.flatnonzero((labels[:, SAW_CLASS_INDEX] > 0) & parameter_mask[SAW_PERIOD_COLUMN])
    if len(selected) != 320:
        raise RuntimeError(f"expected 320 SawFM-positive oracle identities, got {len(selected)}")
    bandwidth = parameter_values[SAW_BANDWIDTH_COLUMN, selected]
    period = parameter_values[SAW_PERIOD_COLUMN, selected]
    delay = parameter_values[SAW_DELAY_COLUMN, selected]
    oracle_iq = reconstruct_sawfm_component(bandwidth, period, delay)
    constant_prediction = np.full(len(selected), float(np.median(period)), dtype=np.float64)
    constant_metrics = metrics(constant_prediction, period)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in SEEDS:
        freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
        mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
        checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
        model = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        raw_prediction, raw_target = predict_raw(model, data_file, selected, device, mean, scale, args.batch_size)
        oracle_prediction = predict_tensor(model, oracle_iq, device, mean, scale, args.batch_size)
        if not np.allclose(raw_target, period.astype(np.float32), rtol=0.0, atol=1.0e-10):
            raise RuntimeError("raw and oracle SawFM targets differ")
        raw_metrics = metrics(raw_prediction, period)
        oracle_metrics = metrics(oracle_prediction, period)
        run = {
            "seed": seed,
            "constant": constant_metrics,
            "raw_mixture": raw_metrics,
            "exact_component_oracle": oracle_metrics,
            "oracle_reduction_vs_constant": reduction(oracle_metrics, constant_metrics),
            "oracle_reduction_vs_raw": reduction(oracle_metrics, raw_metrics),
            "oracle_correlation_gain_vs_raw": float(oracle_metrics["pearson"] - raw_metrics["pearson"]),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        np.savez_compressed(
            seed_dir / "predictions.npz", sample_index=selected, target=period,
            raw_prediction=raw_prediction, oracle_prediction=oracle_prediction,
        )
        (seed_dir / "receipt.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
        runs.append(run)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"seed": seed, **{key: run[key] for key in ("oracle_reduction_vs_constant", "oracle_reduction_vs_raw", "oracle_correlation_gain_vs_raw")}}), flush=True)

    gates = {
        "each_seed_oracle_reduction_vs_constant_at_least_0p50": bool(all(run["oracle_reduction_vs_constant"] >= 0.50 for run in runs)),
        "each_seed_oracle_pearson_at_least_0p80": bool(all(run["exact_component_oracle"]["pearson"] >= 0.80 for run in runs)),
        "each_seed_oracle_reduction_vs_raw_at_least_0p50": bool(all(run["oracle_reduction_vs_raw"] >= 0.50 for run in runs)),
        "each_seed_oracle_correlation_gain_vs_raw_at_least_0p30": bool(all(run["oracle_correlation_gain_vs_raw"] >= 0.30 for run in runs)),
    }
    passed = bool(all(gates.values()))
    interpretation = (
        "Exact component isolation is physically sufficient; dedicated SawFM component supervision is allowed on a new development seed."
        if passed else
        "Exact isolation is insufficient for the frozen period expert; stop separator expansion and redesign the period estimator."
    )
    summary = {
        "experiment": "P2e exact SawFM component oracle sufficiency",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "identity_count": int(len(identities)),
        "identity_sha256": _hash_int64(identities),
        "SawFM_positive_count": int(len(selected)),
        "SawFM_positive_identity_sha256": _hash_int64(identities[selected]),
        "excluded_prior_splits": exclusions,
        "oracle_reconstruction": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "signal_length": SIGNAL_LENGTH,
            "noise_added": False,
            "deployable": False,
        },
        "promotion_gates": gates,
        "promotion_passed": passed,
        "interpretation": interpretation,
        "runs": runs,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
