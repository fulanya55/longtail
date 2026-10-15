"""Evaluate exact-component ISRJ sample-width identifiability across OOD intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_prominences

from cj0724.data import SIGNAL_LENGTH, V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64


FS_HZ = 10.0e6
PULSE_SAMPLES = 1000
CHIRP_RATE_HZ_PER_S = 5.0e10
POWERS = (25, 35, 50)
REGIMES = ("ID_control", "near_low", "near_high", "far_low", "far_high")
ISRJ_CLASS_INDEX = 5
WIDTH_COLUMN = 53
REPLICAS_COLUMN = 54
ARRIVAL_COLUMN = 56
CONSTANT_SECONDS = 12.5e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_signal() -> np.ndarray:
    time_s = np.arange(PULSE_SAMPLES, dtype=np.float64) / FS_HZ
    pulse = np.exp(1j * np.pi * CHIRP_RATE_HZ_PER_S * np.square(time_s))
    reference = np.zeros(SIGNAL_LENGTH, dtype=np.complex128)
    reference[:PULSE_SAMPLES] = pulse
    return reference


def reconstruct_isrj_component(
    width_seconds: np.ndarray,
    replicas: np.ndarray,
    arrival_seconds: np.ndarray,
) -> np.ndarray:
    reference = reference_signal()
    output = np.zeros((len(width_seconds), 2, SIGNAL_LENGTH), dtype=np.float32)
    sample_index = np.arange(SIGNAL_LENGTH)
    for position, (width_s, replica_value, arrival_s) in enumerate(zip(width_seconds, replicas, arrival_seconds)):
        width = max(1, int(round(float(width_s) * FS_HZ)))
        replica_count = int(round(float(replica_value)))
        period = (1 + replica_count) * width
        sampled = reference * (np.mod(sample_index, period) < width)
        signal = np.zeros(SIGNAL_LENGTH, dtype=np.complex128)
        for replica in range(replica_count + 1):
            shift = replica * width
            if shift < SIGNAL_LENGTH:
                signal[shift:] += sampled[: SIGNAL_LENGTH - shift]
        arrival = int(round(float(arrival_s) * FS_HZ))
        signal = np.roll(signal, arrival)
        energy = max(float(np.mean(np.abs(signal) ** 2)), 1.0e-12)
        signal = signal / np.sqrt(energy)
        output[position, 0] = signal.real.astype(np.float32)
        output[position, 1] = signal.imag.astype(np.float32)
    return output


def estimate_width_acf(iq: np.ndarray) -> dict[str, np.ndarray]:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [N,2,{SIGNAL_LENGTH}]")
    prediction, prominence_score = [], []
    low, high = 10, 501
    for row in iq:
        signal = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        spectrum = np.fft.fft(signal, n=2 * SIGNAL_LENGTH)
        acf = np.fft.ifft(spectrum * spectrum.conj()).real[:SIGNAL_LENGTH]
        acf /= max(float(acf[0]), 1.0e-12)
        residual = acf - uniform_filter1d(acf, size=31, mode="nearest")
        peaks, _ = find_peaks(residual[low:high], prominence=0.005, distance=5)
        lags = peaks + low
        if len(lags):
            prominences = peak_prominences(residual, lags)[0]
            selected = int(np.argmax(prominences))
            lag = int(lags[selected])
            score = float(prominences[selected])
        else:
            lag = int(np.argmax(residual[low:high]) + low)
            score = float(residual[lag])
        prediction.append(lag / FS_HZ)
        prominence_score.append(score)
    return {
        "prediction": np.asarray(prediction, dtype=np.float64),
        "prominence": np.asarray(prominence_score, dtype=np.float64),
    }


def deterministic_shuffle(iq: np.ndarray) -> np.ndarray:
    permutation = np.random.default_rng(20260801).permutation(SIGNAL_LENGTH)
    return np.asarray(iq)[:, :, permutation]


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = np.abs(prediction - target)
    constant_error = np.abs(CONSTANT_SECONDS - target)
    pearson = 0.0
    if len(target) >= 2 and np.std(prediction) > 0 and np.std(target) > 0:
        pearson = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "count": int(len(target)),
        "mae_s": float(error.mean()),
        "mae_us": float(error.mean() * 1.0e6),
        "constant_mae_us": float(constant_error.mean() * 1.0e6),
        "mae_reduction_vs_constant": float(1.0 - error.mean() / max(constant_error.mean(), 1.0e-12)),
        "pearson": pearson,
        "catastrophic_fraction": float(np.mean(error > np.maximum(5.0e-6, 0.50 * target))),
    }


def raw_iq(path: Path, indices: np.ndarray) -> np.ndarray:
    dataset = V73IQDataset(path, indices, [WIDTH_COLUMN])
    try:
        return np.stack([dataset[position]["iq"].numpy() for position in range(len(dataset))])
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2w_isrj_width_ood_seed3900000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p2w_isrj_width_oracle_ood_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2w_isrj_width_oracle_ood_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded_paths = [Path(protocol["data_file"])]
    excluded_paths.extend(Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files)
    identity_reference, records = None, []
    for regime in REGIMES:
        files = transition_files(project / args.data_root_template.format(regime=regime), "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            values = np.asarray(handle["parameter_values"], dtype=np.float64)
            mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, ISRJ_CLASS_INDEX] > 0) & mask & np.isfinite(values[WIDTH_COLUMN]))
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2w count mismatch in {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P2w regimes are not identity paired")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2w identity overlap")
        target = values[WIDTH_COLUMN, selected]
        exact = reconstruct_isrj_component(target, values[REPLICAS_COLUMN, selected], values[ARRIVAL_COLUMN, selected])
        exact_result = estimate_width_acf(exact)
        shuffled_result = estimate_width_acf(deterministic_shuffle(exact))
        arrays = {
            "identity": [], "power": [], "combination_size": [], "target": [],
            "exact_prediction": [], "raw_prediction": [], "shuffled_prediction": [],
            "exact_prominence": [], "raw_prominence": [],
        }
        for power in POWERS:
            raw_result = estimate_width_acf(raw_iq(files[power], selected))
            arrays["identity"].append(identities[selected])
            arrays["power"].append(np.full(len(selected), power, dtype=np.int16))
            arrays["combination_size"].append(labels[selected].sum(axis=1).astype(np.int8))
            arrays["target"].append(target)
            arrays["exact_prediction"].append(exact_result["prediction"])
            arrays["raw_prediction"].append(raw_result["prediction"])
            arrays["shuffled_prediction"].append(shuffled_result["prediction"])
            arrays["exact_prominence"].append(exact_result["prominence"])
            arrays["raw_prominence"].append(raw_result["prominence"])
        combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
        compound = combined["combination_size"] >= 2
        variants = {}
        for name, key in (("C1_exact_acf", "exact_prediction"), ("C2_raw_acf", "raw_prediction"), ("C3_time_shuffled_exact", "shuffled_prediction")):
            variants[name] = {
                "overall": metrics(combined[key], combined["target"]),
                "compound": metrics(combined[key][compound], combined["target"][compound]),
                "by_power": {str(power): metrics(combined[key][combined["power"] == power], combined["target"][combined["power"] == power]) for power in POWERS},
            }
        run_dir = output_root / regime
        run_dir.mkdir()
        np.savez_compressed(run_dir / "predictions.npz", **combined)
        row = {"regime": regime, "variants": variants, "alignment": alignment, "exclusions": exclusions}
        (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
        records.append(row)
        print(json.dumps({"regime": regime, "exact": variants["C1_exact_acf"]["overall"], "raw": variants["C2_raw_acf"]["overall"]}), flush=True)

    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for row in records for ex in row["exclusions"])),
        "exact_each_regime_reduction_vs_constant_at_least_0p10": bool(all(row["variants"]["C1_exact_acf"]["overall"]["mae_reduction_vs_constant"] >= 0.10 for row in records)),
        "exact_each_regime_pearson_at_least_0p30": bool(all(row["variants"]["C1_exact_acf"]["overall"]["pearson"] >= 0.30 for row in records)),
        "exact_compound_each_regime_reduction_vs_constant_at_least_0p10": bool(all(row["variants"]["C1_exact_acf"]["compound"]["mae_reduction_vs_constant"] >= 0.10 for row in records)),
        "exact_compound_each_regime_pearson_at_least_0p30": bool(all(row["variants"]["C1_exact_acf"]["compound"]["pearson"] >= 0.30 for row in records)),
        "exact_each_regime_better_than_shuffled": bool(all(row["variants"]["C1_exact_acf"]["overall"]["mae_s"] < row["variants"]["C3_time_shuffled_exact"]["overall"]["mae_s"] for row in records)),
        "exact_each_regime_catastrophic_fraction_at_most_0p10": bool(all(row["variants"]["C1_exact_acf"]["overall"]["catastrophic_fraction"] <= 0.10 for row in records)),
    }
    passed = bool(all(gates.values()))
    summary_path = output_root / "summary.json"
    summary = {
        "experiment": "P2w ISRJ sample-width exact-component OOD identifiability",
        "preregister_sha256": sha256_file(preregister),
        "identity_audit": {"identity_count": int(len(identity_reference)), "identity_sha256": _hash_int64(identity_reference), "paired_across_regimes": True},
        "promotion_gates": gates,
        "development_passed": passed,
        "per_regime": {row["regime"]: row["variants"] for row in records},
        "interpretation": (
            "Exact-component width is physically identifiable; test isolation on a fresh seed."
            if passed else
            "The preregistered single-PRI ACF width mechanism is insufficient; do not attach or route it."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if passed:
        (output_root / "method_freeze.json").write_text(json.dumps({
            "method": "ISRJ exact-component residual-autocorrelation width estimator",
            "lag_samples": [10, 500],
            "background_window": 31,
            "peak_prominence": 0.005,
            "peak_distance": 5,
            "summary_sha256": sha256_file(summary_path),
            "preregister_sha256": sha256_file(preregister),
            "locked_test_used": False,
        }, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
