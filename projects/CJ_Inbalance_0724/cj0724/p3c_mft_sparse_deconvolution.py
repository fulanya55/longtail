"""Evaluate MFT target count from a sparse delay impulse train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import (
    CHIRP_RATE_HZ_PER_S, FS_HZ, POWERS, PULSE_SAMPLES, SIGNAL_LENGTH,
    raw_iq, sha256_file,
)


REGIMES = ("ID_control", "low_count", "close_spacing", "ultra_close", "weak_target")
MFT_CLASS_INDEX = 3
COUNT_COLUMN = 28
DELAY_COLUMNS = np.arange(31, 39)
AMPLITUDE_COLUMNS = np.arange(39, 47)
CONSTANT_COUNT = 5.5


def reference_signal(chirp_rate: float = CHIRP_RATE_HZ_PER_S) -> np.ndarray:
    time_s = np.arange(PULSE_SAMPLES, dtype=np.float64) / FS_HZ
    pulse = np.exp(1j * np.pi * float(chirp_rate) * np.square(time_s))
    reference = np.zeros(SIGNAL_LENGTH, dtype=np.complex128)
    reference[:PULSE_SAMPLES] = pulse
    return reference


def reconstruct_mft_component(delays_s: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    delays_s = np.asarray(delays_s, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if delays_s.shape != amplitudes.shape or delays_s.ndim != 2 or delays_s.shape[1] != 8:
        raise ValueError("delays and amplitudes must have shape [N,8]")
    reference = reference_signal()
    output = np.zeros((len(delays_s), 2, SIGNAL_LENGTH), dtype=np.float32)
    for row in range(len(delays_s)):
        signal = np.zeros(SIGNAL_LENGTH, dtype=np.complex128)
        active = np.isfinite(delays_s[row]) & np.isfinite(amplitudes[row])
        for delay_s, amplitude in zip(delays_s[row, active], amplitudes[row, active]):
            signal += float(amplitude) * np.roll(reference, int(round(float(delay_s) * FS_HZ)))
        energy = max(float(np.mean(np.abs(signal) ** 2)), 1.0e-12)
        signal /= np.sqrt(energy)
        output[row, 0] = signal.real.astype(np.float32)
        output[row, 1] = signal.imag.astype(np.float32)
    return output


def estimate_sparse_count(iq: np.ndarray, chirp_rate: float = CHIRP_RATE_HZ_PER_S) -> dict[str, np.ndarray]:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [N,2,{SIGNAL_LENGTH}]")
    signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    reference = reference_signal(chirp_rate)
    reference_spectrum = np.fft.fft(reference)
    power = np.abs(reference_spectrum) ** 2
    regularizer = 1.0e-6 * float(power.max())
    impulse = np.fft.ifft(np.fft.fft(signal, axis=1) * reference_spectrum.conj()[None, :] / (power[None, :] + regularizer), axis=1)
    support = np.abs(impulse[:, :1501])
    normalized = support / np.maximum(support.max(axis=1, keepdims=True), 1.0e-12)
    raw_count = (normalized >= 0.02).sum(axis=1)
    prediction = np.clip(raw_count, 0, 8).astype(np.float64)
    return {"prediction": prediction, "raw_support_count": raw_count.astype(np.int32), "peak": normalized.max(axis=1)}


def deterministic_shuffle(iq: np.ndarray) -> np.ndarray:
    permutation = np.random.default_rng(20260801).permutation(SIGNAL_LENGTH)
    return np.asarray(iq)[:, :, permutation]


def count_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = np.abs(prediction - target)
    constant_error = np.abs(CONSTANT_COUNT - target)
    pearson = 0.0
    if np.std(prediction) > 0 and np.std(target) > 0:
        pearson = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "count": int(len(target)), "mae": float(error.mean()),
        "constant_mae": float(constant_error.mean()),
        "mae_reduction_vs_constant": float(1.0 - error.mean() / max(constant_error.mean(), 1.0e-12)),
        "pearson": pearson, "exact_accuracy": float(np.mean(prediction == target)),
        "error_ge_2_fraction": float(np.mean(error >= 2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p3c_mft_count_ood_seed450000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p3c_mft_sparse_deconvolution_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3c_mft_sparse_deconvolution_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    preregister = project / args.preregister
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
            mask = np.asarray(handle["parameter_mask"][COUNT_COLUMN, :], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, MFT_CLASS_INDEX] > 0) & mask & np.isfinite(values[COUNT_COLUMN]))
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P3c count mismatch in {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P3c regimes are not identity paired")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P3c identity overlap")

        target = values[COUNT_COLUMN, selected]
        delays = values[DELAY_COLUMNS[:, None], selected[None, :]].T
        amplitudes = values[AMPLITUDE_COLUMNS[:, None], selected[None, :]].T
        exact = reconstruct_mft_component(delays, amplitudes)
        exact_correct = estimate_sparse_count(exact)
        exact_wrong = estimate_sparse_count(exact, chirp_rate=0.8 * CHIRP_RATE_HZ_PER_S)
        exact_shuffled = estimate_sparse_count(deterministic_shuffle(exact))
        arrays = {"identity": [], "power": [], "combination_size": [], "target": [], "exact": [], "raw": [], "wrong": [], "shuffled": [], "exact_support_count": [], "raw_support_count": []}
        for power_level in POWERS:
            raw_result = estimate_sparse_count(raw_iq(files[power_level], selected))
            arrays["identity"].append(identities[selected])
            arrays["power"].append(np.full(len(selected), power_level, dtype=np.int16))
            arrays["combination_size"].append(labels[selected].sum(axis=1).astype(np.int8))
            arrays["target"].append(target)
            arrays["exact"].append(exact_correct["prediction"])
            arrays["raw"].append(raw_result["prediction"])
            arrays["wrong"].append(exact_wrong["prediction"])
            arrays["shuffled"].append(exact_shuffled["prediction"])
            arrays["exact_support_count"].append(exact_correct["raw_support_count"])
            arrays["raw_support_count"].append(raw_result["raw_support_count"])
        combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
        compound = combined["combination_size"] >= 2
        variants = {}
        for name, key in (("C1_exact_correct_reference", "exact"), ("C2_raw_correct_reference", "raw"), ("C3_exact_wrong_reference", "wrong"), ("C4_time_shuffled_exact", "shuffled")):
            variants[name] = {
                "overall": count_metrics(combined[key], combined["target"]),
                "compound": count_metrics(combined[key][compound], combined["target"][compound]),
                "by_power": {str(power_level): count_metrics(combined[key][combined["power"] == power_level], combined["target"][combined["power"] == power_level]) for power_level in POWERS},
            }
        run_dir = output_root / regime
        run_dir.mkdir()
        np.savez_compressed(run_dir / "predictions.npz", **combined)
        row = {"regime": regime, "variants": variants, "alignment": alignment, "exclusions": exclusions}
        (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
        records.append(row)
        print(json.dumps({"regime": regime, "exact": variants["C1_exact_correct_reference"]["overall"], "raw": variants["C2_raw_correct_reference"]["overall"]}), flush=True)

    exact = lambda row, subset="overall": row["variants"]["C1_exact_correct_reference"][subset]
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for row in records for ex in row["exclusions"])),
        "exact_each_regime_reduction_vs_constant_at_least_0p10": bool(all(exact(row)["mae_reduction_vs_constant"] >= 0.10 for row in records)),
        "exact_each_regime_pearson_at_least_0p30": bool(all(exact(row)["pearson"] >= 0.30 for row in records)),
        "exact_compound_each_regime_reduction_vs_constant_at_least_0p10": bool(all(exact(row, "compound")["mae_reduction_vs_constant"] >= 0.10 for row in records)),
        "exact_compound_each_regime_pearson_at_least_0p30": bool(all(exact(row, "compound")["pearson"] >= 0.30 for row in records)),
        "exact_each_regime_accuracy_at_least_0p90": bool(all(exact(row)["exact_accuracy"] >= 0.90 for row in records)),
        "exact_each_regime_better_than_wrong_and_shuffled": bool(all(exact(row)["mae"] < row["variants"][control]["overall"]["mae"] for row in records for control in ("C3_exact_wrong_reference", "C4_time_shuffled_exact"))),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P3c MFT sparse-delay deconvolution count oracle",
        "preregister_sha256": sha256_file(preregister),
        "identity_audit": {"identity_count": int(len(identity_reference)), "identity_sha256": _hash_int64(identity_reference), "paired_across_regimes": True},
        "promotion_gates": gates, "development_passed": passed,
        "per_regime": {row["regime"]: row["variants"] for row in records},
        "interpretation": "Sparse delay deconvolution is physically sufficient; test compound-IQ support denoising on new identities." if passed else "The frozen inverse-filter count mechanism is insufficient; do not train or attach an MFT count head.",
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
