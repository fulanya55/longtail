"""Evaluate ISRJ sample width from the instantaneous-frequency reset sequence."""

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
    ARRIVAL_COLUMN,
    CONSTANT_SECONDS,
    FS_HZ,
    ISRJ_CLASS_INDEX,
    POWERS,
    REGIMES,
    REPLICAS_COLUMN,
    SIGNAL_LENGTH,
    WIDTH_COLUMN,
    deterministic_shuffle,
    estimate_width_acf,
    metrics,
    raw_iq,
    reconstruct_isrj_component,
    sha256_file,
)


ACTIVE_AMPLITUDE_RATIO = 0.001
RESET_THRESHOLD_RAD = 0.05
MIN_GAP_SAMPLES = 10
MAX_GAP_SAMPLES = 500


def _cluster_centres(events: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return np.empty(0, dtype=np.float64)
    groups = np.split(events, np.flatnonzero(np.diff(events) > 1) + 1)
    return np.asarray([float(np.mean(group)) for group in groups], dtype=np.float64)


def estimate_width_phase_resets(iq: np.ndarray) -> dict[str, np.ndarray]:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [N,2,{SIGNAL_LENGTH}]")
    predictions, confidences, reset_counts = [], [], []
    fallback_samples = int(round(CONSTANT_SECONDS * FS_HZ))
    for row in iq:
        signal = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        amplitude = np.abs(signal)
        peak_amplitude = max(float(amplitude.max()), 1.0e-12)
        active = amplitude >= ACTIVE_AMPLITUDE_RATIO * peak_amplitude
        phase_increment = np.angle(signal[1:] * signal[:-1].conj())
        phase_curvature = np.abs(
            np.angle(np.exp(1j * (phase_increment[1:] - phase_increment[:-1])))
        )
        valid = active[:-2] & active[1:-1] & active[2:]
        events = np.flatnonzero(valid & (phase_curvature >= RESET_THRESHOLD_RAD)) + 1
        centres = _cluster_centres(events)
        gaps = np.rint(np.diff(centres)).astype(np.int64)
        candidates = gaps[(gaps >= MIN_GAP_SAMPLES) & (gaps <= MAX_GAP_SAMPLES)]
        if len(candidates) == 0:
            selected = fallback_samples
            confidence = 0.0
        else:
            counts = np.bincount(candidates, minlength=MAX_GAP_SAMPLES + 1)
            modes = np.flatnonzero(counts == counts.max())
            median_gap = float(np.median(candidates))
            selected = int(sorted(modes.tolist(), key=lambda value: (abs(value - median_gap), value))[0])
            confidence = float(counts[selected] / len(candidates))
        predictions.append(selected / FS_HZ)
        confidences.append(confidence)
        reset_counts.append(len(centres))
    return {
        "prediction": np.asarray(predictions, dtype=np.float64),
        "confidence": np.asarray(confidences, dtype=np.float64),
        "reset_count": np.asarray(reset_counts, dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2x_isrj_phase_reset_seed4100000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p2x_isrj_phase_reset_ood_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2x_isrj_phase_reset_ood_development_20260801")
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
            raise RuntimeError(f"P2x count mismatch in {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P2x regimes are not identity paired")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2x identity overlap")

        target = values[WIDTH_COLUMN, selected]
        exact = reconstruct_isrj_component(target, values[REPLICAS_COLUMN, selected], values[ARRIVAL_COLUMN, selected])
        exact_reset = estimate_width_phase_resets(exact)
        shuffled_reset = estimate_width_phase_resets(deterministic_shuffle(exact))
        exact_acf = estimate_width_acf(exact)
        arrays = {
            "identity": [], "power": [], "combination_size": [], "target": [],
            "exact_reset_prediction": [], "raw_reset_prediction": [],
            "shuffled_reset_prediction": [], "exact_acf_prediction": [],
            "exact_reset_confidence": [], "raw_reset_confidence": [],
            "exact_reset_count": [], "raw_reset_count": [],
        }
        for power in POWERS:
            raw_reset = estimate_width_phase_resets(raw_iq(files[power], selected))
            arrays["identity"].append(identities[selected])
            arrays["power"].append(np.full(len(selected), power, dtype=np.int16))
            arrays["combination_size"].append(labels[selected].sum(axis=1).astype(np.int8))
            arrays["target"].append(target)
            arrays["exact_reset_prediction"].append(exact_reset["prediction"])
            arrays["raw_reset_prediction"].append(raw_reset["prediction"])
            arrays["shuffled_reset_prediction"].append(shuffled_reset["prediction"])
            arrays["exact_acf_prediction"].append(exact_acf["prediction"])
            arrays["exact_reset_confidence"].append(exact_reset["confidence"])
            arrays["raw_reset_confidence"].append(raw_reset["confidence"])
            arrays["exact_reset_count"].append(exact_reset["reset_count"])
            arrays["raw_reset_count"].append(raw_reset["reset_count"])
        combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
        compound = combined["combination_size"] >= 2
        variants = {}
        for name, key in (
            ("C1_exact_phase_reset", "exact_reset_prediction"),
            ("C2_raw_phase_reset", "raw_reset_prediction"),
            ("C3_time_shuffled_exact", "shuffled_reset_prediction"),
            ("C4_exact_acf", "exact_acf_prediction"),
        ):
            variants[name] = {
                "overall": metrics(combined[key], combined["target"]),
                "compound": metrics(combined[key][compound], combined["target"][compound]),
                "by_power": {
                    str(power): metrics(combined[key][combined["power"] == power], combined["target"][combined["power"] == power])
                    for power in POWERS
                },
            }
        run_dir = output_root / regime
        run_dir.mkdir()
        np.savez_compressed(run_dir / "predictions.npz", **combined)
        row = {"regime": regime, "variants": variants, "alignment": alignment, "exclusions": exclusions}
        (run_dir / "result.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
        records.append(row)
        print(json.dumps({"regime": regime, "reset": variants["C1_exact_phase_reset"]["overall"], "acf": variants["C4_exact_acf"]["overall"]}), flush=True)

    exact = lambda row, subset="overall": row["variants"]["C1_exact_phase_reset"][subset]
    shuffled = lambda row: row["variants"]["C3_time_shuffled_exact"]["overall"]
    far_low = next(row for row in records if row["regime"] == "far_low")
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for row in records for ex in row["exclusions"])),
        "exact_each_regime_reduction_vs_constant_at_least_0p10": bool(all(exact(row)["mae_reduction_vs_constant"] >= 0.10 for row in records)),
        "exact_each_regime_pearson_at_least_0p30": bool(all(exact(row)["pearson"] >= 0.30 for row in records)),
        "exact_compound_each_regime_reduction_vs_constant_at_least_0p10": bool(all(exact(row, "compound")["mae_reduction_vs_constant"] >= 0.10 for row in records)),
        "exact_compound_each_regime_pearson_at_least_0p30": bool(all(exact(row, "compound")["pearson"] >= 0.30 for row in records)),
        "exact_each_regime_better_than_shuffled": bool(all(exact(row)["mae_s"] < shuffled(row)["mae_s"] for row in records)),
        "exact_each_regime_catastrophic_fraction_at_most_0p10": bool(all(exact(row)["catastrophic_fraction"] <= 0.10 for row in records)),
        "far_low_mae_better_than_frozen_acf": bool(exact(far_low)["mae_s"] < far_low["variants"]["C4_exact_acf"]["overall"]["mae_s"]),
        "far_low_pearson_better_than_frozen_acf": bool(exact(far_low)["pearson"] > far_low["variants"]["C4_exact_acf"]["overall"]["pearson"]),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P2x ISRJ sample-width phase-reset-sequence OOD identifiability",
        "preregister_sha256": sha256_file(preregister),
        "identity_audit": {"identity_count": int(len(identity_reference)), "identity_sha256": _hash_int64(identity_reference), "paired_across_regimes": True},
        "promotion_gates": gates,
        "development_passed": passed,
        "per_regime": {row["regime"]: row["variants"] for row in records},
        "interpretation": (
            "The exact-component phase-reset sequence is physically sufficient; test isolation on a fresh seed."
            if passed else
            "The phase-reset sequence is insufficient; stop ISRJ-width expert development and retain the failure boundary."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
