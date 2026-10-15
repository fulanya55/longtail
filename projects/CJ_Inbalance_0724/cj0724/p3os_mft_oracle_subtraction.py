"""P3o-S: oracle MFT subtraction causal diagnosis for frozen classification.

This is deliberately an oracle diagnostic, not a deployable separator.  The
only changed inference input is the exact, generator-power-calibrated MFT
component (or an equal-energy non-MFT control) before the ordinary per-view
RMS normalization.  All classifier weights and safe thresholds are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from cj0724.data import SIGNAL_LENGTH, complex_rms_normalize
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_main
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import (
    AMPLITUDE_COLUMNS,
    DELAY_COLUMNS,
    MFT_CLASS_INDEX,
    REGIMES,
    reconstruct_mft_component,
)


MODEL_SEEDS = (2041, 2042, 2043)
ISRJ_CLASS_INDEX = 5
MFT_JNR_COLUMN = 27
BOOTSTRAP_RESAMPLES = 2000


def _hash_seed(identity: int, power: int) -> int:
    payload = f"p3os-phase-control:{int(identity)}:{int(power)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def rms_normalize_batch(iq: np.ndarray) -> np.ndarray:
    """Exact vectorized counterpart of :func:`complex_rms_normalize`."""
    values = np.asarray(iq, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"Expected [B, 2, {SIGNAL_LENGTH}], got {values.shape}")
    rms = np.sqrt(np.mean(np.square(values[:, 0], dtype=np.float64) + np.square(values[:, 1], dtype=np.float64), axis=1))
    if not np.all(np.isfinite(rms)) or np.any(rms <= 1.0e-12):
        raise ValueError("Cannot RMS-normalize a non-finite or zero input waveform")
    return (values / rms[:, None, None]).astype(np.float32)


def reconstruct_power_scaled_mft(
    delays_s: np.ndarray,
    amplitudes: np.ndarray,
    mft_jnr_db: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    """Reconstruct the exact MFT shape and apply generator-equivalent RMS scale.

    ``compose_jamming_sample`` calibrates every non-Traction component to
    ``noise_power_dB + effective JNR_dB``.  The returned P3c reconstruction is
    unit-RMS, so multiplying it by this target RMS reproduces that calibration.
    """
    jnr = np.asarray(mft_jnr_db, dtype=np.float64).reshape(-1)
    if not np.isfinite(noise_power) or float(noise_power) <= 0.0:
        raise ValueError("noise_power must be finite and positive")
    if not np.all(np.isfinite(jnr)):
        raise ValueError("MFT JNR must be finite for every reconstructed component")
    shape = reconstruct_mft_component(delays_s, amplitudes).astype(np.float64)
    target_power_db = 10.0 * np.log10(float(noise_power)) + jnr
    target_rms = np.power(10.0, target_power_db / 20.0)
    result = shape * target_rms[:, None, None]
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Non-finite power-scaled MFT reconstruction")
    return result.astype(np.float32)


def phase_scrambled_energy_control(component: np.ndarray, identities: np.ndarray, power: int) -> np.ndarray:
    """Return a deterministic, equal-energy non-MFT waveform control."""
    values = np.asarray(component, dtype=np.float32)
    identity = np.asarray(identities, dtype=np.int64).reshape(-1)
    if values.ndim != 3 or values.shape[1:] != (2, SIGNAL_LENGTH) or len(values) != len(identity):
        raise ValueError("component and identities have incompatible shapes")
    signal = values[:, 0].astype(np.float64) + 1j * values[:, 1].astype(np.float64)
    spectrum = np.fft.fft(signal, axis=1)
    for row, source_identity in enumerate(identity):
        phase = np.random.default_rng(_hash_seed(int(source_identity), int(power))).uniform(-np.pi, np.pi, SIGNAL_LENGTH)
        spectrum[row] *= np.exp(1j * phase)
    scrambled = np.fft.ifft(spectrum, axis=1)
    result = np.stack((scrambled.real, scrambled.imag), axis=1).astype(np.float32)
    before = np.mean(np.square(values[:, 0], dtype=np.float64) + np.square(values[:, 1], dtype=np.float64), axis=1)
    after = np.mean(np.square(result[:, 0], dtype=np.float64) + np.square(result[:, 1], dtype=np.float64), axis=1)
    if not np.allclose(before, after, rtol=1.0e-5, atol=1.0e-7):
        raise RuntimeError("Phase-scrambled control did not preserve component energy")
    return result


def transformed_iq(raw_iq: np.ndarray, mft_component: np.ndarray, identities: np.ndarray, power: int, condition: str) -> np.ndarray:
    """Apply C0/C1/C2 before the unchanged view-level RMS normalization."""
    raw = np.asarray(raw_iq, dtype=np.float32)
    component = np.asarray(mft_component, dtype=np.float32)
    if raw.shape != component.shape:
        raise ValueError("raw_iq and mft_component must have the same shape")
    if condition == "C0_raw":
        candidate = raw
    elif condition == "C1_exact_mft_subtraction":
        candidate = raw - component
    elif condition == "C2_energy_matched_non_mft_control":
        candidate = raw - phase_scrambled_energy_control(component, identities, power)
    else:
        raise ValueError(f"Unknown P3o-S condition {condition!r}")
    return rms_normalize_batch(candidate)


def paired_bootstrap_source_fpr_reduction(
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    labels: np.ndarray,
    identities: np.ndarray,
    threshold: float,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    """Identity-clustered paired bootstrap of FPR(A)-FPR(B) on MFT+ / ISRJ-."""
    a, b = np.asarray(probability_a), np.asarray(probability_b)
    label = np.asarray(labels)
    identity = np.asarray(identities, dtype=np.int64)
    source = (label[:, MFT_CLASS_INDEX] > 0) & (label[:, ISRJ_CLASS_INDEX] == 0)
    if a.shape != b.shape or len(a) != len(label) or len(label) != len(identity) or not bool(source.any()):
        raise ValueError("Invalid paired-bootstrap source arrays")
    unique = np.unique(identity)
    groups = [np.flatnonzero((identity == value) & source) for value in unique]
    groups = [group for group in groups if group.size]
    if len(groups) < 2:
        raise ValueError("Need at least two source identities for paired bootstrap")
    def reduction(rows: np.ndarray) -> float:
        return float((a[rows] >= threshold).mean() - (b[rows] >= threshold).mean())
    observed_rows = np.concatenate(groups)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), dtype=np.float64)
    for index in range(len(draws)):
        rows = np.concatenate([groups[int(choice)] for choice in rng.integers(0, len(groups), len(groups))])
        draws[index] = reduction(rows)
    return {
        "observed_reduction": reduction(observed_rows),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "resamples": int(resamples),
        "cluster_count": int(len(groups)),
    }


def _read_noise_power(handle: h5py.File) -> float:
    """Read generator noise power, following a rechunk-cache source receipt if needed."""
    dataset = handle.get("config/radar/noise_power")
    if dataset is None:
        source_path = handle.attrs.get("source_path")
        if source_path is None:
            raise RuntimeError("Missing generator config/radar/noise_power and cache source_path")
        source = Path(str(source_path))
        if not source.is_file():
            raise RuntimeError(f"Rechunk cache source_path does not exist: {source}")
        with h5py.File(source, "r") as source_handle:
            dataset = source_handle.get("config/radar/noise_power")
            if dataset is None:
                raise RuntimeError("Missing generator config/radar/noise_power in cache source")
            value = float(np.asarray(dataset).reshape(-1)[0])
    else:
        value = float(np.asarray(dataset).reshape(-1)[0])
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("Invalid generator noise_power")
    return value


def _raw_iq(handle: h5py.File, indices: np.ndarray) -> np.ndarray:
    legacy = np.asarray(handle["data_legacy"][:, indices], dtype=np.float32).T
    if legacy.shape != (len(indices), 2 * SIGNAL_LENGTH):
        raise RuntimeError(f"Unexpected data_legacy shape {legacy.shape}")
    return legacy.reshape(len(indices), 2, SIGNAL_LENGTH)


def _metrics(probability: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, regime: np.ndarray, power: np.ndarray) -> dict[str, object]:
    prediction = probability >= thresholds[None, :]
    isrj_truth = labels[:, ISRJ_CLASS_INDEX].astype(bool)
    source = (labels[:, MFT_CLASS_INDEX] > 0) & ~isrj_truth
    if not bool(source.any()) or not bool(isrj_truth.any()) or not bool((~isrj_truth).any()):
        raise RuntimeError("P3o-S missing required ISRJ/source subsets")
    aupr = [float(average_precision_score(labels[:, index], probability[:, index])) for index in range(labels.shape[1])]
    cells = []
    for regime_index, regime_name in enumerate(REGIMES):
        for power_value in POWERS:
            cell = (regime == regime_index) & (power == power_value)
            source_cell = cell & source
            truth_cell = cell & isrj_truth
            negative_cell = cell & ~isrj_truth
            if not bool(source_cell.any()) or not bool(truth_cell.any()) or not bool(negative_cell.any()):
                raise RuntimeError(f"P3o-S malformed cell {regime_name}/{power_value}")
            cells.append({
                "regime": regime_name,
                "power": int(power_value),
                "source_isrj_fpr": float(prediction[source_cell, ISRJ_CLASS_INDEX].mean()),
                "isrj_recall": float(prediction[truth_cell, ISRJ_CLASS_INDEX].mean()),
                "isrj_fpr": float(prediction[negative_cell, ISRJ_CLASS_INDEX].mean()),
            })
    return {
        "macro_aupr": float(np.mean(aupr)),
        "isrj_recall": float(prediction[isrj_truth, ISRJ_CLASS_INDEX].mean()),
        "isrj_fpr": float(prediction[~isrj_truth, ISRJ_CLASS_INDEX].mean()),
        "source_isrj_fpr": float(prediction[source, ISRJ_CLASS_INDEX].mean()),
        "worst_source_cell_fpr": float(max(row["source_isrj_fpr"] for row in cells)),
        "cells": cells,
    }


@torch.inference_mode()
def _predict_conditions_for_view(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    batch_size: int,
    regime_index: int,
    power: int,
) -> dict[str, np.ndarray]:
    """Run all frozen C0/C1/C2 inputs on one view without modifying the file."""
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ("identity", "labels", "regime", "power", "C0_raw", "C1_exact_mft_subtraction", "C2_energy_matched_non_mft_control")}
    with h5py.File(path, "r") as handle:
        labels = np.asarray(handle["label_selected"], dtype=np.uint8).T
        identities = np.rint(np.asarray(handle["receiver_noise_seed"], dtype=np.float64).reshape(-1)).astype(np.int64)
        values = handle["parameter_values"]
        masks = handle["parameter_mask"]
        noise_power = _read_noise_power(handle)
        if len(labels) != len(identities):
            raise RuntimeError("P3o-S identity/label length mismatch")
        for start in range(0, len(labels), int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), len(labels)), dtype=np.int64)
            batch_labels = labels[indices]
            raw = _raw_iq(handle, indices)
            component = np.zeros_like(raw)
            positive = batch_labels[:, MFT_CLASS_INDEX] > 0
            if bool(positive.any()):
                source_indices = indices[positive]
                jnr = np.asarray(values[MFT_JNR_COLUMN, source_indices], dtype=np.float64)
                jnr_mask = np.asarray(masks[MFT_JNR_COLUMN, source_indices], dtype=np.uint8).astype(bool)
                if not bool(jnr_mask.all()):
                    raise RuntimeError("MFT-positive row lacks a valid MFT.JNR target")
                delays = np.asarray(values[DELAY_COLUMNS, :], dtype=np.float64)[:, source_indices].T
                amplitudes = np.asarray(values[AMPLITUDE_COLUMNS, :], dtype=np.float64)[:, source_indices].T
                component[positive] = reconstruct_power_scaled_mft(delays, amplitudes, jnr, noise_power)
            batch_identities = identities[indices]
            model_inputs = [
                transformed_iq(raw, component, batch_identities, power, condition)
                for condition in ("C0_raw", "C1_exact_mft_subtraction", "C2_energy_matched_non_mft_control")
            ]
            stacked = torch.from_numpy(np.concatenate(model_inputs, axis=0)).to(device)
            logits, _ = model(stacked)
            probabilities = torch.sigmoid(logits).float().cpu().numpy().reshape(3, len(indices), labels.shape[1])
            parts["identity"].append(batch_identities)
            parts["labels"].append(batch_labels)
            parts["regime"].append(np.full(len(indices), regime_index, dtype=np.int8))
            parts["power"].append(np.full(len(indices), power, dtype=np.int16))
            for position, condition in enumerate(("C0_raw", "C1_exact_mft_subtraction", "C2_energy_matched_non_mft_control")):
                parts[condition].append(probabilities[position])
    return {name: np.concatenate(values) for name, values in parts.items()}


def collect_data_seed(
    model: torch.nn.Module,
    paths: dict[str, dict[int, Path]],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    all_parts: dict[str, list[np.ndarray]] = {}
    for regime_index, regime in enumerate(REGIMES):
        for power in POWERS:
            current = _predict_conditions_for_view(model, paths[regime][power], device, batch_size, regime_index, power)
            for name, value in current.items():
                all_parts.setdefault(name, []).append(value)
    merged = {name: np.concatenate(values) for name, values in all_parts.items()}
    if not (np.array_equal(merged["labels"], merged["labels"].astype(np.uint8)) and np.isfinite(merged["C1_exact_mft_subtraction"]).all()):
        raise RuntimeError("P3o-S collection produced invalid values")
    return merged


def role_paths(root: Path, seed: int, suffix: str) -> dict[str, dict[int, Path]]:
    prefix = f"p3os_mft_oracle_subtraction_seed{seed}"
    return {regime: transition_files(root / f"{prefix}_{regime}{suffix}", "train", POWERS) for regime in REGIMES}


def audit_data_seed(paths: dict[str, dict[int, Path]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    reference_identity = reference_labels = None
    rows = []
    for regime, files in paths.items():
        alignment = audit_power_alignment(files)
        identity, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            jnr = np.asarray(handle["parameter_values"][MFT_JNR_COLUMN], dtype=np.float64)
            jnr_mask = np.asarray(handle["parameter_mask"][MFT_JNR_COLUMN], dtype=np.uint8).astype(bool)
            noise_power = _read_noise_power(handle)
        mft = labels[:, MFT_CLASS_INDEX] > 0
        if len(identity) != 2079 or not bool(mft.any()) or not bool(np.all(jnr_mask[mft] & np.isfinite(jnr[mft]))) or bool(np.any(jnr_mask[~mft])):
            raise RuntimeError(f"P3o-S malformed MFT JNR contract: {regime}")
        if reference_identity is None:
            reference_identity, reference_labels = identity, labels
        elif not np.array_equal(reference_identity, identity) or not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"P3o-S identities or labels differ across {regime}")
        rows.append({"regime": regime, "identity_count": int(len(identity)), "mft_positive": int(mft.sum()), "noise_power": noise_power, "alignment": alignment})
    assert reference_identity is not None and reference_labels is not None
    return reference_identity, reference_labels, rows


def audit_prior_identities(
    project: Path,
    first_identity: np.ndarray,
    second_identity: np.ndarray,
    current_paths: set[Path],
    extra_roots: list[Path],
) -> dict[str, object]:
    """Check both new diagnostic roles against every discoverable prior MAT file."""
    roots = [project / "generated_data", Path("D:/CJ0724_generated_data"), Path("D:/CJ0724_rechunked_data")] + extra_roots
    previous: set[Path] = set()
    for root in roots:
        if root.is_dir():
            previous.update(path.resolve() for path in root.rglob("closedset_6class_train.mat"))
    protocol = json.loads((project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156" / "protocol.json").read_text(encoding="utf-8"))
    original = Path(protocol["data_file"])
    if original.is_file():
        previous.add(original.resolve())
    prior_rows, maximum = [], 0
    for path in sorted(previous):
        if path in current_paths:
            continue
        old_identity, _ = read_identity_metadata(path)
        first_overlap = int(np.intersect1d(first_identity, old_identity).size)
        second_overlap = int(np.intersect1d(second_identity, old_identity).size)
        maximum = max(maximum, first_overlap, second_overlap)
        prior_rows.append({"path": str(path), "first_overlap": first_overlap, "second_overlap": second_overlap})
    if maximum:
        raise RuntimeError("P3o-S identity overlap with a prior dataset")
    return {"prior_files_checked": len(prior_rows), "maximum_prior_overlap": maximum, "rows": prior_rows}


def _difference(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    return {
        "source_fpr_reduction": float(baseline["source_isrj_fpr"] - candidate["source_isrj_fpr"]),
        "worst_source_cell_fpr_reduction": float(baseline["worst_source_cell_fpr"] - candidate["worst_source_cell_fpr"]),
        "isrj_recall_delta": float(candidate["isrj_recall"] - baseline["isrj_recall"]),
        "macro_aupr_delta": float(candidate["macro_aupr"] - baseline["macro_aupr"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", required=True)
    parser.add_argument("--rechunk-suffix", default="_rechunked_v2")
    parser.add_argument("--data-seeds", nargs=2, type=int, default=(1920000000, 1930000000))
    parser.add_argument("--preregister", default="configs/p3os_mft_oracle_subtraction_preregister_20260803.json")
    parser.add_argument("--method-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--prior-identity-root", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    preregister, method_freeze = project / args.preregister, project / args.method_freeze
    freeze = json.loads(method_freeze.read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(preregister) or freeze["implementation_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("P3o-S preregister or implementation hash mismatch")
    output = project / args.output_root
    output.mkdir(parents=True, exist_ok=False)
    root = Path(args.rechunk_root)
    data = {seed: role_paths(root, seed, args.rechunk_suffix) for seed in args.data_seeds}
    audits = {}
    identities = {}
    labels = {}
    for seed, paths in data.items():
        identity, label, rows = audit_data_seed(paths)
        identities[seed], labels[seed] = identity, label
        audits[str(seed)] = {"identity_sha256": hashlib.sha256(np.ascontiguousarray(identity).tobytes()).hexdigest(), "contract": rows}
    if np.intersect1d(identities[args.data_seeds[0]], identities[args.data_seeds[1]]).size:
        raise RuntimeError("P3o-S two diagnostic identity sets overlap")
    current_paths = {path.resolve() for paths in data.values() for by_power in paths.values() for path in by_power.values()}
    prior_audit = audit_prior_identities(project, identities[args.data_seeds[0]], identities[args.data_seeds[1]], current_paths, [Path(value) for value in args.prior_identity_root])
    run_root, p2c_root = project / args.run_root, project / args.p2c_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    rows = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    p2c = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8"))
    selected = {int(row["seed"]): row for row in rows if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}
    seed_freezes = {int(row["seed"]): row for row in p2c["seed_freezes"]}
    if set(MODEL_SEEDS) != set(selected) or set(MODEL_SEEDS) != set(seed_freezes):
        raise RuntimeError("P3o-S frozen model or threshold receipts are incomplete")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    runs = []
    saved_predictions: dict[str, np.ndarray] = {}
    for model_seed in MODEL_SEEDS:
        model, checkpoint = _load_main(selected[model_seed], protocol, project, device)
        thresholds = np.asarray(seed_freezes[model_seed]["safe_thresholds"], dtype=np.float64)
        for data_seed in args.data_seeds:
            result = collect_data_seed(model, data[data_seed], device, args.batch_size)
            metrics = {condition: _metrics(result[condition], result["labels"], thresholds, result["regime"], result["power"]) for condition in ("C0_raw", "C1_exact_mft_subtraction", "C2_energy_matched_non_mft_control")}
            bootstrap_c0_c1 = paired_bootstrap_source_fpr_reduction(result["C0_raw"][:, ISRJ_CLASS_INDEX], result["C1_exact_mft_subtraction"][:, ISRJ_CLASS_INDEX], result["labels"], result["identity"], thresholds[ISRJ_CLASS_INDEX], 700000 + model_seed + data_seed, args.bootstrap_resamples)
            bootstrap_c2_c1 = paired_bootstrap_source_fpr_reduction(result["C2_energy_matched_non_mft_control"][:, ISRJ_CLASS_INDEX], result["C1_exact_mft_subtraction"][:, ISRJ_CLASS_INDEX], result["labels"], result["identity"], thresholds[ISRJ_CLASS_INDEX], 800000 + model_seed + data_seed, args.bootstrap_resamples)
            run = {"model_seed": model_seed, "data_seed": data_seed, "checkpoint": str(checkpoint), "safe_thresholds": thresholds, "metrics": metrics, "C1_minus_C0": _difference(metrics["C1_exact_mft_subtraction"], metrics["C0_raw"]), "C1_minus_C2": _difference(metrics["C1_exact_mft_subtraction"], metrics["C2_energy_matched_non_mft_control"]), "bootstrap_C0_minus_C1": bootstrap_c0_c1, "bootstrap_C2_minus_C1": bootstrap_c2_c1}
            runs.append(run)
            key = f"model{model_seed}_data{data_seed}"
            saved_predictions |= {f"{key}_{name}": result[name] for name in ("identity", "labels", "regime", "power", "C0_raw", "C1_exact_mft_subtraction", "C2_energy_matched_non_mft_control")}
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    gates = {
        "C1_source_fpr_reduction_vs_C0_at_least_0p02_each_replication": all(run["C1_minus_C0"]["source_fpr_reduction"] >= 0.02 for run in runs),
        "C1_source_fpr_reduction_vs_C2_at_least_0p01_each_replication": all(run["C1_minus_C2"]["source_fpr_reduction"] >= 0.01 for run in runs),
        "C1_isrj_recall_drop_vs_C0_at_most_0p02_each_replication": all(run["C1_minus_C0"]["isrj_recall_delta"] >= -0.02 for run in runs),
        "C1_macro_AUPR_delta_vs_C0_at_least_minus_0p005_each_replication": all(run["C1_minus_C0"]["macro_aupr_delta"] >= -0.005 for run in runs),
        "C0_minus_C1_bootstrap_ci_low_positive_each_replication": all(run["bootstrap_C0_minus_C1"]["ci95_low"] > 0.0 for run in runs),
        "C2_minus_C1_bootstrap_ci_low_positive_each_replication": all(run["bootstrap_C2_minus_C1"]["ci95_low"] > 0.0 for run in runs),
        "identity_overlap_zero": True,
    }
    summary = {"experiment": "P3o-S oracle MFT subtraction causal diagnosis", "development_passed": bool(all(gates.values())), "promotion_gates": gates, "runs": runs, "data_audit": audits | {"prior_identity_audit": prior_audit}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **saved_predictions)
    print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
