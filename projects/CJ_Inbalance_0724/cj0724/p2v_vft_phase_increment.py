"""P2v fixed-lag phase-increment VFT Doppler physical-sufficiency screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import SIGNAL_LENGTH, V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2u_vft_wide_grid_ood import CONSTANT_HZ, POWERS, REGIMES, regression_metrics
from cj0724.vft_doppler_aligned_expert import (
    CHIRP_RATE_HZ_PER_S,
    FS_HZ,
    PULSE_SAMPLES,
    VFT_CLASS_INDEX,
    VFT_DOPPLER_COLUMN,
    estimate_aligned_doppler,
    lfm_reference,
)


PHASE_LAG = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def estimate_phase_increment_doppler(
    iq: torch.Tensor,
    chirp_rate_hz_per_s: float = CHIRP_RATE_HZ_PER_S,
    delay_min_samples: int = 100,
    delay_max_samples: int = 500,
    delay_candidates: int = 32,
    phase_lag_samples: int = PHASE_LAG,
    search_delay: bool = True,
) -> dict[str, torch.Tensor]:
    """Estimate signed Doppler from a coherent dechirped phase increment."""
    if iq.ndim != 3 or tuple(iq.shape[1:]) != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [B,2,{SIGNAL_LENGTH}]")
    if not 0 < phase_lag_samples < PULSE_SAMPLES:
        raise ValueError("phase lag must be within the reference pulse")
    if not 0 <= delay_min_samples <= delay_max_samples < SIGNAL_LENGTH:
        raise ValueError("invalid delay search interval")

    signal = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    signal = signal / signal.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    reference = lfm_reference(signal.device, chirp_rate_hz_per_s)
    pulse = reference[:PULSE_SAMPLES]
    pulse_index = torch.arange(PULSE_SAMPLES, device=signal.device)

    if search_delay:
        correlation = torch.fft.ifft(torch.fft.fft(signal) * torch.fft.fft(reference).conj()).abs()
        restricted = correlation[:, delay_min_samples:delay_max_samples + 1]
        candidate_count = min(int(delay_candidates), int(restricted.shape[1]))
        top = torch.topk(restricted, candidate_count, dim=1)
        candidate_delay = top.indices + delay_min_samples
    else:
        candidate_delay = torch.zeros((len(signal), 1), dtype=torch.long, device=signal.device)

    gather_index = (candidate_delay[:, :, None] + pulse_index[None, None, :]) % SIGNAL_LENGTH
    expanded = signal[:, None, :].expand(-1, candidate_delay.shape[1], -1)
    dechirped = torch.gather(expanded, 2, gather_index) * pulse.conj()[None, None, :]
    lagged_product = (
        dechirped[:, :, phase_lag_samples:]
        * dechirped[:, :, :-phase_lag_samples].conj()
    )
    phasor = lagged_product.sum(dim=-1)
    coherence = phasor.abs() / lagged_product.abs().sum(dim=-1).clamp_min(1.0e-8)
    pulse_energy = dechirped.abs().sum(dim=-1)
    normalized_energy = pulse_energy / pulse_energy.max(dim=1, keepdim=True).values.clamp_min(1.0e-8)
    selection_score = coherence * normalized_energy
    delay_slot = selection_score.argmax(dim=1)
    batch_index = torch.arange(len(signal), device=signal.device)
    selected_phasor = phasor[batch_index, delay_slot]
    prediction = torch.angle(selected_phasor) * (
        FS_HZ / (2.0 * torch.pi * float(phase_lag_samples))
    )
    return {
        "prediction_hz": prediction.float(),
        "coherence": coherence[batch_index, delay_slot].float(),
        "selection_score": selection_score[batch_index, delay_slot].float(),
        "delay_samples": candidate_delay[batch_index, delay_slot].float(),
    }


@torch.inference_mode()
def predict(path: Path, indices: np.ndarray, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    dataset = V73IQDataset(path, indices, [VFT_DOPPLER_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    output = {
        name: [] for name in (
            "sample_index", "C1_frozen_grid", "C2_phase_increment", "C3_wrong_reference",
            "C4_no_delay_alignment", "C2_coherence", "C2_score", "C2_delay",
        )
    }
    try:
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            c1 = estimate_aligned_doppler(
                iq, doppler_min_hz=1000.0, doppler_max_hz=7000.0, doppler_step_hz=100.0,
            )
            c2 = estimate_phase_increment_doppler(iq)
            c3 = estimate_phase_increment_doppler(iq, chirp_rate_hz_per_s=0.8 * CHIRP_RATE_HZ_PER_S)
            c4 = estimate_phase_increment_doppler(iq, search_delay=False)
            output["sample_index"].append(batch["sample_index"].numpy())
            output["C1_frozen_grid"].append(c1["prediction_hz"].cpu().numpy())
            output["C2_phase_increment"].append(c2["prediction_hz"].cpu().numpy())
            output["C3_wrong_reference"].append(c3["prediction_hz"].cpu().numpy())
            output["C4_no_delay_alignment"].append(c4["prediction_hz"].cpu().numpy())
            output["C2_coherence"].append(c2["coherence"].cpu().numpy())
            output["C2_score"].append(c2["selection_score"].cpu().numpy())
            output["C2_delay"].append(c2["delay_samples"].cpu().numpy())
    finally:
        dataset.close()
    return {name: np.concatenate(parts) for name, parts in output.items()}


def summarize_variant(prediction: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    target = arrays["target"]
    compound = arrays["combination_size"] >= 2
    return {
        "overall": regression_metrics(prediction, target),
        "compound": regression_metrics(prediction[compound], target[compound]),
        "by_power": {
            str(power): regression_metrics(prediction[arrays["power"] == power], target[arrays["power"] == power])
            for power in POWERS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2v_vft_phase_increment_seed3600000000_{regime}_rechunked")
    parser.add_argument("--preregister", default="configs/p2v_vft_phase_increment_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2v_vft_phase_increment_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    preregister = project / args.preregister
    excluded_paths = [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    identity_reference = None
    records = []
    variant_names = ("C1_frozen_grid", "C2_phase_increment", "C3_wrong_reference", "C4_no_delay_alignment")
    for regime in REGIMES:
        root = project / args.data_root_template.format(regime=regime)
        files = transition_files(root, "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        selected = np.flatnonzero(labels[:, VFT_CLASS_INDEX] > 0)
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2v count mismatch for {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P2v regimes are not identity paired")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2v identity overlap")

        parts = {name: [] for name in ("identity", "power", "combination_size", "target", *variant_names, "C2_coherence", "C2_score", "C2_delay")}
        for power in POWERS:
            path = files[power]
            with h5py.File(path, "r") as handle:
                target_all = np.asarray(handle["parameter_values"][VFT_DOPPLER_COLUMN, :], dtype=np.float64)
            result = predict(path, selected, device, args.batch_size)
            if not np.array_equal(result["sample_index"], selected):
                raise RuntimeError("P2v prediction order mismatch")
            parts["identity"].append(identities[selected])
            parts["power"].append(np.full(len(selected), power, dtype=np.int16))
            parts["combination_size"].append(labels[selected].sum(axis=1).astype(np.int8))
            parts["target"].append(target_all[selected])
            for name in variant_names:
                parts[name].append(result[name])
            for name in ("C2_coherence", "C2_score", "C2_delay"):
                parts[name].append(result[name])
        arrays = {name: np.concatenate(values) for name, values in parts.items()}
        run_dir = output_root / regime
        run_dir.mkdir()
        np.savez_compressed(run_dir / "predictions.npz", **arrays)
        variants = {name: summarize_variant(arrays[name], arrays) for name in variant_names}
        records.append({"regime": regime, "alignment": alignment, "exclusions": exclusions, "variants": variants})
        print(json.dumps({
            "regime": regime,
            "phase": variants["C2_phase_increment"]["overall"],
            "grid": variants["C1_frozen_grid"]["overall"],
            "mean_coherence": float(np.mean(arrays["C2_coherence"])),
        }), flush=True)

    by_regime = {row["regime"]: row for row in records}
    candidate = {regime: by_regime[regime]["variants"]["C2_phase_increment"] for regime in REGIMES}
    frozen = {regime: by_regime[regime]["variants"]["C1_frozen_grid"] for regime in REGIMES}
    controls = ("C3_wrong_reference", "C4_no_delay_alignment")
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for row in records for ex in row["exclusions"])),
        "candidate_each_regime_reduction_vs_constant_at_least_0p10": bool(all(candidate[r]["overall"]["mae_reduction_vs_constant"] >= 0.10 for r in REGIMES)),
        "candidate_each_regime_pearson_at_least_0p30": bool(all(candidate[r]["overall"]["pearson"] >= 0.30 for r in REGIMES)),
        "candidate_each_regime_each_power_positive_direction": bool(all(
            candidate[r]["by_power"][str(power)]["mae_reduction_vs_constant"] >= 0.10
            and candidate[r]["by_power"][str(power)]["pearson"] >= 0.30
            for r in REGIMES for power in POWERS
        )),
        "candidate_ID_mae_ratio_vs_frozen_at_most_1p05": bool(candidate["ID_control"]["overall"]["mae_hz"] <= 1.05 * frozen["ID_control"]["overall"]["mae_hz"]),
        "candidate_ID_pearson_delta_vs_frozen_at_least_minus_0p02": bool(candidate["ID_control"]["overall"]["pearson"] >= frozen["ID_control"]["overall"]["pearson"] - 0.02),
        "candidate_each_OOD_mae_reduction_vs_frozen_at_least_0p20": bool(all(
            1.0 - candidate[r]["overall"]["mae_hz"] / frozen[r]["overall"]["mae_hz"] >= 0.20
            for r in REGIMES if r != "ID_control"
        )),
        "candidate_compound_each_regime_reduction_vs_constant_at_least_0p10": bool(all(candidate[r]["compound"]["mae_reduction_vs_constant"] >= 0.10 for r in REGIMES)),
        "candidate_compound_each_regime_pearson_at_least_0p30": bool(all(candidate[r]["compound"]["pearson"] >= 0.30 for r in REGIMES)),
        "candidate_mean_mae_at_most_0p8_of_controls": bool(all(
            np.mean([candidate[r]["overall"]["mae_hz"] for r in REGIMES])
            <= 0.8 * np.mean([by_regime[r]["variants"][control]["overall"]["mae_hz"] for r in REGIMES])
            for control in controls
        )),
        "candidate_prediction_finite_rate_is_one": bool(all(
            np.isfinite(np.load(output_root / r / "predictions.npz")["C2_phase_increment"]).all() for r in REGIMES
        )),
    }
    passed = bool(all(gates.values()))
    summary_path = output_root / "summary.json"
    summary = {
        "experiment": "P2v reference-aligned phase-increment VFT Doppler physical sufficiency",
        "preregister_sha256": sha256_file(preregister),
        "implementation_sha256": sha256_file(Path(__file__)),
        "identity_audit": {"identity_count": int(len(identity_reference)), "identity_sha256": _hash_int64(identity_reference), "paired_across_regimes": True},
        "algorithm": {"phase_lag_samples": PHASE_LAG, "unambiguous_frequency_hz": [-FS_HZ / (2 * PHASE_LAG), FS_HZ / (2 * PHASE_LAG)]},
        "promotion_gates": gates,
        "development_passed": passed,
        "per_regime": {row["regime"]: row["variants"] for row in records},
        "records": records,
        "interpretation": (
            "P2v passed physical sufficiency; run two fresh no-tuning development validations before confirmation."
            if passed else
            "P2v failed physical sufficiency; do not deepen or attach the VFT expert."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if passed:
        (output_root / "method_freeze.json").write_text(json.dumps({
            "method": "P2v reference-aligned phase-increment VFT Doppler",
            "phase_lag_samples": PHASE_LAG,
            "summary_sha256": sha256_file(summary_path),
            "preregister_sha256": sha256_file(preregister),
            "locked_test_used": False,
        }, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates}, indent=2))


if __name__ == "__main__":
    main()
