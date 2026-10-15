"""Evaluate a range-agnostic signed VFT delay-Doppler grid on fresh OOD identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.vft_doppler_aligned_expert import CHIRP_RATE_HZ_PER_S, VFT_CLASS_INDEX, VFT_DOPPLER_COLUMN, estimate_aligned_doppler


POWERS = (25, 35, 50)
REGIMES = ("ID_control", "near_low", "near_high", "far_negative", "far_negative_high")
CONSTANT_HZ = 4000.0
VARIANTS = {
    "C1_frozen_grid": {"doppler_min_hz": 1000.0, "doppler_max_hz": 7000.0, "doppler_step_hz": 100.0},
    "C2_wide_signed_grid": {"doppler_min_hz": -30000.0, "doppler_max_hz": 10000.0, "doppler_step_hz": 250.0},
    "C3_wrong_reference": {"doppler_min_hz": -30000.0, "doppler_max_hz": 10000.0, "doppler_step_hz": 250.0, "chirp_rate_hz_per_s": 0.8 * CHIRP_RATE_HZ_PER_S},
    "C4_no_delay_alignment": {"doppler_min_hz": -30000.0, "doppler_max_hz": 10000.0, "doppler_step_hz": 250.0, "search_delay": False},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mae = float(np.mean(np.abs(prediction - target)))
    constant_mae = float(np.mean(np.abs(CONSTANT_HZ - target)))
    pearson = 0.0
    if len(target) >= 2 and np.std(prediction) > 0 and np.std(target) > 0:
        pearson = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "count": int(len(target)),
        "mae_hz": mae,
        "constant_mae_hz": constant_mae,
        "mae_reduction_vs_constant": float(1.0 - mae / max(constant_mae, 1.0e-12)),
        "pearson": pearson,
    }


@torch.inference_mode()
def predict(path: Path, indices: np.ndarray, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    dataset = V73IQDataset(path, indices, [VFT_DOPPLER_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    outputs = {name: [] for name in VARIANTS}
    sample_indices = []
    try:
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            sample_indices.append(batch["sample_index"].numpy())
            for name, kwargs in VARIANTS.items():
                outputs[name].append(estimate_aligned_doppler(iq, **kwargs)["prediction_hz"].cpu().numpy())
    finally:
        dataset.close()
    return {"sample_index": np.concatenate(sample_indices), **{name: np.concatenate(parts) for name, parts in outputs.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p2u_vft_doppler_ood_v2_seed3300000000_{regime}_rechunked")
    parser.add_argument("--preregister", default="configs/p2u_vft_wide_grid_ood_preregister_20260801.json")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--output-root", default="outputs/p2u_vft_wide_grid_ood_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    preregister = project / args.preregister
    training_protocol = project / args.training_protocol
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    excluded_paths = [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    protocol_payload = json.loads(training_protocol.read_text(encoding="utf-8"))
    excluded_paths.insert(0, Path(protocol_payload["data_file"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identity_reference = None
    records = []
    for regime in REGIMES:
        root = project / args.data_root_template.format(regime=regime)
        files = transition_files(root, "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        selected = np.flatnonzero(labels[:, VFT_CLASS_INDEX] > 0)
        if len(identities) != 630 or len(selected) != 320:
            raise RuntimeError(f"P2u count mismatch for {regime}")
        if identity_reference is None:
            identity_reference = identities
        elif not np.array_equal(identity_reference, identities):
            raise RuntimeError("P2u regimes are not identity paired")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2u identity overlap")
        regime_arrays = {"identity": [], "power": [], "combination_size": [], "target": []}
        for name in VARIANTS:
            regime_arrays[name] = []
        for power in POWERS:
            path = files[power]
            with __import__("h5py").File(path, "r") as handle:
                target_all = np.asarray(handle["parameter_values"][VFT_DOPPLER_COLUMN, :], dtype=np.float64)
            result = predict(path, selected, device, args.batch_size)
            if not np.array_equal(result["sample_index"], selected):
                raise RuntimeError("P2u prediction order mismatch")
            regime_arrays["identity"].append(identities[selected])
            regime_arrays["power"].append(np.full(len(selected), power, dtype=np.int16))
            regime_arrays["combination_size"].append(labels[selected].sum(axis=1).astype(np.int8))
            regime_arrays["target"].append(target_all[selected])
            for name in VARIANTS:
                regime_arrays[name].append(result[name])
        arrays = {name: np.concatenate(parts) for name, parts in regime_arrays.items()}
        run_dir = output_root / regime
        run_dir.mkdir()
        np.savez_compressed(run_dir / "predictions.npz", **arrays)
        variants = {}
        for name in VARIANTS:
            variants[name] = {
                "overall": regression_metrics(arrays[name], arrays["target"]),
                "compound": regression_metrics(arrays[name][arrays["combination_size"] >= 2], arrays["target"][arrays["combination_size"] >= 2]),
                "by_power": {str(power): regression_metrics(arrays[name][arrays["power"] == power], arrays["target"][arrays["power"] == power]) for power in POWERS},
            }
        records.append({"regime": regime, "alignment": alignment, "exclusions": exclusions, "variants": variants})
        print(json.dumps({"regime": regime, "candidate": variants["C2_wide_signed_grid"]["overall"], "frozen": variants["C1_frozen_grid"]["overall"]}), flush=True)

    by_regime = {row["regime"]: row for row in records}
    candidate = {regime: by_regime[regime]["variants"]["C2_wide_signed_grid"] for regime in REGIMES}
    frozen = {regime: by_regime[regime]["variants"]["C1_frozen_grid"] for regime in REGIMES}
    controls = ("C3_wrong_reference", "C4_no_delay_alignment")
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for row in records for ex in row["exclusions"])),
        "candidate_each_regime_reduction_vs_constant_at_least_0p10": bool(all(candidate[r]["overall"]["mae_reduction_vs_constant"] >= 0.10 for r in REGIMES)),
        "candidate_each_regime_pearson_at_least_0p30": bool(all(candidate[r]["overall"]["pearson"] >= 0.30 for r in REGIMES)),
        "candidate_each_regime_each_power_positive_direction": bool(all(
            candidate[r]["by_power"][str(p)]["mae_reduction_vs_constant"] >= 0.10
            and candidate[r]["by_power"][str(p)]["pearson"] >= 0.30
            for r in REGIMES for p in POWERS
        )),
        "candidate_ID_mae_ratio_vs_frozen_at_most_1p05": bool(candidate["ID_control"]["overall"]["mae_hz"] <= 1.05 * frozen["ID_control"]["overall"]["mae_hz"]),
        "candidate_ID_pearson_delta_vs_frozen_at_least_minus_0p02": bool(candidate["ID_control"]["overall"]["pearson"] >= frozen["ID_control"]["overall"]["pearson"] - 0.02),
        "candidate_each_OOD_mean_mae_reduction_vs_frozen_at_least_0p20": bool(all(
            1.0 - candidate[r]["overall"]["mae_hz"] / frozen[r]["overall"]["mae_hz"] >= 0.20
            for r in REGIMES if r != "ID_control"
        )),
        "candidate_compound_each_regime_reduction_vs_constant_at_least_0p10": bool(all(candidate[r]["compound"]["mae_reduction_vs_constant"] >= 0.10 for r in REGIMES)),
        "candidate_compound_each_regime_pearson_at_least_0p30": bool(all(candidate[r]["compound"]["pearson"] >= 0.30 for r in REGIMES)),
        "candidate_beats_controls_mean_mae": bool(all(
            np.mean([candidate[r]["overall"]["mae_hz"] for r in REGIMES])
            < np.mean([by_regime[r]["variants"][control]["overall"]["mae_hz"] for r in REGIMES])
            for control in controls
        )),
    }
    passed = bool(all(gates.values()))
    summary_path = output_root / "summary.json"
    summary = {
        "experiment": "P2u range-agnostic VFT Doppler physical expert development",
        "preregister_sha256": sha256_file(preregister),
        "identity_audit": {"identity_count": int(len(identity_reference)), "identity_sha256": _hash_int64(identity_reference), "paired_across_regimes": True},
        "promotion_gates": gates,
        "development_passed": passed,
        "per_regime": {row["regime"]: row["variants"] for row in records},
        "records": records,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if passed:
        (output_root / "method_freeze.json").write_text(json.dumps({
            "method": "reference-aligned wide signed VFT Doppler grid",
            "doppler_min_hz": -30000.0,
            "doppler_max_hz": 10000.0,
            "doppler_step_hz": 250.0,
            "delay_search": [100, 500, 8],
            "summary_sha256": sha256_file(summary_path),
            "preregister_sha256": sha256_file(preregister),
            "locked_test_used": False,
        }, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates}, indent=2))


if __name__ == "__main__":
    main()
