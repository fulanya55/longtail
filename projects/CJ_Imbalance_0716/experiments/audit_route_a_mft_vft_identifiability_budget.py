"""Read the official *train* metadata and reproduce the MFT/VFT information budget."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name]).reshape(-1)[0])


def audit(data_path: Path, output: Path, *, hash_source: bool = False) -> dict:
    if "test" in data_path.name.lower():
        raise ValueError("identifiability budget accepts a train-only metadata source")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    with h5py.File(data_path, "r") as handle:
        fs_hz = _scalar(handle, "config/radar/fs_Hz")
        pulse_s = _scalar(handle, "config/radar/pulse_width_s")
        pri_s = _scalar(handle, "config/radar/pri_s")
        num_pulses = int(round(_scalar(handle, "config/radar/num_pulses")))
        vft_hz = np.asarray(handle["config/params/VFT/doppler_Hz"]).reshape(-1).astype(float)
        mft_targets = np.asarray(handle["config/params/MFT/num_targets"]).reshape(-1).astype(int)
        mft_delay_s = np.asarray(handle["config/params/MFT/target_delay_s"]).reshape(-1).astype(float)

    pulse_resolution_hz = 1.0 / pulse_s
    vft_cycles = vft_hz * pulse_s
    delay_span_s = float(mft_delay_s.max() - mft_delay_s.min())
    overlap_lower_bound_s = max(0.0, pulse_s - delay_span_s)
    result = {
        "status": "completed_train_metadata_only",
        "experiment": "RouteA_MFTVFTIdentifiabilityBudget_20260803",
        "evidence_role": "official_train_metadata_physical_budget_audit",
        "source": {
            "path": str(data_path),
            "bytes": data_path.stat().st_size,
            "sha256": _sha256(data_path) if hash_source else None,
            "sha256_computed": bool(hash_source),
            "role": "official_train_only_metadata",
        },
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "vft": {
            "sampling_rate_hz": fs_hz,
            "active_pulse_width_s": pulse_s,
            "num_pulses_per_record": num_pulses,
            "pri_s": pri_s,
            "single_pulse_fourier_scale_hz": pulse_resolution_hz,
            "doppler_range_hz": [float(vft_hz.min()), float(vft_hz.max())],
            "doppler_span_over_fourier_scale": float((vft_hz.max() - vft_hz.min()) / pulse_resolution_hz),
            "cycles_over_active_pulse": [float(vft_cycles.min()), float(vft_cycles.max())],
            "phase_excursion_rad": [float(2 * np.pi * vft_cycles.min()), float(2 * np.pi * vft_cycles.max())],
        },
        "mft": {
            "num_targets_range": [int(mft_targets.min()), int(mft_targets.max())],
            "relative_delay_range_s": [float(mft_delay_s.min()), float(mft_delay_s.max())],
            "relative_delay_span_s": delay_span_s,
            "pulse_width_s": pulse_s,
            "overlap_lower_bound_s": overlap_lower_bound_s,
            "overlap_lower_bound_fraction": float(overlap_lower_bound_s / pulse_s),
            "equal_energy_per_copy_penalty_db": [float(10 * np.log10(mft_targets.min())), float(10 * np.log10(mft_targets.max()))],
            "penalty_interpretation": "Equal-energy, noncoherent approximation only; actual coherent random-amplitude copies can differ.",
        },
        "contract_checks": {
            "single_pulse": num_pulses == 1,
            "vft_span_below_single_pulse_fourier_scale": bool((vft_hz.max() - vft_hz.min()) < pulse_resolution_hz),
            "mft_delay_span_below_pulse_width": delay_span_s < pulse_s,
            "mft_overlap_at_least_0p80": overlap_lower_bound_s / pulse_s >= 0.80 - 1e-12,
        },
    }
    if not all(result["contract_checks"].values()):
        raise ValueError("unexpected 0716 metadata contract; do not reuse this interpretation")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_vft_identifiability_budget/metadata_budget.json")
    parser.add_argument("--hash-source", action="store_true", help="Hash the complete train file; disabled by default because metadata audit does not need payload hashing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = audit(args.data_path, args.output, hash_source=args.hash_source)
    print(json.dumps({"status": value["status"], "output": str(args.output), "checks": value["contract_checks"]}, ensure_ascii=False))
