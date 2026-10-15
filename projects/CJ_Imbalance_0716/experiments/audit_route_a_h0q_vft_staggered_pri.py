"""Provenance audit for the one-shot paired Route-A H0q staggered-PRI VFT source."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from cjlt.models.vft_staggered_pri import record_samples
from experiments.find_route_a_h0o_crosspulse_identity_range import historic_receiver_ids


DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
COUNT, BASE_SEED = 640, 2_600_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    fields = ("pair_seed", "receiver_noise_seed", "doppler_Hz", "velocity_mps", "effective_JNR_dB", "arrival_delay_samples", "doppler_quantile", "pulse_start_samples")
    with h5py.File(path, "r") as handle:
        if not set(fields).issubset(handle.keys()) or not {"iq_real", "iq_imag"}.issubset(handle.keys()):
            raise ValueError(f"missing H0q fields in {path}")
        metadata = {name: np.asarray(handle[name]) for name in fields}
        iq = np.asarray(handle["iq_real"]) + 1j * np.asarray(handle["iq_imag"])
    return metadata, iq


def audit(args: argparse.Namespace) -> dict[str, object]:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    selection = json.loads(args.identity_selection.read_text(encoding="utf-8"))
    if not receipt.get("executed") or selection.get("identity_overlap_count") != 0:
        raise ValueError("H0q receipt or pre-generation identity selection invalid")
    if any(seal["test_v3"][name] for name in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    low, low_iq = _load(args.low_path); high, high_iq = _load(args.high_path)
    receiver = low["receiver_noise_seed"].reshape(-1).astype(np.uint64)
    historic, sources = historic_receiver_ids()
    common = ("pair_seed", "receiver_noise_seed", "effective_JNR_dB", "arrival_delay_samples", "doppler_quantile", "pulse_start_samples")
    checks = {
        "payload_shape_exact": low_iq.shape == high_iq.shape == (COUNT, record_samples()),
        "pair_seed_exact": np.array_equal(low["pair_seed"].reshape(-1), np.arange(BASE_SEED, BASE_SEED + COUNT)),
        "receiver_identity_unique": np.unique(receiver).size == COUNT,
        "all_common_counterfactual_metadata_exact": all(np.array_equal(low[name], high[name]) for name in common),
        "low_doppler_in_preregistered_band": bool(np.all((low["doppler_Hz"] >= 1000.0) & (low["doppler_Hz"] <= 2500.0))),
        "high_doppler_in_preregistered_band": bool(np.all((high["doppler_Hz"] >= 5500.0) & (high["doppler_Hz"] <= 7000.0))),
        "velocity_is_doppler_derivative": bool(np.allclose(low["velocity_mps"], low["doppler_Hz"] * 3e8 / (2e10)) and np.allclose(high["velocity_mps"], high["doppler_Hz"] * 3e8 / (2e10))),
        "declared_doppler_intervention_changes_iq": not np.array_equal(low_iq, high_iq),
        "historical_train_identity_overlap_zero": not bool(historic.intersection(map(int, receiver))),
        "test_v3_still_sealed": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"H0q provenance contract failed: {checks}")
    result = {
        "status": "completed_train_only_paired_audit",
        "experiment": "RouteA_H0q_VFTStaggeredPRIPhaseConsistency",
        "low_sha256": _sha256(args.low_path), "high_sha256": _sha256(args.high_path),
        "sample_count_per_arm": COUNT, "record_samples": record_samples(), "contracts": checks,
        "pre_generation_identity_selection": str(args.identity_selection),
        "historic_train_sources_scanned": sources, "identity_audit_passed": True,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    source = DATA_ROOT / "h0q_vft_staggered_pri_20260803"
    root = ROOT / "outputs/tapc/route_a_h0q_vft_staggered_pri_generation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-path", type=Path, default=source / "vft_low_1p0_2p5kHz.mat")
    parser.add_argument("--high-path", type=Path, default=source / "vft_high_5p5_7p0kHz.mat")
    parser.add_argument("--receipt", type=Path, default=root / "formal_generation_receipt.json")
    parser.add_argument("--identity-selection", type=Path, default=root / "pre_generation_identity_selection.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=root / "paired_identity_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    value = audit(parse_args())
    print(json.dumps({"status": value["status"], "identity_audit_passed": value["identity_audit_passed"]}, ensure_ascii=False))
