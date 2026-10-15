"""Provenance and intervention audit for Route-A H0n; never reads sealed data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0n_mft_crosspulse_pairs_20260803_r2"
N_PAIRS = 640


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receiver_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def audit(args: argparse.Namespace) -> dict[str, object]:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if not receipt.get("executed") or any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("generation receipt invalid or test-v3 seal no longer intact")
    with h5py.File(args.coherent_path, "r") as coherent, h5py.File(args.scrambled_path, "r") as scrambled:
        required = {"iq_real", "iq_imag", "pair_seed", "receiver_noise_seed", "mft_total_jnr_db", "arrival_delay_samples", "delay_pool_samples", "amplitude_pool", "scramble_phase_rad"}
        common_keys = required.issubset(coherent.keys()) and required.issubset(scrambled.keys())
        shape_ok = coherent["iq_real"].shape == scrambled["iq_real"].shape == (N_PAIRS, 12000)
        metadata_equal = all(np.array_equal(np.asarray(coherent[key]), np.asarray(scrambled[key])) for key in required - {"iq_real", "iq_imag"})
        pair = np.asarray(coherent["pair_seed"]).reshape(-1)
        receiver = np.asarray(coherent["receiver_noise_seed"]).reshape(-1)
        # The intervention preserves a PRI's pointwise magnitude relative to
        # the coherent component, but total IQ differs because its phase is altered.
        coherent_iq = np.asarray(coherent["iq_real"]) + 1j * np.asarray(coherent["iq_imag"])
        scrambled_iq = np.asarray(scrambled["iq_real"]) + 1j * np.asarray(scrambled["iq_imag"])
        iq_changed = not np.array_equal(coherent_iq, scrambled_iq)
    historic_overlaps: list[dict[str, object]] = []
    # Deliberately inspect only explicitly named development/training files.
    # No test file, including test-v3, is globbed or opened.
    for historic in DATA_ROOT.rglob("*train*.mat"):
        if historic.resolve() in {args.coherent_path.resolve(), args.scrambled_path.resolve()}:
            continue
        try:
            ids = _receiver_ids(historic)
        except (OSError, KeyError):
            continue
        historic_overlaps.append({"path": str(historic), "receiver_overlap_count": int(np.intersect1d(receiver, ids).size)})
    checks = {
        "required_fields_present": common_keys,
        "payload_shape_exact": shape_ok,
        "pair_identity_exact": np.array_equal(pair, np.arange(880_000_000, 880_000_000 + N_PAIRS)),
        "receiver_identity_unique": len(np.unique(receiver)) == N_PAIRS,
        "non_iq_metadata_exact": metadata_equal,
        "iq_changed_by_declared_phase_intervention": iq_changed,
        "historical_train_receiver_overlap_zero": not any(row["receiver_overlap_count"] for row in historic_overlaps),
        "test_v3_still_sealed": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"H0n audit failed: {checks}")
    result = {
        "status": "completed_train_only_paired_audit",
        "experiment": "RouteA_H0n_MFTCrossPulseCoherence",
        "coherent_sha256": _sha256(args.coherent_path),
        "scrambled_sha256": _sha256(args.scrambled_path),
        "sample_count_per_arm": N_PAIRS,
        "contracts": checks,
        "historical_train_identity_audit": historic_overlaps,
        "identity_audit_passed": True,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coherent-path", type=Path, default=SOURCE / "mft_coherent.mat")
    parser.add_argument("--scrambled-path", type=Path, default=SOURCE / "mft_scrambled.mat")
    parser.add_argument("--receipt", type=Path, default=ROOT / "outputs/tapc/route_a_h0n_mft_crosspulse_generation/formal_generation_receipt.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0n_mft_crosspulse_generation/paired_identity_audit.json")
    result = audit(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
