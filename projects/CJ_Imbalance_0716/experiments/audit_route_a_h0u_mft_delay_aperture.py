"""Strict source and identity audit for the preregistered Route-A H0u source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0u_mft_delay_aperture_20260803_r1"
COUNT, BASE_SEED, RECEIVER_OFFSET = 640, 3_900_000_000, 2 * 2_246_822_519


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def audit(args: argparse.Namespace) -> dict[str, object]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not receipt.get("executed"):
        raise ValueError("formal source receipt is not executed")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    fields = {"iq_real", "iq_imag", "pair_seed", "receiver_noise_seed", "mft_present", "background_component_seeds", "mft_component_seed", "mft_total_jnr_db", "mft_delay_samples", "mft_amplitude", "num_pulses", "samples_per_pri"}
    with h5py.File(args.absent_path, "r") as absent, h5py.File(args.present_path, "r") as present:
        required = fields.issubset(absent.keys()) and fields.issubset(present.keys())
        shape_ok = absent["iq_real"].shape == present["iq_real"].shape == (COUNT, 10_000)
        a = {name: np.asarray(absent[name]) for name in fields - {"iq_real", "iq_imag"}}
        p = {name: np.asarray(present[name]) for name in fields - {"iq_real", "iq_imag"}}
        aiq = np.asarray(absent["iq_real"]) + 1j * np.asarray(absent["iq_imag"])
        piq = np.asarray(present["iq_real"]) + 1j * np.asarray(present["iq_imag"])
    receiver = a["receiver_noise_seed"].reshape(-1).astype(np.int64)
    expected_receiver = ((np.arange(BASE_SEED, BASE_SEED + COUNT, dtype=np.uint64) + np.uint64(RECEIVER_OFFSET)) % np.uint64(2**32)).astype(np.int64)
    historic: list[dict[str, object]] = []
    for path in DATA_ROOT.rglob("*train*.mat"):
        if path.resolve() in {args.absent_path.resolve(), args.present_path.resolve()}:
            continue
        try:
            overlap = int(np.intersect1d(receiver, _ids(path)).size)
        except (OSError, KeyError):
            continue
        historic.append({"path": str(path), "receiver_overlap_count": overlap})
    # MATLAB's 3-by-N payload is exposed by h5py as N-by-3.  Sort replicas
    # within each identity rather than down a jitter column across identities.
    delays = p["mft_delay_samples"]
    if delays.shape != (COUNT, 3):
        raise ValueError(f"unexpected H0u delay payload shape: {delays.shape}")
    minimum_spacing = np.min(np.diff(np.sort(delays, axis=1), axis=1))
    checks = {
        "required_fields_present": required,
        "payload_shape_exact": shape_ok,
        "pair_seed_exact": np.array_equal(a["pair_seed"].reshape(-1), np.arange(BASE_SEED, BASE_SEED + COUNT)),
        "receiver_mapping_exact": np.array_equal(receiver, expected_receiver),
        "receiver_identity_unique": len(np.unique(receiver)) == COUNT,
        "historical_train_receiver_identity_overlap_zero": not any(row["receiver_overlap_count"] for row in historic),
        "background_substreams_exact": np.array_equal(a["background_component_seeds"], p["background_component_seeds"]),
        "receiver_and_pair_metadata_exact": np.array_equal(a["pair_seed"], p["pair_seed"]) and np.array_equal(a["receiver_noise_seed"], p["receiver_noise_seed"]),
        "only_mft_metadata_and_iq_intervene": np.all(a["mft_present"] == 0) and np.all(p["mft_present"] == 1) and np.all(np.isnan(a["mft_component_seed"])) and np.all(np.isfinite(p["mft_component_seed"])) and not np.array_equal(aiq, piq),
        "mft_jnr_exact": np.allclose(p["mft_total_jnr_db"], 8.0),
        "separated_replica_contract": bool(minimum_spacing >= 1600) and np.all(np.isfinite(delays)) and np.all(np.isnan(a["mft_delay_samples"])),
        "one_pri_one_ms_contract": np.all(a["num_pulses"] == 1) and np.all(p["num_pulses"] == 1) and np.all(a["samples_per_pri"] == 10_000) and np.all(p["samples_per_pri"] == 10_000),
        "test_v3_still_sealed": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"H0u provenance contract failed: {checks}")
    result = {"status": "completed_train_only_paired_audit", "experiment": "RouteA_H0u_MFTDelayAperture", "sample_count_per_arm": COUNT, "absent_sha256": _sha256(args.absent_path), "present_sha256": _sha256(args.present_path), "minimum_observed_replica_spacing_samples": int(minimum_spacing), "contracts": checks, "historical_train_identity_audit": historic, "identity_audit_passed": True, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--absent-path", type=Path, default=SOURCE / "mft_absent.mat")
    parser.add_argument("--present-path", type=Path, default=SOURCE / "mft_present_weak_separated.mat")
    parser.add_argument("--receipt", type=Path, default=ROOT / "outputs/tapc/route_a_h0u_mft_delay_aperture_generation/formal_generation_receipt.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0u_mft_delay_aperture_generation/paired_identity_audit.json")
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False, indent=2))
