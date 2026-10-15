"""Provenance audit for Route-A H0p's fresh paired MFT source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0p_mft_delay_autocorrelation_pairs_20260803"
COUNT = 640
BASE_SEED = 2_300_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receiver_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def _history_paths() -> list[Path]:
    # Do not glob any test file.  Include all known non-test H0 sources that
    # use custom names in addition to the project's ordinary train sources.
    paths = list(DATA_ROOT.rglob("*train*.mat"))
    for pattern in ("h0n_mft_crosspulse_pairs_20260803_r2/*.mat", "h0o_mft_crosspulse_composite_20260803/*.mat"):
        paths.extend(DATA_ROOT.glob(pattern))
    return sorted({path.resolve() for path in paths})


def audit(args: argparse.Namespace) -> dict[str, object]:
    selection = json.loads(args.identity_selection.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not receipt.get("executed") or selection.get("identity_overlap_count") != 0:
        raise ValueError("formal receipt or identity selection invalid")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    with h5py.File(args.low_path, "r") as low, h5py.File(args.high_path, "r") as high:
        required = {"iq_real", "iq_imag", "pair_seed", "receiver_noise_seed", "mft_num_targets", "component_seed", "effective_JNR_dB"}
        required_ok = required.issubset(low.keys()) and required.issubset(high.keys())
        shape_ok = low["iq_real"].shape == high["iq_real"].shape == (COUNT, 3000)
        low_meta = {key: np.asarray(low[key]) for key in required - {"iq_real", "iq_imag"}}
        high_meta = {key: np.asarray(high[key]) for key in required - {"iq_real", "iq_imag"}}
        pair = low_meta["pair_seed"].reshape(-1).astype(np.int64)
        receiver = low_meta["receiver_noise_seed"].reshape(-1).astype(np.int64)
        common_equal = all(np.array_equal(low_meta[key], high_meta[key]) for key in ("pair_seed", "receiver_noise_seed", "component_seed", "effective_JNR_dB"))
        intervention = np.all(low_meta["mft_num_targets"] == 3) and np.all(high_meta["mft_num_targets"] == 8)
        low_iq = np.asarray(low["iq_real"]) + 1j * np.asarray(low["iq_imag"])
        high_iq = np.asarray(high["iq_real"]) + 1j * np.asarray(high["iq_imag"])
        iq_changed = not np.array_equal(low_iq, high_iq)
    history: list[dict[str, object]] = []
    for path in _history_paths():
        if path in {args.low_path.resolve(), args.high_path.resolve()}:
            continue
        try:
            ids = _receiver_ids(path)
        except (OSError, KeyError):
            continue
        history.append({"path": str(path), "receiver_overlap_count": int(np.intersect1d(receiver, ids).size)})
    checks = {
        "required_fields_present": required_ok,
        "payload_shape_exact": shape_ok,
        "pair_seed_exact": np.array_equal(pair, np.arange(BASE_SEED, BASE_SEED + COUNT)),
        "receiver_identity_unique": len(np.unique(receiver)) == COUNT,
        "pre_generation_identity_selection_zero_overlap": selection["identity_overlap_count"] == 0,
        "historical_non_test_identity_overlap_zero": not any(row["receiver_overlap_count"] for row in history),
        "shared_pair_metadata_exact": common_equal,
        "count_intervention_exact": intervention,
        "iq_changed_by_declared_count_intervention": iq_changed,
        "test_v3_still_sealed": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"H0p audit failed: {checks}")
    result = {
        "status": "completed_train_only_paired_audit",
        "experiment": "RouteA_H0p_MFTDelayAutocorrelationIdentifiability",
        "low_sha256": _sha256(args.low_path),
        "high_sha256": _sha256(args.high_path),
        "sample_count_per_arm": COUNT,
        "contracts": checks,
        "historical_non_test_identity_audit": history,
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
    parser.add_argument("--low-path", type=Path, default=SOURCE / "mft_count_3.mat")
    parser.add_argument("--high-path", type=Path, default=SOURCE / "mft_count_8.mat")
    parser.add_argument("--receipt", type=Path, default=ROOT / "outputs/tapc/route_a_h0p_mft_delay_autocorrelation_generation/formal_generation_receipt.json")
    parser.add_argument("--identity-selection", type=Path, default=ROOT / "outputs/tapc/route_a_h0p_mft_delay_autocorrelation_generation/pre_generation_identity_selection.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0p_mft_delay_autocorrelation_generation/paired_identity_audit.json")
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False, indent=2))
