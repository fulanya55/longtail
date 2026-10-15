"""Strict provenance audit for the official-generator Route-A H0o source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "h0o_mft_crosspulse_composite_20260803"
COUNT = 640
BASE_SEED = 2_000_000_000


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
    selection = json.loads(args.identity_selection.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if not receipt.get("executed") or selection.get("identity_overlap_count") != 0:
        raise ValueError("formal receipt or pre-generation identity selection invalid")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    fields = {"iq_real", "iq_imag", "pair_seed", "receiver_noise_seed", "mft_present", "background_component_seeds", "mft_component_seed", "mft_effective_jnr_db", "num_pulses"}
    with h5py.File(args.absent_path, "r") as absent, h5py.File(args.present_path, "r") as present:
        required = fields.issubset(absent.keys()) and fields.issubset(present.keys())
        # h5py presents MATLAB's 12000-by-640 payload as [batch, samples].
        shape_ok = absent["iq_real"].shape == present["iq_real"].shape == (COUNT, 12000)
        a = {field: np.asarray(absent[field]) for field in fields - {"iq_real", "iq_imag"}}
        p = {field: np.asarray(present[field]) for field in fields - {"iq_real", "iq_imag"}}
        absent_iq = np.asarray(absent["iq_real"]) + 1j * np.asarray(absent["iq_imag"])
        present_iq = np.asarray(present["iq_real"]) + 1j * np.asarray(present["iq_imag"])
        pair_seed = a["pair_seed"].reshape(-1).astype(np.int64)
        receiver = a["receiver_noise_seed"].reshape(-1).astype(np.int64)
        background_equal = np.array_equal(a["background_component_seeds"], p["background_component_seeds"])
        common_equal = np.array_equal(a["pair_seed"], p["pair_seed"]) and np.array_equal(a["receiver_noise_seed"], p["receiver_noise_seed"]) and np.array_equal(a["num_pulses"], p["num_pulses"])
        mft_contract = np.all(a["mft_present"] == 0) and np.all(p["mft_present"] == 1) and np.all(np.isnan(a["mft_component_seed"])) and np.all(np.isfinite(p["mft_component_seed"])) and np.all(np.isnan(a["mft_effective_jnr_db"])) and np.allclose(p["mft_effective_jnr_db"], 8.0)
        iq_changed = not np.array_equal(absent_iq, present_iq)
    historic: list[dict[str, object]] = []
    for path in DATA_ROOT.rglob("*train*.mat"):
        if path.resolve() in {args.absent_path.resolve(), args.present_path.resolve()}:
            continue
        try:
            ids = _receiver_ids(path)
        except (OSError, KeyError):
            continue
        historic.append({"path": str(path), "receiver_overlap_count": int(np.intersect1d(receiver, ids).size)})
    checks = {
        "required_fields_present": required,
        "payload_shape_exact": shape_ok,
        "pair_seed_exact": np.array_equal(pair_seed, np.arange(BASE_SEED, BASE_SEED + COUNT)),
        "receiver_identity_unique": len(np.unique(receiver)) == COUNT,
        "pre_generation_identity_selection_zero_overlap": selection["identity_overlap_count"] == 0,
        "post_generation_historical_train_identity_overlap_zero": not any(row["receiver_overlap_count"] for row in historic),
        "background_substreams_exact": background_equal,
        "receiver_and_pair_metadata_exact": common_equal,
        "only_mft_contract_metadata_differs": mft_contract,
        "iq_changed_by_declared_mft_intervention": iq_changed,
        "test_v3_still_sealed": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"H0o provenance contract failed: {checks}")
    result = {
        "status": "completed_train_only_paired_audit",
        "experiment": "RouteA_H0o_MFTCrossPulseCompositeAttribution",
        "absent_sha256": _sha256(args.absent_path),
        "present_sha256": _sha256(args.present_path),
        "sample_count_per_arm": COUNT,
        "contracts": checks,
        "pre_generation_identity_selection": str(args.identity_selection),
        "historical_train_identity_audit": historic,
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
    parser.add_argument("--absent-path", type=Path, default=SOURCE / "mft_absent.mat")
    parser.add_argument("--present-path", type=Path, default=SOURCE / "mft_present_weak_m12db.mat")
    parser.add_argument("--receipt", type=Path, default=ROOT / "outputs/tapc/route_a_h0o_mft_crosspulse_generation/formal_generation_receipt.json")
    parser.add_argument("--identity-selection", type=Path, default=ROOT / "outputs/tapc/route_a_h0o_mft_crosspulse_generation/pre_generation_identity_selection.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0o_mft_crosspulse_generation/paired_identity_audit.json")
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False, indent=2))
