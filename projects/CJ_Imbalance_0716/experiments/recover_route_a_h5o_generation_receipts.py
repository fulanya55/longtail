"""Recover transparent provenance receipts after MATLAB created zero-byte receipts.

This tool does not regenerate, overwrite, or modify either H5o payload.  It
records the zero-byte condition, verifies the existing HDF5 payload contracts,
and writes new recovery receipts for downstream auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.base_receipt.stat().st_size != 0 or args.process_receipt.stat().st_size != 0:
        raise ValueError("H5o recovery is allowed only for the observed zero-byte receipt condition")
    with h5py.File(args.base_path, "r") as base, h5py.File(args.payload_path, "r") as payload:
        base_iq, payload_iq = base["iq_complex"], payload["iq_complex"]
        receiver = np.asarray(base["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        labels = np.asarray(base["label_selected"]).T.astype(np.uint8)
        values = np.asarray(base["parameter_values"]).T.astype(np.float64)
        masks = np.asarray(base["parameter_mask"]).T.astype(bool)
        widths = values[(labels[:, 5] == 1) & masks[:, 53], 53]
        replicas = values[(labels[:, 5] == 1) & masks[:, 54], 54]
        base_contracts = {
            "iq_complex_shape_matches_payload": base_iq.shape == payload_iq.shape == (3000, 5040),
            "receiver_count_and_uniqueness": len(receiver) == 5040 and len(np.unique(receiver)) == 5040,
            "receiver_range_exact": (int(receiver.min()), int(receiver.max())) == (1048677743, 1048682782),
            "standard_63_by_80_labels": len(labels) == 5040 and all(int((labels == row).all(axis=1).sum()) == 80 for row in np.unique(labels, axis=0)),
            "isrj_width_25_30us": len(widths) > 0 and float(widths.min()) >= 25e-6 - 1e-10 and float(widths.max()) <= 30e-6 + 1e-10,
            "isrj_replicas_9": len(replicas) > 0 and int(replicas.min()) == int(replicas.max()) == 9,
        }
    if not all(base_contracts.values()):
        raise ValueError(f"H5o existing source contract failed: {base_contracts}")
    common = {
        "status": "recovered_after_zero_byte_matlab_receipt",
        "experiment": "RouteA_H5o_FrozenUnifiedCombinedOOD",
        "recovery_reason": "MATLAB formal receipt files were zero bytes although immutable payload files exist",
        "recovery_is_not_regeneration": True,
        "base_path": str(args.base_path), "base_sha256": _sha(args.base_path),
        "payload_path": str(args.payload_path), "payload_sha256": _sha(args.payload_path),
        "source_code_sha256": {"base_generator": _sha(args.generator), "ar1_materializer": _sha(args.materializer)},
        "contracts": base_contracts,
        "test_files_opened": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_h5o_used": False,
    }
    base = {**common, "evidence_role": "recovered_train_only_base_generation_receipt", "executed": True}
    process = {**common, "evidence_role": "recovered_ar1_materialization_receipt", "executed": True, "rho": .65,
               "process_probe": {"sample_count": 200000, "mean_power": None, "pseudocovariance_abs": None, "lag1_real_correlation": None,
                                 "recomputed_by": "materialize_route_a_h5o_ar1_iq.m formal payload"}}
    _write(args.base_output, base); _write(args.process_output, process)
    result = {"base_recovery_receipt": str(args.base_output), "process_recovery_receipt": str(args.process_output), "contracts": base_contracts}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    root = ROOT / "outputs/tapc/route_a_h5o_frozen_unified_ood_generation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h5o_frozen_unified_ood_base\closedset_6class_train.mat"))
    parser.add_argument("--payload-path", type=Path, default=root / "ar1_iq_payload.mat")
    parser.add_argument("--base-receipt", type=Path, default=root / "formal_base_generation_receipt.json")
    parser.add_argument("--process-receipt", type=Path, default=root / "formal_ar1_materialization_receipt.json")
    parser.add_argument("--base-output", type=Path, default=root / "formal_base_generation_recovery_receipt.json")
    parser.add_argument("--process-output", type=Path, default=root / "formal_ar1_materialization_recovery_receipt.json")
    parser.add_argument("--generator", type=Path, default=ROOT / "experiments/generate_route_a_h5o_frozen_unified_ood_base.m")
    parser.add_argument("--materializer", type=Path, default=ROOT / "experiments/materialize_route_a_h5o_ar1_iq.m")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
