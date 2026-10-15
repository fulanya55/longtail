"""Audit H5o source provenance before frozen-candidate inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
COUNT = 5_040
RANGE = (1_048_677_743, 1_048_682_782)
ISRJ_WIDTH_COL, ISRJ_REPLICA_COL = 53, 54


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receiver(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def run(args: argparse.Namespace) -> dict[str, object]:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    material = json.loads(args.dataset_receipt.read_text(encoding="utf-8"))
    process = json.loads(args.process_receipt.read_text(encoding="utf-8"))
    if material.get("status") != "completed" or not process.get("executed", False):
        raise ValueError("H5o materialization receipt is incomplete")
    if material["base_sha256"] != _sha(args.base_path) or material["output_sha256"] != _sha(args.data_path):
        raise ValueError("H5o materialization hash mismatch")
    if float(material["rho"]) != .65 or float(process["rho"]) != .65:
        raise ValueError("H5o AR(1) rho mismatch")
    probe = process["process_probe"]
    probe_ok = (int(probe["sample_count"]) == 200000 and abs(float(probe["mean_power"]) - 1.0) <= .02
                and float(probe["pseudocovariance_abs"]) <= .02 and abs(float(probe["lag1_real_correlation"]) - .65) <= .02)
    with h5py.File(args.base_path, "r") as base, h5py.File(args.data_path, "r") as derived:
        receiver = np.asarray(derived["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        base_receiver = np.asarray(base["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        labels = np.asarray(derived["label_selected"]).T.astype(np.uint8)
        values = np.asarray(derived["parameter_values"]).T.astype(np.float64)
        masks = np.asarray(derived["parameter_mask"]).T.astype(bool)
        labels_exact = np.array_equal(derived["label_selected"][...], base["label_selected"][...])
        non_iq_exact = all(np.array_equal(derived[key][...], base[key][...], equal_nan=True) for key in base.keys() if key != "iq_complex" and isinstance(base[key], h5py.Dataset) and base[key].dtype.kind != "O")
        columns = np.array([0, 71, 233, 997, 2029, 3511, 5039])
        iq_changed = not np.array_equal(derived["iq_complex"][:, columns], base["iq_complex"][:, columns])
    present = labels[:, 5].astype(bool)
    widths = values[present & masks[:, ISRJ_WIDTH_COL], ISRJ_WIDTH_COL]
    replicas = values[present & masks[:, ISRJ_REPLICA_COL], ISRJ_REPLICA_COL]
    historic = []
    excluded = {args.data_path.resolve(), args.base_path.resolve()}
    for path in sorted(DATA_ROOT.rglob("*.mat")):
        if path.resolve() in excluded or "test" in path.name.lower() or "train" not in path.name.lower():
            continue
        try:
            historic.append({"path": str(path), "receiver_overlap_count": int(np.intersect1d(receiver, _receiver(path)).size)})
        except (OSError, KeyError):
            continue
    if any(row["receiver_overlap_count"] for row in historic):
        raise ValueError("H5o receiver identity overlaps a historical train source")
    contracts = {
        "sample_count_5040": len(receiver) == COUNT,
        "unique_receiver_identities": len(np.unique(receiver)) == COUNT,
        "receiver_range_exact": (int(receiver.min()), int(receiver.max())) == RANGE,
        "labels_exact_from_base": bool(labels_exact),
        "non_iq_payload_exact": bool(non_iq_exact),
        "iq_changed": bool(iq_changed),
        "isrj_width_25_30us": len(widths) > 0 and float(widths.min()) >= 25e-6 - 1e-10 and float(widths.max()) <= 30e-6 + 1e-10,
        "isrj_replicas_9": len(replicas) > 0 and int(replicas.min()) == 9 and int(replicas.max()) == 9,
        "proper_complex_ar1_rho_0p65": bool(probe_ok),
        "test_v3_sealed": True,
    }
    if not all(contracts.values()):
        raise ValueError(f"H5o source contract failed: {contracts}")
    result = {
        "status": "completed_train_only_identity_parameter_noise_audit",
        "experiment": "RouteA_H5o_FrozenUnifiedCombinedOOD",
        "evidence_role": "fresh_combined_parameter_noise_ood_train_only",
        "path": str(args.data_path), "sha256": _sha(args.data_path),
        "base_path": str(args.base_path), "base_sha256": _sha(args.base_path),
        "sample_count": int(len(receiver)), "receiver_noise_seed_range": list(RANGE),
        "noise_model": {"proper_complex_ar1_rho": .65, "process_probe": probe},
        "contracts": contracts, "identity_audit_passed": True, "historic_train_identity_audit": historic,
        "test_files_opened": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_h5o_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h5o_frozen_unified_ood\closedset_6class_train.mat"))
    parser.add_argument("--base-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h5o_frozen_unified_ood_base\closedset_6class_train.mat"))
    parser.add_argument("--dataset-receipt", type=Path, default=ROOT / "outputs/tapc/route_a_h5o_frozen_unified_ood_generation/formal_dataset_materialization_receipt.json")
    parser.add_argument("--process-receipt", type=Path, default=ROOT / "outputs/tapc/route_a_h5o_frozen_unified_ood_generation/formal_ar1_materialization_receipt.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h5o_frozen_unified_ood_generation/identity_parameter_noise_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
