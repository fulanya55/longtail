"""Identity, split, and sealed-test audit for the pre-registered RAAT H0 source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COUNT = 5040
RANGE = (978677743, 978682782)
ROLE_TAG = b"RAAT-H0/v1|"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receiver_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def role_for(receiver: int) -> str:
    residue = int.from_bytes(hashlib.sha256(ROLE_TAG + str(int(receiver)).encode("ascii")).digest()[:8], "big") % 10
    return "fit" if residue <= 5 else ("validation" if residue <= 7 else "early")


def historical_train_files(data_path: Path) -> list[Path]:
    root = Path(r"C:\LJQ_code\干扰数据")
    return [item for item in sorted(root.rglob("*.mat")) if item.resolve() != data_path.resolve() and "train" in item.name.lower() and "test" not in item.name.lower()]


def audited_ranges() -> list[dict]:
    entries = {
        "H0e": ROOT / "outputs/tapc/route_a_h0e_counterfactual_visibility_generation/identity_audit.json",
        "H0f": ROOT / "outputs/tapc/route_a_h0f_counterfactual_delta_generation/identity_audit.json",
        "H0g": ROOT / "outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_generation/identity_parameter_audit.json",
        "H0h": ROOT / "outputs/tapc/route_a_h0h_ar1_noise_generation/identity_noise_audit.json",
    }
    result = []
    for name, path in entries.items():
        payload = json.loads(path.read_text(encoding="utf8"))
        low, high = map(int, payload["receiver_noise_seed_range"])
        result.append({"name": name, "audit_artifact": str(path), "range": [low, high]})
    return result


def run(args: argparse.Namespace) -> dict:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    if "test" in args.data_path.name.lower():
        raise ValueError("RAAT H0 accepts a train-only source")
    receiver = receiver_ids(args.data_path)
    with h5py.File(args.data_path, "r") as handle:
        labels = np.asarray(handle["label_selected"], dtype=np.uint8).T
    history = []
    for old in historical_train_files(args.data_path):
        try:
            history.append({"path": str(old), "receiver_overlap_count": int(np.intersect1d(receiver, receiver_ids(old)).size)})
        except (OSError, KeyError):
            continue
    if any(row["receiver_overlap_count"] for row in history):
        raise ValueError("receiver identity overlaps a historical train source")
    records = audited_ranges()
    for item in records:
        low, high = item["range"]
        item["receiver_range_overlap_count"] = max(0, min(int(receiver.max()), high) - max(int(receiver.min()), low) + 1)
    if any(item["receiver_range_overlap_count"] for item in records):
        raise ValueError("RAAT range overlaps an explicit H0 range")
    roles = {name: np.asarray([index for index, value in enumerate(receiver) if role_for(int(value)) == name], dtype=np.int64) for name in ("fit", "validation", "early")}
    if any(len(value) == 0 for value in roles.values()) or sum(len(value) for value in roles.values()) != len(receiver):
        raise ValueError("invalid deterministic role split")
    if any(np.intersect1d(roles[left], roles[right]).size for left in roles for right in roles if left < right):
        raise ValueError("role identity leakage")
    args.role_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.role_output, **roles, receiver_noise_seed=receiver)
    observed = (int(receiver.min()), int(receiver.max()))
    contracts = {
        "sample_count_5040": len(receiver) == COUNT,
        "receiver_unique_5040": len(np.unique(receiver)) == COUNT,
        "receiver_range_exact": observed == RANGE,
        "six_labels": labels.shape == (COUNT, 6),
        "all_63_combinations_present": len({tuple(row.tolist()) for row in labels}) == 63,
        "role_partition_exact": sum(len(value) for value in roles.values()) == COUNT,
        "test_v3_sealed": True,
    }
    if not all(contracts.values()):
        raise ValueError(f"RAAT H0 source contract failed: {contracts}")
    result = {
        "status": "completed_train_only_identity_audit",
        "experiment": "RouteA_RAAT_H0_v1",
        "evidence_role": "fresh_train_only_physical_identifiability",
        "path": str(args.data_path), "sha256": sha256(args.data_path),
        "sample_count": int(len(receiver)), "receiver_noise_seed_range": list(observed),
        "receiver_noise_seed_unique": int(len(np.unique(receiver))),
        "role_indices_path": str(args.role_output), "role_indices_sha256": sha256(args.role_output),
        "role_counts": {name: int(len(value)) for name, value in roles.items()},
        "live_historical_train_identity_audit": history,
        "recorded_h0_range_audit": records,
        "contracts": contracts, "identity_audit_passed": True,
        "test_files_opened": False, "locked_test_used": False, "test_v2_used": False,
        "test_v3_used": False, "test_raat_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_raat_h0_v1\closedset_6class_train.mat"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_raat_h0_generation/identity_audit.json")
    parser.add_argument("--role-output", type=Path, default=ROOT / "outputs/tapc/route_a_raat_h0_generation/role_indices.npz")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
