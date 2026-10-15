"""Audit the fresh ISRJ delay-block H0 source and store identity-safe roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ISRJ, REPLICA = 5, 54


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_paths() -> list[Path]:
    base = Path(r"C:\LJQ_code\干扰数据")
    return [item for item in base.glob("复合干扰数据*\**\closedset_6class_train.mat") if "0802_isrj_delay_block_h0" not in str(item)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_delay_block_h0\closedset_6class_train.mat"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0_isrj_delay_block_generation/identity_audit.json")
    parser.add_argument("--roles-output", type=Path, default=ROOT / "outputs/tapc/route_a_h0_isrj_delay_block_generation/role_indices.npz")
    args = parser.parse_args()
    with h5py.File(args.data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        labels = np.asarray(handle["label_selected"], dtype=np.uint8).T
        values = np.asarray(handle["parameter_values"], dtype=np.float64).T
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).T.astype(bool)
    if len(receiver) != 5040 or np.unique(receiver).size != 5040:
        raise ValueError("H0 source must contain 5,040 unique receiver identities")
    active = np.flatnonzero(labels[:, ISRJ] == 1)
    if len(active) != 2560 or not np.all(mask[active, REPLICA]) or not np.all(np.isfinite(values[active, REPLICA])):
        raise ValueError("ISRJ replica target contract failed")
    replicas = values[active, REPLICA]
    if replicas.min() < 3 or replicas.max() > 8 or not np.allclose(replicas, np.round(replicas)):
        raise ValueError("ISRJ replica range contract failed")
    remainder = receiver % 10
    roles = {"fit": active[remainder[active] <= 5], "early_stop": active[(remainder[active] >= 6) & (remainder[active] <= 7)], "evaluation": active[remainder[active] >= 8]}
    expected = {"fit": 1536, "early_stop": 512, "evaluation": 512}
    if {name: len(index) for name, index in roles.items()} != expected:
        raise ValueError("H0 role count contract failed")
    for left in roles:
        for right in roles:
            if left < right and np.intersect1d(receiver[roles[left]], receiver[roles[right]]).size:
                raise ValueError("receiver identity leakage")
    overlaps = []
    for old_path in _history_paths():
        try:
            with h5py.File(old_path, "r") as handle:
                old = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
            overlaps.append({"path": str(old_path), "receiver_overlap_count": int(np.intersect1d(receiver, old).size)})
        except OSError:
            continue
    if any(row["receiver_overlap_count"] for row in overlaps):
        raise ValueError("H0 source overlaps a historical train source")
    payload = {
        "status": "completed_train_only_identity_audit",
        "experiment": "RouteA_H0_ISRJDelayBlockSelfSimilarity",
        "evidence_role": "fresh_train_only_isrj_replica_identifiability",
        "path": str(args.data_path),
        "sha256": _sha(args.data_path),
        "sample_count": int(len(receiver)),
        "receiver_noise_seed_range": [int(receiver.min()), int(receiver.max())],
        "receiver_noise_seed_unique": int(np.unique(receiver).size),
        "isrj_eligible_count": int(len(active)),
        "replica_range": [int(replicas.min()), int(replicas.max())],
        "roles": {name: {"count": int(len(index)), "receiver_count": int(np.unique(receiver[index]).size), "label_positive_counts": [int(value) for value in labels[index].sum(axis=0)], "replica_histogram": {str(item): int((values[index, REPLICA] == item).sum()) for item in range(3, 9)}} for name, index in roles.items()},
        "historical_train_identity_audit": overlaps,
        "identity_audit_passed": True,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_h0_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.roles_output, **roles)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
