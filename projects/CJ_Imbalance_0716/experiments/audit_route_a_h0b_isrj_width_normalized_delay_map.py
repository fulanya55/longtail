"""Audit the fresh oracle width-normalized ISRJ H0b source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ISRJ, WIDTH, REPLICA = 5, 53, 54


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_width_normalized_h0b\closedset_6class_train.mat"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h0b_isrj_width_normalized_generation/identity_audit.json")
    parser.add_argument("--roles-output", type=Path, default=ROOT / "outputs/tapc/route_a_h0b_isrj_width_normalized_generation/role_indices.npz")
    args = parser.parse_args()
    with h5py.File(args.data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        labels = np.asarray(handle["label_selected"], dtype=np.uint8).T
        values = np.asarray(handle["parameter_values"], dtype=np.float64).T
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).T.astype(bool)
    active = np.flatnonzero(labels[:, ISRJ] == 1)
    if len(receiver) != 5040 or np.unique(receiver).size != 5040 or len(active) != 2560:
        raise ValueError("H0b sample/identity contract failed")
    widths, replicas = values[active, WIDTH], values[active, REPLICA]
    if not np.all(mask[active, WIDTH] & mask[active, REPLICA]) or not np.all(np.isfinite(widths)) or not np.all(np.isfinite(replicas)) or widths.min() < 10e-6 - 1e-9 or widths.max() > 30e-6 + 1e-9 or replicas.min() < 3 or replicas.max() > 8:
        raise ValueError("H0b active parameter contract failed")
    remainder = receiver % 10
    roles = {"fit": active[remainder[active] <= 5], "early_stop": active[(remainder[active] >= 6) & (remainder[active] <= 7)], "evaluation": active[remainder[active] >= 8]}
    if [len(roles[key]) for key in ("fit", "early_stop", "evaluation")] != [1536, 512, 512]:
        raise ValueError("H0b role count contract failed")
    if any(np.intersect1d(receiver[roles[a]], receiver[roles[b]]).size for a in roles for b in roles if a < b):
        raise ValueError("H0b receiver leakage")
    history = []
    for old in Path(r"C:\LJQ_code\干扰数据").glob("复合干扰数据*\**\closedset_6class_train.mat"):
        if "0802_isrj_width_normalized_h0b" in str(old):
            continue
        try:
            with h5py.File(old, "r") as handle:
                old_receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
            history.append({"path": str(old), "receiver_overlap_count": int(np.intersect1d(receiver, old_receiver).size)})
        except OSError:
            continue
    if any(item["receiver_overlap_count"] for item in history):
        raise ValueError("H0b historical train identity overlap")
    report = {"status": "completed_train_only_identity_audit", "experiment": "RouteA_H0b_ISRJWidthNormalizedDelayMap", "evidence_role": "fresh_train_only_oracle_width_normalized_replica_identifiability", "path": str(args.data_path), "sha256": _sha(args.data_path), "sample_count": int(len(receiver)), "receiver_noise_seed_range": [int(receiver.min()), int(receiver.max())], "receiver_noise_seed_unique": int(np.unique(receiver).size), "isrj_eligible_count": int(len(active)), "width_range_s": [float(widths.min()), float(widths.max())], "replica_range": [int(replicas.min()), int(replicas.max())], "roles": {name: {"count": int(len(index)), "receiver_count": int(np.unique(receiver[index]).size), "label_positive_counts": [int(value) for value in labels[index].sum(0)], "replica_histogram": {str(replica): int((values[index, REPLICA] == replica).sum()) for replica in range(3, 9)}} for name, index in roles.items()}, "historical_train_identity_audit": history, "identity_audit_passed": True, "test_files_opened": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_h0b_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); np.savez_compressed(args.roles_output, **roles); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
