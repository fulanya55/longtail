"""Audit fresh v18 identities and write deterministic fit/select/eval roles."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VFT, ISRJ, WIDTH, REPLICA = 4, 5, 53, 54

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def audit(data_path: Path) -> tuple[dict, dict]:
    with h5py.File(data_path, "r") as h:
        receiver = np.asarray(h["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        labels = np.asarray(h["label_selected"], dtype=np.uint8).T
        values = np.asarray(h["parameter_values"], dtype=np.float64).T
        mask = np.asarray(h["parameter_mask"], dtype=np.uint8).T.astype(bool)
    if len(receiver) != 5040 or np.unique(receiver).size != 5040:
        raise ValueError("v18 receiver identity contract failed")
    isrj = labels[:, ISRJ].astype(bool); active = isrj & mask[:, WIDTH] & mask[:, REPLICA]
    widths = values[active, WIDTH]; replicas = values[active, REPLICA]
    if not (widths.min() >= 10e-6 - 1e-9 and widths.max() <= 30e-6 + 1e-9 and replicas.min() >= 3 and replicas.max() <= 4):
        raise ValueError("v18 ISRJ parameter contract failed")
    remainder = receiver % 10
    roles = {"fit": np.flatnonzero(remainder <= 5), "selection": np.flatnonzero((remainder >= 6) & (remainder <= 7)), "evaluation": np.flatnonzero(remainder >= 8)}
    if sum(len(x) for x in roles.values()) != len(receiver):
        raise ValueError("role partition incomplete")
    identities = [receiver[x] for x in roles.values()]
    if any(np.intersect1d(identities[i], identities[j]).size for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("receiver identity leakage across v18 roles")
    hard = (labels[:, VFT] == 0) & isrj
    payload = {
        "status": "completed_train_only_identity_audit", "evidence_role": "fresh_v18_bounded_width_residual_development",
        "path": str(data_path), "sha256": _sha(data_path), "sample_count": int(len(receiver)),
        "receiver_noise_seed_range": [int(receiver.min()), int(receiver.max())], "receiver_noise_seed_unique": int(np.unique(receiver).size),
        "isrj_positive_count": int(isrj.sum()), "hard_negative_count": int(hard.sum()),
        "width_range_s": [float(widths.min()), float(widths.max())], "replica_range": [int(replicas.min()), int(replicas.max())],
        "roles": {name: {"count": int(len(idx)), "receiver_count": int(np.unique(receiver[idx]).size), "hard_negative_count": int(hard[idx].sum())} for name, idx in roles.items()},
        "identity_audit_passed": True, "test_files_opened": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v18_used": False,
    }
    return payload, roles

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v18_bounded_width_residual\closedset_6class_train.mat")); parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v18_bounded_width_residual_generation/identity_audit.json"); parser.add_argument("--roles-output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v18_bounded_width_residual_generation/role_indices.npz"); args = parser.parse_args()
    result, roles = audit(args.data_path); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); np.savez_compressed(args.roles_output, **roles); print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
