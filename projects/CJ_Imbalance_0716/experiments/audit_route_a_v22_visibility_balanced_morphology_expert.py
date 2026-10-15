"""Audit the fresh v22 source and write identity-isolated morphology roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from experiments.run_route_a_development_v2_shared_gradient import _component_power

VFT, ISRJ, WIDTH, REPLICA = 4, 5, 53, 54


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(data_path: Path, weak_threshold_db: float = 10.0):
    with h5py.File(data_path, "r") as h5:
        receiver = np.asarray(h5["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        labels = np.asarray(h5["label_selected"], dtype=np.uint8).T
        values = np.asarray(h5["parameter_values"], dtype=np.float64).T
        mask = np.asarray(h5["parameter_mask"], dtype=np.uint8).T.astype(bool)
    if len(receiver) != 5040 or np.unique(receiver).size != 5040:
        raise ValueError("v22 receiver identity contract failed")
    isrj = labels[:, ISRJ].astype(bool)
    widths = values[isrj & mask[:, WIDTH], WIDTH]
    replicas = values[isrj & mask[:, REPLICA], REPLICA]
    if not (widths.min() >= 10e-6 - 1e-9 and widths.max() <= 30e-6 + 1e-9 and replicas.min() >= 3 and replicas.max() <= 4):
        raise ValueError("v22 ISRJ parameter contract failed")
    weak = compute_weak_component_mask(labels, _component_power(values), weak_threshold_db)
    remainder = receiver % 10
    roles = {
        "fit": np.flatnonzero(isrj & (remainder <= 5)),
        "early_stop": np.flatnonzero(isrj & (remainder >= 6) & (remainder <= 7)),
        "evaluation": np.flatnonzero(isrj & (remainder >= 8)),
    }
    names = tuple(roles)
    if any(np.intersect1d(receiver[roles[a]], receiver[roles[b]]).size for i, a in enumerate(names) for b in names[i + 1 :]):
        raise ValueError("v22 receiver identity leakage")
    detail = {}
    for name, index in roles.items():
        weak_positive = (labels[index, VFT] == 1) & weak[index, VFT]
        strong_positive = (labels[index, VFT] == 1) & ~weak[index, VFT]
        hard_negative = labels[index, VFT] == 0
        if min(weak_positive.sum(), strong_positive.sum(), hard_negative.sum()) < 20:
            raise ValueError(f"v22 role {name} lacks weak/strong/negative stratum")
        detail[name] = {
            "count": int(len(index)),
            "receiver_count": int(np.unique(receiver[index]).size),
            "weak_vft_positive_count": int(weak_positive.sum()),
            "strong_vft_positive_count": int(strong_positive.sum()),
            "hard_negative_count": int(hard_negative.sum()),
        }
    return {
        "status": "completed_train_only_identity_audit",
        "evidence_role": "fresh_v22_visibility_balanced_morphology_expert",
        "path": str(data_path),
        "sha256": _sha(data_path),
        "sample_count": int(len(receiver)),
        "receiver_noise_seed_range": [int(receiver.min()), int(receiver.max())],
        "receiver_noise_seed_unique": int(np.unique(receiver).size),
        "width_range_s": [float(widths.min()), float(widths.max())],
        "replica_range": [int(replicas.min()), int(replicas.max())],
        "roles": detail,
        "identity_audit_passed": True,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v22_used": False,
    }, roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v22_visibility_balanced_morphology\closedset_6class_train.mat"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v22_visibility_balanced_morphology_generation/identity_audit.json")
    parser.add_argument("--roles-output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v22_visibility_balanced_morphology_generation/role_indices.npz")
    args = parser.parse_args()
    payload, roles = audit(args.data_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    np.savez_compressed(args.roles_output, **roles)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
