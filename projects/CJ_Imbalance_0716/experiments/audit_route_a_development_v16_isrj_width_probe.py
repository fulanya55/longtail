"""Audit the v16 ISRJ-width probe train role without opening any test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNT = 5040
EXPECTED_RECEIVER = (718677743, 718682782)
ISRJ_INDEX, WIDTH_INDEX, REPLICA_INDEX = 5, 53, 54
REFERENCES = {
    "registered_development_roles_through_v15": (218938474, 648702942),
    "vft_isrj_source_diagnostics": (352677743, 380702942),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_train(path: Path) -> dict:
    path = Path(path).resolve()
    if "test" in path.name.lower():
        raise ValueError("refusing to open a test-named file")
    with h5py.File(path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1)
        labels = np.asarray(handle["label_selected"], dtype=np.uint8).T
        values = np.asarray(handle["parameter_values"], dtype=np.float64).T
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).T.astype(bool)
    observed = (int(receiver.min()), int(receiver.max()))
    isrj = labels[:, ISRJ_INDEX].astype(bool)
    width = values[isrj & mask[:, WIDTH_INDEX], WIDTH_INDEX]
    replicas = values[isrj & mask[:, REPLICA_INDEX], REPLICA_INDEX]
    if (
        receiver.size != COUNT or observed != EXPECTED_RECEIVER
        or np.unique(receiver).size != COUNT or width.size != int(isrj.sum())
        or replicas.size != int(isrj.sum()) or width.min() < 10e-6 - 1e-9
        or width.max() > 30e-6 + 1e-9 or replicas.min() < 3 or replicas.max() > 8
    ):
        raise ValueError("development-v16 width/identity contract failed")
    disjoint = {
        name: observed[1] < bounds[0] or observed[0] > bounds[1]
        for name, bounds in REFERENCES.items()
    }
    if not all(disjoint.values()):
        raise ValueError("development-v16 overlaps a registered role")
    return {
        "status": "completed_train_only_identity_parameter_audit",
        "evidence_role": "development_v16_isrj_width_identifiability",
        "train_path": str(path), "sha256": _sha256(path),
        "sample_count": COUNT, "receiver_noise_seed_unique": COUNT,
        "observed_receiver_noise_seed_range": list(observed),
        "isrj_positive_count": int(isrj.sum()),
        "isrj_sample_width_s_observed": [float(width.min()), float(width.max())],
        "isrj_num_replicas_observed": [int(replicas.min()), int(replicas.max())],
        "reference_receiver_noise_seed_ranges": {key: list(value) for key, value in REFERENCES.items()},
        "receiver_identity_disjoint": disjoint, "identity_audit_passed": True,
        "test_files_opened": False, "locked_test_used": False,
        "test_v2_used": False, "test_v3_used": False, "test_v16_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v16_isrj_width_probe\closedset_6class_train.mat"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v16_isrj_width_probe_generation/identity_parameter_audit.json")
    args = parser.parse_args()
    payload = audit_train(args.data_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
