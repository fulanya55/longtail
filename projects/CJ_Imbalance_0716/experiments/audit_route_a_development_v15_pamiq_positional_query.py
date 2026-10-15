"""Audit the fresh v15 PAM-IQ position-query train role without opening tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNT = 25200
EXPECTED_RECEIVER = (648677743, 648702942)
REFERENCES = {
    "development_v2_to_v14_and_reserved_tests": (218938474, 345702942),
    "vft_isrj_occupancy_and_windowstate": (370677743, 380702942),
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
        labels = np.asarray(handle["label_selected"])
    observed = (int(receiver.min()), int(receiver.max()))
    if (
        receiver.size != COUNT
        or COUNT not in labels.shape
        or observed != EXPECTED_RECEIVER
        or np.unique(receiver).size != COUNT
    ):
        raise ValueError("development-v15 identity contract failed")
    disjoint = {
        name: observed[1] < bounds[0] or observed[0] > bounds[1]
        for name, bounds in REFERENCES.items()
    }
    if not all(disjoint.values()):
        raise ValueError("development-v15 overlaps a registered role")
    return {
        "status": "completed_train_only_identity_audit",
        "evidence_role": "development_v15_pamiq_positional_query",
        "train_path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "sample_count": COUNT,
        "observed_receiver_noise_seed_range": list(observed),
        "receiver_noise_seed_unique": COUNT,
        "reference_receiver_noise_seed_ranges": {
            name: list(bounds) for name, bounds in REFERENCES.items()
        },
        "receiver_identity_disjoint": disjoint,
        "identity_audit_passed": True,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v15_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v15_pamiq_positional_query\closedset_6class_train.mat"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/tapc/route_a_development_v15_pamiq_positional_query_generation/identity_audit.json",
    )
    args = parser.parse_args()
    payload = audit_train(args.data_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
