"""Audit train-only development-v5 without opening frozen or test files."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNT = 25200
EXPECTED_SAMPLE_SEED_RANGE = (44000001, 44025200)
EXPECTED_RECEIVER_RANGE = (242677743, 242702942)
REFERENCE_RECEIVER_RANGES = {
    "development_v2_train": (218938474, 218963673),
    "test_v2_frozen_contract": (221938474, 221963673),
    "development_v3_train": (222677743, 222702942),
    "sealed_test_v3_contract": (225677743, 225702942),
    "development_v4_train": (232677743, 232702942),
    "reserved_test_v4_contract": (235677743, 235702942),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disjoint(first, second):
    return int(first[1]) < int(second[0]) or int(first[0]) > int(second[1])


def audit_dataset(path):
    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1)
        label_key = "label_selected" if "label_selected" in handle else "known_label"
        labels = np.asarray(handle[label_key])
    if receiver.size != EXPECTED_COUNT or EXPECTED_COUNT not in labels.shape:
        raise ValueError("development-v5 sample count differs from preregistration")
    if not np.all(np.isfinite(receiver)):
        raise ValueError("development-v5 receiver identities contain non-finite values")
    observed = (int(receiver.min()), int(receiver.max()))
    if observed != EXPECTED_RECEIVER_RANGE:
        raise ValueError("development-v5 receiver range differs from contract")
    if np.unique(receiver).size != EXPECTED_COUNT:
        raise ValueError("development-v5 receiver identities are not unique")
    disjoint = {name: bool(_disjoint(observed, value)) for name, value in REFERENCE_RECEIVER_RANGES.items()}
    if not all(disjoint.values()):
        raise ValueError("development-v5 overlaps a prior/frozen role")
    return {
        "status": "completed",
        "evidence_role": "development_v5",
        "development_v5_path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "sample_count": EXPECTED_COUNT,
        "expected_sample_seed_range": list(EXPECTED_SAMPLE_SEED_RANGE),
        "observed_receiver_noise_seed_range": list(observed),
        "receiver_noise_seed_unique": EXPECTED_COUNT,
        "reference_receiver_noise_seed_ranges": {name: list(value) for name, value in REFERENCE_RECEIVER_RANGES.items()},
        "receiver_identity_disjoint": disjoint,
        "identity_audit_passed": True,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
        "test_v5_used": False,
    }


def run(args):
    payload = audit_dataset(args.data_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_v5\closedset_6class_train.mat"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v5_generation" / "identity_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
