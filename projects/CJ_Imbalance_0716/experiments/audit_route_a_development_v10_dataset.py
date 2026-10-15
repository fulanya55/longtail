"""Audit train-only development-v10 without opening any test file."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNT = 25200
EXPECTED_RECEIVER = (302677743, 302702942)
REFERENCES = {
    "development_v2_train": (218938474, 218963673), "test_v2_contract": (221938474, 221963673),
    "development_v3_train": (222677743, 222702942), "test_v3_contract": (225677743, 225702942),
    "development_v4_train": (232677743, 232702942), "test_v4_contract": (235677743, 235702942),
    "development_v5_train": (242677743, 242702942), "test_v5_contract": (245677743, 245702942),
    "development_v6_train": (252677743, 252702942), "test_v6_contract": (255677743, 255702942),
    "isrj_parameter_ood_v1": (262677743, 262682782),
    "development_v7_train": (272677743, 272702942), "test_v7_contract": (275677743, 275702942),
    "development_v8_train": (282677743, 282702942), "test_v8_contract": (285677743, 285702942),
    "development_v9_train": (292677743, 292702942), "test_v9_contract": (295677743, 295702942),
}


def _default_path():
    hits = list(Path(r"C:\LJQ_code").glob("*/*0801_v10/closedset_6class_train.mat"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one development-v10 MAT, found {len(hits)}")
    return hits[0]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path):
    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1)
        labels = np.asarray(handle["label_selected"])
    observed = (int(receiver.min()), int(receiver.max()))
    if receiver.size != COUNT or COUNT not in labels.shape or observed != EXPECTED_RECEIVER or np.unique(receiver).size != COUNT:
        raise ValueError("development-v10 identity contract failed")
    disjoint = {name: observed[1] < value[0] or observed[0] > value[1] for name, value in REFERENCES.items()}
    if not all(disjoint.values()):
        raise ValueError("development-v10 overlaps a prior role")
    return {
        "status": "completed", "evidence_role": "development_v10", "development_v10_path": str(path),
        "size_bytes": path.stat().st_size, "sha256": _sha(path), "sample_count": COUNT,
        "expected_sample_seed_range": [104000001, 104025200],
        "observed_receiver_noise_seed_range": list(observed), "receiver_noise_seed_unique": COUNT,
        "reference_receiver_noise_seed_ranges": {key: list(value) for key, value in REFERENCES.items()},
        "receiver_identity_disjoint": disjoint, "identity_audit_passed": True,
        "test_files_opened": False, "locked_test_used": False, "test_v3_used": False, "test_v10_used": False,
    }


def run(args):
    path = args.data_path if args.data_path is not None else _default_path()
    payload = audit(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v10_generation/identity_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

