"""Identity contracts for TRB-PTV development-v14; never opens test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNT = 25200
SAMPLE_RANGE = (144000001, 144025200)
RECEIVER_RANGE = (342677743, 342702942)

# Includes all documented train and reserved-test identities through v13.  v14
# preflight reads these contracts only; it never opens a test file.
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
    "development_v10_train": (302677743, 302702942), "test_v10_contract": (305677743, 305702942),
    "development_v11_train": (312677743, 312702942), "test_v11_contract": (315677743, 315702942),
    "development_v12_train": (322677743, 322702942), "test_v12_contract": (325677743, 325702942),
    "development_v13_reserved_train": (332677743, 332702942), "test_v13_reserved_contract": (335677743, 335702942),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disjoint(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[1] < right[0] or left[0] > right[1]


def preflight() -> dict[str, Any]:
    receiver_disjoint = {name: disjoint(RECEIVER_RANGE, bounds) for name, bounds in REFERENCES.items()}
    if not all(receiver_disjoint.values()):
        raise ValueError("development-v14 receiver identity overlaps a registered role")
    return {
        "status": "preflight_passed_no_data_opened",
        "evidence_role": "development_v14",
        "sample_seed_range": list(SAMPLE_RANGE),
        "receiver_noise_seed_range": list(RECEIVER_RANGE),
        "expected_samples": COUNT,
        "reference_receiver_noise_seed_ranges": {name: list(bounds) for name, bounds in REFERENCES.items()},
        "receiver_identity_disjoint": receiver_disjoint,
        "identity_audit_passed": True,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v14_used": False,
    }


def audit_train(path: Path) -> dict[str, Any]:
    """Audit the generated train MAT only; callers must not pass a test path."""
    train_path = Path(path).resolve()
    if "test" in train_path.name.lower():
        raise ValueError("refusing to open a test-named file")
    base = preflight()
    with h5py.File(train_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1)
        labels = np.asarray(handle["label_selected"])
    observed = (int(receiver.min()), int(receiver.max()))
    if receiver.size != COUNT or COUNT not in labels.shape or observed != RECEIVER_RANGE or np.unique(receiver).size != COUNT:
        raise ValueError("development-v14 train identity contract failed")
    base.update({
        "status": "completed_train_only_identity_audit",
        "train_path": str(train_path),
        "size_bytes": train_path.stat().st_size,
        "sha256": _sha256(train_path),
        "sample_count": COUNT,
        "observed_receiver_noise_seed_range": list(observed),
        "receiver_noise_seed_unique": COUNT,
    })
    return base


def write_preflight(output: Path) -> dict[str, Any]:
    payload = preflight()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload
