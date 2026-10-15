"""Seal test-v3 without exposing it to model selection or metric analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_receiver_ids(path: Path) -> np.ndarray:
    """Read only the identity field required by the leakage audit."""
    with h5py.File(path, "r") as handle:
        if "receiver_noise_seed" not in handle:
            raise KeyError(f"receiver_noise_seed is missing from {path}")
        values = np.asarray(handle["receiver_noise_seed"]).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"invalid receiver_noise_seed values in {path}")
    return values.astype(np.uint64, copy=False)


def identity_audit(development_path: Path, test_path: Path) -> dict:
    development_ids = read_receiver_ids(development_path)
    test_ids = read_receiver_ids(test_path)
    development_unique = np.unique(development_ids)
    test_unique = np.unique(test_ids)
    intersection = np.intersect1d(development_unique, test_unique)
    result = {
        "development_sample_count": int(development_ids.size),
        "development_unique_receiver_ids": int(development_unique.size),
        "test_sample_count": int(test_ids.size),
        "test_unique_receiver_ids": int(test_unique.size),
        "cross_role_receiver_id_intersection_count": int(intersection.size),
        "development_identity_unique": bool(development_unique.size == development_ids.size),
        "test_identity_unique": bool(test_unique.size == test_ids.size),
        "cross_role_identity_disjoint": bool(intersection.size == 0),
    }
    result["passed"] = bool(
        result["development_identity_unique"]
        and result["test_identity_unique"]
        and result["cross_role_identity_disjoint"]
    )
    return result


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def run(args: argparse.Namespace) -> dict:
    development_path = args.development_path.resolve()
    test_path = args.test_path.resolve()
    if not development_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("development-v3 or test-v3 file is missing")
    if args.sealed_root.exists():
        raise FileExistsError(f"sealed root already exists: {args.sealed_root}")

    audit = identity_audit(development_path, test_path)
    if not audit["passed"]:
        raise ValueError(f"receiver identity audit failed: {audit}")

    test_hash = sha256(test_path)
    development_hash = sha256(development_path)
    protocol_files = [
        PROJECT_ROOT / "experiments" / "generate_route_a_v3_dataset.m",
        PROJECT_ROOT / "experiments" / "seal_route_a_test_v3.py",
        PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml",
        PROJECT_ROOT / "docs" / "plans" / "route_a_development_v3_test_v3_protocol_20260731.md",
    ]
    file_records = []
    for path in protocol_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        file_records.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "sealed",
        "created_at_utc": created_at,
        "protocol": "RouteA development-v3/test-v3 identity-isolated protocol",
        "development_v3": {
            "path": str(development_path),
            "sha256": development_hash,
            "role": "development_only",
        },
        "test_v3": {
            "path": str(test_path),
            "bytes": test_path.stat().st_size,
            "sha256": test_hash,
            "role": "sealed_independent_test_once_after_final_freeze",
            "analysis_allowed": False,
            "access_authorized": False,
            "used": False,
            "rerun_allowed_after_execution": False,
        },
        "identity_audit": audit,
        "historic_locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "protocol_files": file_records,
    }
    write_json_exclusive(args.sealed_root / "test_v3_sealed_manifest.json", manifest)
    write_json_exclusive(args.sealed_root / "TEST_V3_DO_NOT_OPEN.json", {
        "sealed_at_utc": created_at,
        "test_v3_sha256": test_hash,
        "access_authorized": False,
        "authorized_only_after_all_development_gates": True,
        "required_started_guard": "test_v3_execution_started.json",
        "failure_still_consumes_test": True,
    })
    os.chmod(test_path, stat.S_IREAD)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-path",
        type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat"),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_test.mat"),
    )
    parser.add_argument(
        "--sealed-root",
        type=Path,
        default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
