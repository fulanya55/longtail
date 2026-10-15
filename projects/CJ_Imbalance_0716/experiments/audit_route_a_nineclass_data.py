"""Audit the E-drive nine-class power-sweep contract for Route-A experiments.

This audit intentionally reads only HDF5 metadata, labels and receiver identity
vectors. It never reads model inputs for training/evaluation and never opens a
locked test.  The resulting manifest resolves all paths to E: and records the
file hash when requested.  Existing G-drive paths in the generation manifest
are treated as historical text, not as live paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CLASS_NAMES = (
    "FMB",
    "AMA",
    "SawFM",
    "SinFM",
    "SN",
    "MFT",
    "VFT",
    "ISRJ",
    "RangeFakeTarget",
)
POWER_LEVELS = (10, 15, 20, 25, 30, 35, 40, 45, 50)
REQUIRED_KEYS = (
    "iq_complex",
    "known_label",
    "label_all",
    "label_selected",
    "receiver_noise_seed",
    "parameter_mask",
    "parameter_values",
)


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def decode_char_dataset(handle: h5py.File, reference: Any) -> str:
    try:
        values = np.asarray(handle[reference]).reshape(-1)
        return "".join(chr(int(value)) for value in values if int(value) > 0)
    except Exception:
        return ""


def registry_codes(handle: h5py.File) -> list[str]:
    group = handle.get("registry_snapshot")
    if group is None or "code" not in group:
        return []
    refs = np.asarray(group["code"]).reshape(-1)
    return [decode_char_dataset(handle, ref) for ref in refs]


def audit_file(path: Path, hash_files: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "sha256": None,
        "errors": [],
    }
    if not path.is_file():
        record["errors"].append("missing_file")
        return record

    try:
        with h5py.File(path, "r") as handle:
            record["keys"] = sorted(str(key) for key in handle.keys())
            missing = [key for key in REQUIRED_KEYS if key not in handle]
            if missing:
                record["errors"].append({"missing_keys": missing})
                return record

            iq = handle["iq_complex"]
            labels = np.asarray(handle["known_label"][:, :], dtype=np.uint8)
            labels_all = np.asarray(handle["label_all"][:, :], dtype=np.uint8)
            labels_selected = np.asarray(handle["label_selected"][:, :], dtype=np.uint8)
            receiver = np.asarray(handle["receiver_noise_seed"][:, :], dtype=np.float64)
            parameter_mask = handle["parameter_mask"]
            parameter_values = handle["parameter_values"]

            record["shapes"] = {
                "iq_complex": [int(value) for value in iq.shape],
                "known_label": [int(value) for value in labels.shape],
                "label_all": [int(value) for value in labels_all.shape],
                "label_selected": [int(value) for value in labels_selected.shape],
                "receiver_noise_seed": [int(value) for value in receiver.shape],
                "parameter_mask": [int(value) for value in parameter_mask.shape],
                "parameter_values": [int(value) for value in parameter_values.shape],
            }
            record["class_names"] = list(CLASS_NAMES)
            record["registry_codes"] = registry_codes(handle)
            record["label_marginal_counts"] = [int(value) for value in labels.sum(axis=1)]
            record["label_all_marginal_counts"] = [int(value) for value in labels_all.sum(axis=1)]
            record["label_selected_marginal_counts"] = [int(value) for value in labels_selected.sum(axis=1)]
            sample_major = labels.T
            patterns = np.unique(sample_major, axis=0)
            record["unique_known_patterns"] = int(patterns.shape[0])
            record["unique_projected_six_patterns_min"] = int(
                min(
                    np.unique(sample_major[:, subset], axis=0).shape[0]
                    for subset in _six_subsets()
                )
            )
            record["receiver_unique_count"] = int(np.unique(receiver.reshape(-1)).size)
            record["receiver_first_last"] = [
                float(receiver.reshape(-1)[0]),
                float(receiver.reshape(-1)[-1]),
            ]
            if labels.shape != (9, 186000):
                record["errors"].append({"known_label_shape": list(labels.shape)})
            if labels_all.shape[0] != 10 or labels_all.shape[1] != labels.shape[1]:
                record["errors"].append({"label_all_shape": list(labels_all.shape)})
            if labels_all.shape[0] == 10 and int(labels_all[-1].sum()) != 0:
                record["errors"].append("label_all_tenth_row_not_zero")
            if labels_selected.shape != labels.shape or not np.array_equal(labels_selected, labels):
                record["errors"].append("label_selected_not_equal_known_label")
            if receiver.shape != (1, labels.shape[1]):
                record["errors"].append({"receiver_shape": list(receiver.shape)})
    except Exception as exc:  # pragma: no cover - audit receipt must retain the error
        record["errors"].append({"exception": repr(exc)})

    if hash_files and not record["errors"]:
        record["sha256"] = sha256_file(path)
    return record


def _six_subsets() -> list[tuple[int, ...]]:
    import itertools

    return list(itertools.combinations(range(9), 6))


def build_manifest(data_root: Path, output_root: Path, hash_files: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for power in POWER_LEVELS:
        level_dir = data_root / "power_sweep_9level_v1" / f"level_{POWER_LEVELS.index(power)+1:02d}__jnr_{power}dB"
        paths = {
            "train": level_dir / "closedset_9class_train.mat",
            "test": level_dir / "closedset_9class_test.mat",
        }
        for role, path in paths.items():
            record = audit_file(path, hash_files=hash_files)
            record.update({"power_db": power, "role": role})
            audit_records.append(record)
            rows.append(
                {
                    "power_db": power,
                    "role": role,
                    "path": str(path.resolve()),
                    "size_bytes": record.get("size_bytes"),
                    "sha256": record.get("sha256"),
                    "sample_count": (record.get("shapes") or {}).get("known_label", [None, None])[-1],
                    "known_label_rows": (record.get("shapes") or {}).get("known_label", [None, None])[0],
                    "unique_known_patterns": record.get("unique_known_patterns"),
                    "receiver_unique_count": record.get("receiver_unique_count"),
                    "errors": json.dumps(record.get("errors", []), ensure_ascii=False),
                }
            )

    train_records = [record for record in audit_records if record["role"] == "train"]
    test_records = [record for record in audit_records if record["role"] == "test"]
    pair_errors: list[str] = []
    for train, test in zip(train_records, test_records):
        if train.get("errors") or test.get("errors"):
            continue
        if train.get("shapes") != test.get("shapes"):
            pair_errors.append(f"shape_mismatch_{train['power_db']}dB")

    report = {
        "schema": "route-a-nineclass-data-audit-v1",
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "class_names": list(CLASS_NAMES),
        "power_levels_db": list(POWER_LEVELS),
        "files_checked": len(audit_records),
        "hash_files": bool(hash_files),
        "locked_test_used": False,
        "old_manifest_paths_are_historical_only": True,
        "identity_key": "receiver_noise_seed",
        "pair_errors": pair_errors,
        "records": audit_records,
        "status": "PASS" if not any(record.get("errors") for record in audit_records) and not pair_errors else "FAIL",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "nineclass_data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / "nineclass_e_drive_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hash-files", action="store_true")
    args = parser.parse_args()
    report = build_manifest(args.data_root, args.output_root, args.hash_files)
    print(json.dumps({"status": report["status"], "files_checked": report["files_checked"], "output_root": report["output_root"], "locked_test_used": False}, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
