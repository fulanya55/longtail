"""Audit development-v3 train-only power views without opening sealed test-v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256  # noqa: E402


def _array_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def run(args):
    if _sha256(args.base_path).lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 base SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    test_record = seal["test_v3"]
    if test_record["access_authorized"] or test_record["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    base_identity, base_labels = load_level_identity(args.base_path)
    files = discover_power_sweep_train_files(args.sweep_root)
    expected_levels = tuple(range(10, 51, 5))
    if tuple(files) != expected_levels:
        raise ValueError(f"expected nine levels {expected_levels}, got {tuple(files)}")
    if len(np.unique(base_identity)) != len(base_identity):
        raise AssertionError("base development-v3 identities are not unique")

    level_rows = []
    for level, path in files.items():
        identity, labels = load_level_identity(path)
        identity_equal = np.array_equal(identity, base_identity)
        labels_equal = np.array_equal(labels, base_labels)
        if not identity_equal or not labels_equal:
            raise AssertionError(f"level {level} does not align with development-v3 base")
        level_rows.append({
            "level_db": int(level), "path": str(path), "bytes": int(path.stat().st_size),
            "sha256": _sha256(path), "sample_count": int(len(identity)),
            "unique_identity_count": int(len(np.unique(identity))),
            "identity_sha256": _array_sha(identity, "<i8"),
            "labels_sha256": _array_sha(labels, np.uint8),
            "identity_matches_base": bool(identity_equal),
            "labels_match_base": bool(labels_equal),
        })

    role_audits = {}
    for seed in args.seeds:
        saved = np.load(args.c6_root/f"seed_{seed}"/"role_indices.npz")
        names = ("fit", "validation", "early", "calibration", "threshold")
        role_ids = {name: base_identity[saved[name].astype(np.int64)] for name in names}
        concatenated = np.concatenate([role_ids[name] for name in names])
        if len(np.unique(concatenated)) != len(concatenated):
            raise AssertionError(f"role identity leakage for seed {seed}")
        role_audits[str(seed)] = {
            "role_counts": {name: int(len(role_ids[name])) for name in names},
            "all_roles_unique": True,
            "combined_identity_sha256": _array_sha(concatenated, "<i8"),
            "role_identity_sha256": {name: _array_sha(role_ids[name], "<i8") for name in names},
        }

    checks = {
        "nine_levels_present": len(level_rows) == 9,
        "all_levels_have_25200_samples": all(row["sample_count"] == 25200 for row in level_rows),
        "all_level_identities_unique": all(row["unique_identity_count"] == 25200 for row in level_rows),
        "all_identities_match_base": all(row["identity_matches_base"] for row in level_rows),
        "all_labels_match_base": all(row["labels_match_base"] for row in level_rows),
        "development_roles_identity_disjoint": all(row["all_roles_unique"] for row in role_audits.values()),
        "sealed_manifest_records_base_test_intersection_zero":
            seal.get("identity_audit", {}).get("cross_role_receiver_id_intersection_count") == 0
            and seal.get("identity_audit", {}).get("cross_role_identity_disjoint") is True,
        "test_v3_unauthorized_and_unused": not test_record["access_authorized"] and not test_record["used"],
    }
    payload = {
        "status": "completed", "experiment": "RouteA_V3_PowerSweep_IdentityAudit",
        "base_path": str(args.base_path), "base_sha256": _sha256(args.base_path),
        "base_sample_count": int(len(base_identity)),
        "base_identity_sha256": _array_sha(base_identity, "<i8"),
        "base_labels_sha256": _array_sha(base_labels, np.uint8),
        "sweep_root": str(args.sweep_root), "levels": level_rows,
        "role_audits": role_audits, "checks": checks, "passed": bool(all(checks.values())),
        "test_v3_source": "sealed manifest only; test MAT not opened",
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root/"route_a_v3_power_sweep_identity_audit.json", payload)
    with (args.output_root/"route_a_v3_power_sweep_file_inventory.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(level_rows[0]))
        writer.writeheader(); writer.writerows(level_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat"))
    parser.add_argument("--sweep-root", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\power_sweep_9level_v3"))
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_v3_power_sweep_audit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043])
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
