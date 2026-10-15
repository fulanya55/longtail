"""Create identity-safe source-v1 role manifests and audit estimability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_vft_isrj_raw_source_diagnostic import _hard_source_view  # noqa: E402


COUNT = 5040
RECEIVER_RANGE = (352677743, 352682782)
SEEDS = (2039, 2040, 2041, 2042, 2043)
ROLE_COUNTS = {"fit": 48, "validation": 16, "early": 16}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combination_code(labels: np.ndarray) -> np.ndarray:
    return (np.asarray(labels, dtype=np.uint8) * (1 << np.arange(6, dtype=np.uint8))[None, :]).sum(axis=1).astype(np.int16)


def _roles_for_seed(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    codes = _combination_code(labels)
    roles = {role: [] for role in ROLE_COUNTS}
    for code in np.unique(codes):
        values = np.flatnonzero(codes == code)
        if len(values) != sum(ROLE_COUNTS.values()):
            raise ValueError(f"combination {int(code)} has {len(values)} samples, expected 80")
        ordered = np.random.default_rng(65010000 + int(seed) * 100 + int(code)).permutation(values)
        cursor = 0
        for role, count in ROLE_COUNTS.items():
            roles[role].append(ordered[cursor:cursor + count])
            cursor += count
    return {role: np.sort(np.concatenate(values).astype(np.int64)) for role, values in roles.items()}


def run(args):
    path = Path(args.data_path)
    if "test" in path.name.lower():
        raise ValueError("source-v1 audit refuses a test-named file")
    labels = read_labels(path); parameters, _ = read_parameters(path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    with h5py.File(path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    if len(labels) != COUNT or len(receiver) != COUNT or np.unique(receiver).size != COUNT:
        raise ValueError("source-v1 count/identity uniqueness failed")
    expected_range = tuple(args.expected_receiver_range)
    if (int(receiver.min()), int(receiver.max())) != expected_range:
        raise ValueError("source-v1 receiver range failed")
    codes = _combination_code(labels)
    code_counts = {str(int(code)): int((codes == code).sum()) for code in np.unique(codes)}
    if len(code_counts) != 63 or set(code_counts.values()) != {80}:
        raise ValueError("source-v1 combination contract failed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for seed in args.seeds:
        roles = _roles_for_seed(labels, seed)
        all_ids = np.concatenate(list(roles.values()))
        if len(np.unique(all_ids)) != COUNT or len(all_ids) != COUNT:
            raise ValueError("role coverage/disjointness failed")
        root = args.output_root / f"seed_{seed}"; root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "role_indices.npz", **roles)
        metrics = {}
        for role, indices in roles.items():
            _selected, _labels, receipt = _hard_source_view(labels[indices], weak[indices])
            metrics[role] = {**receipt, "receiver_count": int(np.unique(receiver[indices]).size)}
        _seed_payload = {"seed": int(seed), "roles": metrics, "role_counts": {name: int(len(value)) for name, value in roles.items()}, "locked_test_used": False, "test_v3_used": False}
        (root / "seed_audit.json").write_text(json.dumps(_seed_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        seed_rows.append(_seed_payload)
    checks = {
        "train_only_filename": "test" not in path.name.lower(),
        "sample_count": len(labels) == COUNT,
        "receiver_identity_unique": np.unique(receiver).size == COUNT,
        "receiver_range": (int(receiver.min()), int(receiver.max())) == expected_range,
        "all_63_patterns_have_80": len(code_counts) == 63 and set(code_counts.values()) == {80},
        "roles_cover_each_sample_once": all(sum(row["role_counts"].values()) == COUNT for row in seed_rows),
        "every_role_source_view_estimable": all(all(item["weak_vft_positive_count"] > 0 and item["vft_absent_isrj_present_negative_count"] > 0 for item in row["roles"].values()) for row in seed_rows),
    }
    payload = {
        "status": "completed", "experiment": "RouteA_VFTISRJ_SourceV1_DataAudit", "evidence_role": "fresh_source_identifiability_diagnostic",
        "data_path": str(path), "data_sha256": _sha256(path), "sample_count": COUNT, "receiver_noise_seed_range": list(expected_range),
        "combination_counts": code_counts, "seeds": seed_rows, "checks": checks, "passed": all(checks.values()),
        "weak_threshold_db": float(args.weak_threshold_db), "test_files_opened": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    output = args.output_root / "source_v1_identity_role_audit.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "checks": checks, "output": str(output)}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_source_v1\closedset_6class_train.mat"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_source_v1_generation")
    parser.add_argument("--expected-receiver-range", type=int, nargs=2, default=list(RECEIVER_RANGE))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
