"""Audit the fresh H1t train-only source and create identity-safe roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402


LEVELS = (10, 30, 50)
TARGET_COLUMNS = (3, 4, 5)
TARGET_JNR_COLUMNS = (27, 48, 52)
SAMPLE_COUNT = 8064
# The official generator derives the receiver seed as
# mod(sample_seed + 2*2246822519, 2^32), so it is not the sample seed itself.
RECEIVER_OFFSET = 2 * 2246822519
SEEDS = (2101, 2102, 2103, 2104, 2105)
DEFAULT_ROOT = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h1t_asymmetric_power_v1")
DEFAULT_SEAL = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/tapc/route_a_h1t_source_audit"


def _view_path(root: Path, level: int) -> Path:
    index = {10: 1, 30: 5, 50: 9}[int(level)]
    return root / f"level_{index:02d}__jnr_{level}dB" / "closedset_6class_train.mat"


def _read_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        identities = np.asarray(handle["receiver_noise_seed"][:, :]).reshape(-1)
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        values = np.asarray(handle["parameter_values"][:, :], dtype=np.float64).T
        masks = np.asarray(handle["parameter_mask"][:, :], dtype=np.uint8).T.astype(bool)
    identities = np.rint(identities).astype(np.int64)
    return identities, labels, values, masks


def make_roles(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Composition-stratified 64/16/48 split for every 128-example pattern."""

    labels = np.asarray(labels, dtype=np.uint8)
    rng = np.random.RandomState(int(seed))
    fit, validation, early = [], [], []
    keys = np.packbits(labels, axis=1, bitorder="little").reshape(-1)
    for key in sorted(np.unique(keys).tolist()):
        positions = np.flatnonzero(keys == key)
        if len(positions) != 128:
            raise ValueError(f"composition {int(key)} has {len(positions)} rather than 128 rows")
        values = positions.copy(); rng.shuffle(values)
        fit.append(values[:64]); validation.append(values[64:80]); early.append(values[80:])
    result = {
        "fit": np.sort(np.concatenate(fit)).astype(np.int64),
        "validation": np.sort(np.concatenate(validation)).astype(np.int64),
        "early": np.sort(np.concatenate(early)).astype(np.int64),
    }
    joined = np.concatenate(list(result.values()))
    if len(joined) != len(labels) or len(np.unique(joined)) != len(labels):
        raise AssertionError("H1t roles are not a disjoint partition")
    return result


def _check_seal(path: Path) -> dict:
    seal = json.loads(path.read_text(encoding="utf-8-sig"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    return {key: seal[key] for key in ("access_authorized", "analysis_allowed", "used")}


def run(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite source audit: {args.output_root}")
    seal = _check_seal(args.test_v3_manifest)
    paths = {level: _view_path(args.source_root, level) for level in LEVELS}
    if any(not path.is_file() or "test" in str(path).lower() for path in paths.values()):
        raise FileNotFoundError("H1t needs exactly three completed train-only views")
    metadata = {level: _read_metadata(path) for level, path in paths.items()}
    identity, labels, values30, masks30 = metadata[30]
    expected_sample = (int(args.base_seed) + 1, int(args.base_seed) + SAMPLE_COUNT)
    expected_receiver = tuple(int((value + RECEIVER_OFFSET) % 4294967296) for value in expected_sample)
    if not (len(identity) == SAMPLE_COUNT and np.array_equal(identity, np.arange(expected_receiver[0], expected_receiver[1] + 1))):
        raise AssertionError("fresh H1t receiver identity range mismatch")
    if len(np.unique(identity)) != len(identity):
        raise AssertionError("receiver identities are not unique")
    source_rows = []
    metadata_matching = {}
    for level, path in paths.items():
        current_ids, current_labels, current_values, current_masks = metadata[level]
        if not np.array_equal(current_ids, identity) or not np.array_equal(current_labels, labels):
            raise AssertionError(f"level {level} breaks paired identity/label contract")
        if not np.array_equal(current_masks, masks30):
            raise AssertionError(f"level {level} changes active parameter mask")
        compare_columns = np.ones(values30.shape[1], dtype=bool)
        compare_columns[list(TARGET_JNR_COLUMNS)] = False
        active = masks30[:, compare_columns]
        if not np.allclose(current_values[:, compare_columns][active], values30[:, compare_columns][active], rtol=0.0, atol=0.0):
            raise AssertionError(f"level {level} changes non-JNR parameter truth")
        jnr_contract = {}
        for label_column, parameter_column in zip(TARGET_COLUMNS, TARGET_JNR_COLUMNS):
            selected = current_labels[:, label_column].astype(bool)
            valid = current_masks[:, parameter_column]
            if not np.all(valid[selected]) or not np.allclose(current_values[selected, parameter_column], level, rtol=0.0, atol=1e-9):
                raise AssertionError(f"level {level} target JNR contract fails at label {label_column}")
            jnr_contract[str(label_column)] = float(level)
        source_rows.append({"level_db": level, "path": str(path), "sha256": _sha256(path),
                            "bytes": int(path.stat().st_size), "target_jnr_db": jnr_contract})
        metadata_matching[str(level)] = True
    args.output_root.mkdir(parents=True)
    role_rows = {}
    role_root = args.output_root / "roles"
    role_root.mkdir()
    for seed in args.seeds:
        roles = make_roles(labels, seed)
        target_counts = {role: labels[indices][:, list(TARGET_COLUMNS)].sum(axis=0).astype(int).tolist()
                         for role, indices in roles.items()}
        path = role_root / f"seed_{seed}_roles.npz"
        np.savez_compressed(path, **roles)
        role_rows[str(seed)] = {"path": str(path), "sha256": _sha256(path),
                                "counts": {name: int(len(value)) for name, value in roles.items()},
                                "target_positive_counts": target_counts}
    result = {
        "status": "completed_fresh_source_contract_audit", "experiment": "RouteA_H1t_SourceAudit",
        "evidence_role": "fresh_identity_train_only_h1_source", "source_root": str(args.source_root),
        "views": source_rows, "sample_seed_range": list(expected_sample),
        "receiver_noise_seed_range": list(expected_receiver), "identity_count": int(len(identity)),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(), "roles": role_rows,
        "checks": {"three_train_views_present": True, "identity_paired_across_views": True,
                   "labels_paired_across_views": True, "non_jnr_parameters_exactly_matched": True,
                   "target_jnr_contract": True, "no_test_view_generated": True, "test_v3_seal_intact": True},
        "test_v3_seal": seal, "iq_opened": False, "training_performed": False,
        "calibration_performed": False, "threshold_selection_performed": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(args.output_root / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-seed", type=int, default=160000000)
    parser.add_argument("--test-v3-manifest", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
