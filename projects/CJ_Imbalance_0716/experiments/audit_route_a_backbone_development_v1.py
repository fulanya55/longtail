"""Audit and split the fresh Route-A backbone development source.

The script opens only the explicitly supplied train MAT.  Historical identity
checks are based on text/JSON contracts, with every test-v3-named path skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import (  # noqa: E402
    build_label_longtail,
    compute_weak_component_mask,
)
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.split_builder import build_stratified_roles  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
SEEDS = (2039, 2040, 2041, 2042, 2043)
SAMPLE_COUNT = 25200
SAMPLE_SEED_RANGE = (2550000001, 2550025200)
RECEIVER_RANGE = (2748677743, 2748702942)
LABEL_LT_TARGET_COUNTS = np.asarray(
    [1600, 400, 1213, 919, 696, 528], dtype=np.int64
)
ROLE_NAMES = ("fit", "validation", "early", "calibration", "threshold")
CONTRACT_ROOTS = (
    PROJECT_ROOT / "experiments",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "outputs" / "tapc",
    PROJECT_ROOT / "configs",
)
CONTRACT_SUFFIXES = {".py", ".json", ".md", ".yaml"}
RANGE_PATTERNS = (
    re.compile(
        r'"(?:observed_|expected_)?receiver_noise_seed_range"\s*:\s*'
        r'\[\s*(\d+)\s*,\s*(\d+)\s*\]'
    ),
    re.compile(r'"receiver_seed_range"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]'),
    re.compile(
        r'(?:EXPECTED_RECEIVER|RECEIVER_RANGE)\s*=\s*'
        r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    ),
    re.compile(r'`receiver_noise_seed` range `(\d+)--(\d+)`'),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ranges_disjoint(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return int(left[1]) < int(right[0]) or int(left[0]) > int(right[1])


def receiver_seed(sample_seed: int) -> int:
    return int((int(sample_seed) + 2 * 2246822519) % 4294967296)


def _safe_contract_files():
    for root in CONTRACT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            lowered = str(path).lower()
            if (
                not path.is_file()
                or path.suffix.lower() not in CONTRACT_SUFFIXES
                or "test-v3" in lowered
                or "test_v3" in lowered
            ):
                continue
            yield path


def historical_contract_ranges() -> list[dict]:
    rows = []
    for path in _safe_contract_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern_index, pattern in enumerate(RANGE_PATTERNS):
            for match in pattern.finditer(text):
                low, high = (int(value) for value in match.groups())
                if low > high:
                    continue
                rows.append(
                    {
                        "low": low,
                        "high": high,
                        "source": str(path.resolve()),
                        "pattern_index": pattern_index,
                    }
                )
    unique = {
        (row["low"], row["high"], row["source"], row["pattern_index"]): row
        for row in rows
    }
    return sorted(
        unique.values(), key=lambda row: (row["low"], row["high"], row["source"])
    )


def _seal_flags(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seal = payload.get("test_v3", {})
    flags = {
        key: seal.get(key)
        for key in ("access_authorized", "analysis_allowed", "used")
    }
    if flags != {
        "access_authorized": False,
        "analysis_allowed": False,
        "used": False,
    }:
        raise ValueError("test-v3 seal flags are not all false")
    return flags


def preflight(seal_manifest: Path) -> dict:
    expected = (
        receiver_seed(SAMPLE_SEED_RANGE[0]),
        receiver_seed(SAMPLE_SEED_RANGE[1]),
    )
    if expected != RECEIVER_RANGE:
        raise AssertionError("receiver seed derivation contract drifted")
    rows = [
        row
        for row in historical_contract_ranges()
        if (row["low"], row["high"]) != RECEIVER_RANGE
    ]
    overlaps = [
        row
        for row in rows
        if not ranges_disjoint(RECEIVER_RANGE, (row["low"], row["high"]))
    ]
    if overlaps:
        raise ValueError("candidate receiver range overlaps historical contracts")
    registry_payload = [
        {key: row[key] for key in ("low", "high", "source", "pattern_index")}
        for row in rows
    ]
    return {
        "status": "preflight_passed_no_train_mat_opened",
        "experiment": "RouteA_BackboneDevelopmentV1",
        "sample_seed_range": list(SAMPLE_SEED_RANGE),
        "receiver_noise_seed_range": list(RECEIVER_RANGE),
        "historical_contract_range_occurrences": len(rows),
        "historical_contract_registry_sha256": _json_sha(registry_payload),
        "identity_overlap_count": len(overlaps),
        "identity_audit_passed": not overlaps,
        "test_v3_seal": _seal_flags(seal_manifest),
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def _component_power(values79: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            values79[:, get_parameter_specs(name)[0].column_index]
            for name in CLASS_NAMES
        ],
        axis=1,
    )


def _receiver_values(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1)
    return receiver.astype(np.int64, copy=False)


def _role_hash(role_ids: dict[str, np.ndarray], receiver: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in ROLE_NAMES:
        digest.update(name.encode("ascii"))
        ids = np.asarray(role_ids[name], dtype="<i8")
        digest.update(ids.tobytes())
        digest.update(np.asarray(receiver[ids], dtype="<i8").tobytes())
    return digest.hexdigest()


def build_roles(
    labels: np.ndarray,
    weak: np.ndarray,
    receiver: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    split = build_stratified_roles(
        labels, np.arange(len(labels), dtype=np.int64), seed=int(seed)
    )
    manifest = build_label_longtail(
        labels,
        split.optimization,
        target_label_counts=LABEL_LT_TARGET_COUNTS,
        seed=int(seed),
        tolerance=0.0,
        source_split_sha256=split.sha256,
        target_sample_count=2400,
        sample_count_tolerance=0.0,
    )
    selected = manifest.selected_indices.copy()
    order = np.arange(len(selected), dtype=np.int64)
    np.random.RandomState(int(seed) + 31000).shuffle(order)
    fit_count = int(round(len(order) * 0.8))
    role_ids = {
        "fit": selected[order[:fit_count]],
        "validation": selected[order[fit_count:]],
        "early": split.early_stop.copy(),
        "calibration": split.calibration.copy(),
        "threshold": split.threshold_selection.copy(),
    }
    all_ids = np.concatenate([role_ids[name] for name in ROLE_NAMES])
    all_receiver = receiver[all_ids]
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError("sample indices overlap across roles")
    if len(np.unique(all_receiver)) != len(all_receiver):
        raise AssertionError("receiver identities overlap across roles")
    role_summary = {}
    for name in ROLE_NAMES:
        ids = role_ids[name]
        role_summary[name] = {
            "sample_count": int(len(ids)),
            "receiver_identity_count": int(np.unique(receiver[ids]).size),
            "label_positive_counts": labels[ids].sum(axis=0).astype(int).tolist(),
            "weak_positive_counts": weak[ids].sum(axis=0).astype(int).tolist(),
            "sample_indices_sha256": hashlib.sha256(
                np.asarray(ids, dtype="<i8").tobytes()
            ).hexdigest(),
            "receiver_identities_sha256": hashlib.sha256(
                np.asarray(receiver[ids], dtype="<i8").tobytes()
            ).hexdigest(),
        }
    summary = {
        "seed": int(seed),
        "split_sha256": split.sha256,
        "longtail_manifest_sha256": manifest.sha256,
        "longtail_label_counts": labels[selected].sum(axis=0).astype(int).tolist(),
        "longtail_weak_label_counts": weak[selected].sum(axis=0).astype(int).tolist(),
        "role_hash": _role_hash(role_ids, receiver),
        "roles": role_summary,
        "role_sample_overlap_count": 0,
        "role_receiver_overlap_count": 0,
    }
    return role_ids, summary


def audit_train(path: Path, output_root: Path, seal_manifest: Path) -> dict:
    train_path = Path(path).resolve()
    if "test" in train_path.name.lower():
        raise ValueError("refusing to open a test-named MAT")
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    base = preflight(seal_manifest)
    receiver = _receiver_values(train_path)
    labels = read_labels(train_path)
    values79, _ = read_parameters(train_path)
    if receiver.size != SAMPLE_COUNT or labels.shape != (SAMPLE_COUNT, 6):
        raise ValueError("backbone-development-v1 sample/label shape mismatch")
    observed = (int(receiver.min()), int(receiver.max()))
    if observed != RECEIVER_RANGE or np.unique(receiver).size != SAMPLE_COUNT:
        raise ValueError("backbone-development-v1 receiver identity contract failed")
    weak = compute_weak_component_mask(labels, _component_power(values79), 10.0)
    if np.any(weak & ~labels.astype(bool)):
        raise AssertionError("weak mask contains inactive labels")
    if np.any(weak.sum(axis=0) == 0):
        raise ValueError("at least one label has no weak positives")

    role_summaries = []
    role_root = output_root / "roles"
    for seed in SEEDS:
        role_ids, summary = build_roles(labels, weak, receiver, seed)
        seed_root = role_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(seed_root / "role_indices.npz", **role_ids)
        summary["role_indices_path"] = str(
            (seed_root / "role_indices.npz").resolve()
        )
        summary["role_indices_sha256"] = _sha256(
            seed_root / "role_indices.npz"
        )
        (seed_root / "role_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        role_summaries.append(summary)

    base.update(
        {
            "status": "completed_train_only_identity_role_audit",
            "data_path": str(train_path),
            "data_size_bytes": train_path.stat().st_size,
            "data_sha256": _sha256(train_path),
            "sample_count": SAMPLE_COUNT,
            "observed_receiver_noise_seed_range": list(observed),
            "receiver_noise_seed_unique": int(np.unique(receiver).size),
            "max_views_per_receiver": int(
                np.unique(receiver, return_counts=True)[1].max()
            ),
            "class_names": list(CLASS_NAMES),
            "full_label_positive_counts": labels.sum(axis=0).astype(int).tolist(),
            "full_weak_positive_counts": weak.sum(axis=0).astype(int).tolist(),
            "weak_threshold_db": 10.0,
            "training_seeds": list(SEEDS),
            "role_summaries": role_summaries,
            "test_files_opened": False,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
    )
    return base


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            r"E:\LJQ_code\干扰数据\route_a_backbone-development-v1_20260806"
        )
        / "closedset_6class_train.mat",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_backbone_development_v1",
    )
    parser.add_argument(
        "--seal-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "frozen"
        / "20260731_route_a_test_v3_sealed"
        / "test_v3_sealed_manifest.json",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def run(args):
    if args.preflight_only:
        payload = preflight(args.seal_manifest)
        output = args.output_root / "generation" / "identity_preflight.json"
    else:
        payload = audit_train(args.data_path, args.output_root, args.seal_manifest)
        output = args.output_root / "identity_role_audit.json"
    _write(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    run(parse_args())
