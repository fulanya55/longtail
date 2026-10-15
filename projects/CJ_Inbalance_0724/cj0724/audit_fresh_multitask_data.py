"""Audit identities, combinations, masks and target variability before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724 import DEFAULT_TRAIN_MAT
from cj0724.dual_axis import read_identity_metadata
from cj0724.multitask import PARAMETER_SPECS
from cj0724.multitask_benchmark import _combination_ids, audit_alignment, read_selected_parameters


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--excluded", nargs="*", default=[])
    parser.add_argument(
        "--exclude-root", action="append", default=[],
        help="Recursively audit every closedset_6class_train.mat below this root.",
    )
    parser.add_argument(
        "--skip-path-containing", action="append", default=[],
        help="Substring filter for exclude-root discovery (for the candidate's own raw/rechunked pool).",
    )
    parser.add_argument("--include-default-root", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project / data_path
    identities, labels = read_identity_metadata(data_path)
    all_indices = np.arange(len(labels), dtype=np.int64)
    combination_ids = _combination_ids(labels)
    values, mask = read_selected_parameters(data_path, all_indices)
    valid = mask & labels[:, :, None].astype(bool)

    excluded_paths = [Path(path) for path in args.excluded]
    skip_terms = tuple(str(value).lower() for value in args.skip_path_containing)
    for root in args.exclude_root:
        resolved_root = Path(root) if Path(root).is_absolute() else project / root
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"exclude root does not exist: {resolved_root}")
        excluded_paths.extend(
            path for path in resolved_root.rglob("closedset_6class_train.mat")
            if not any(term in str(path).lower() for term in skip_terms)
        )
    if args.include_default_root:
        excluded_paths.insert(0, Path(DEFAULT_TRAIN_MAT))
    overlap = []
    # A root scan may encounter a symbolic duplicate or a user-supplied path
    # more than once.  Deduplicate before audit so every historical pool has a
    # single receipt row and the audit remains machine-readable.
    unique_paths = []
    seen_paths = set()
    for path in excluded_paths:
        resolved = path if path.is_absolute() else project / path
        key = str(resolved.resolve()).lower()
        if key not in seen_paths:
            unique_paths.append(resolved)
            seen_paths.add(key)
    identity_set = set(identities.tolist())
    for resolved in unique_paths:
        prior, _ = read_identity_metadata(resolved)
        overlap.append({
            "path": str(resolved),
            "identity_count": int(len(prior)),
            "overlap_count": int(len(identity_set.intersection(prior.tolist()))),
        })

    target_rows = []
    for flat_index, spec in enumerate(PARAMETER_SPECS):
        class_index, parameter_index = divmod(flat_index, 2)
        selected = values[valid[:, class_index, parameter_index], class_index, parameter_index]
        target_rows.append({
            "class_name": spec.class_name,
            "parameter": spec.name,
            "count": int(selected.size),
            "finite": bool(np.isfinite(selected).all()),
            "minimum": float(np.min(selected)),
            "maximum": float(np.max(selected)),
            "range": float(np.ptp(selected)),
        })

    combo_values, combo_counts = np.unique(combination_ids, return_counts=True)
    result = {
        "data": str(data_path),
        "identities": {
            "count": int(len(identities)),
            "unique": int(len(np.unique(identities))),
            "minimum": int(identities.min()),
            "maximum": int(identities.max()),
            "sha256": array_sha256(identities.astype(np.int64)),
        },
        "combinations": {
            "count": int(len(combo_values)),
            "minimum_samples": int(combo_counts.min()),
            "maximum_samples": int(combo_counts.max()),
            "all_counts": {str(int(key)): int(value) for key, value in zip(combo_values, combo_counts)},
        },
        "excluded_overlap": overlap,
        "all_excluded_overlap_zero": bool(all(row["overlap_count"] == 0 for row in overlap)),
        "alignment": audit_alignment(data_path, labels, all_indices, np.empty(0, dtype=np.int64)),
        "targets": target_rows,
        "all_targets_finite_and_variable": bool(
            all(row["finite"] and row["count"] > 0 and row["range"] > 0 for row in target_rows)
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = project / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
