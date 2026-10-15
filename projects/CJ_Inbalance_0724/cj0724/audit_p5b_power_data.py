"""Audit paired P5b power views before any frozen-model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724 import DEFAULT_TRAIN_MAT
from cj0724.dual_axis import POWER_LEVELS, audit_power_alignment, discover_power_train_files, read_identity_metadata
from cj0724.multitask_benchmark import read_selected_parameters


def identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-root", required=True)
    parser.add_argument("--excluded", nargs="*", default=[])
    parser.add_argument("--include-default-root", action="store_true")
    parser.add_argument("--levels", nargs="+", type=int, default=list(POWER_LEVELS))
    parser.add_argument("--expected-samples", type=int, default=2520)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    power_root = Path(args.power_root)
    if (power_root / "power_sweep_manifest.csv").is_file():
        files = discover_power_train_files(power_root)
    else:
        files = {
            int(level): power_root
            / f"level_{index:02d}__jnr_{int(level)}dB"
            / "closedset_6class_train.mat"
            for index, level in enumerate(args.levels, start=1)
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing rechunked power files: {missing}")
    alignment = audit_power_alignment(files)
    reference_seeds, reference_labels = read_identity_metadata(files[30])
    indices = np.arange(reference_seeds.size, dtype=np.int64)
    reference_values, reference_mask = read_selected_parameters(files[30], indices)

    excluded_paths = [Path(value) for value in args.excluded]
    if args.include_default_root:
        excluded_paths.insert(0, Path(DEFAULT_TRAIN_MAT))
    identity_set = set(reference_seeds.tolist())
    overlap = []
    for path in excluded_paths:
        resolved = path if path.is_absolute() else project / path
        prior, _ = read_identity_metadata(resolved)
        overlap.append({
            "path": str(resolved),
            "identity_count": int(prior.size),
            "overlap_count": int(len(identity_set.intersection(prior.tolist()))),
        })

    level_contract = []
    non_jnr_equal = True
    masks_equal = True
    jnr_contract = True
    for level in args.levels:
        values, mask = read_selected_parameters(files[level], indices)
        masks_equal = masks_equal and bool(np.array_equal(mask, reference_mask))
        active_non_jnr = mask[:, :, 1]
        current_non_jnr_equal = bool(np.allclose(
            values[:, :, 1][active_non_jnr],
            reference_values[:, :, 1][active_non_jnr],
            rtol=0.0,
            atol=1.0e-9,
        ))
        non_jnr_equal = non_jnr_equal and current_non_jnr_equal
        expected = np.asarray([30.0, 30.0, 30.0, level, level, level], dtype=np.float32)
        active_jnr = mask[:, :, 0] & reference_labels.astype(bool)
        errors = np.abs(values[:, :, 0] - expected[None, :])
        maximum_jnr_error = float(errors[active_jnr].max())
        current_jnr_contract = maximum_jnr_error <= 1.0e-4
        jnr_contract = jnr_contract and current_jnr_contract
        level_contract.append({
            "power_db": int(level),
            "non_jnr_active_parameters_equal_to_30db_view": current_non_jnr_equal,
            "maximum_active_jnr_contract_error_db": maximum_jnr_error,
            "jnr_contract_passed": current_jnr_contract,
        })

    result = {
        "power_root": str(Path(args.power_root).resolve()),
        "alignment": alignment,
        "identity_sha256": identity_hash(reference_seeds),
        "identity_minimum": int(reference_seeds.min()),
        "identity_maximum": int(reference_seeds.max()),
        "excluded_overlap": overlap,
        "all_excluded_overlap_zero": bool(all(row["overlap_count"] == 0 for row in overlap)),
        "parameter_contract": {
            "masks_identical_across_powers": masks_equal,
            "non_jnr_active_parameters_identical_across_powers": non_jnr_equal,
            "jnr_targets_follow_fixed_first3_swept_last3_contract": jnr_contract,
            "levels": level_contract,
        },
        "valid_for_frozen_inference": bool(
            alignment["samples"] == args.expected_samples
            and alignment["unique_identities"] == args.expected_samples
            and alignment["combinations"] == 63
            and all(row["overlap_count"] == 0 for row in overlap)
            and masks_equal and non_jnr_equal and jnr_contract
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
