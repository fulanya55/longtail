"""Audit a P5g paired-power pool against every locally retained train identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724 import DEFAULT_TRAIN_MAT
from cj0724.audit_p5b_power_data import identity_hash
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.multitask_benchmark import read_selected_parameters


POWERS = (10, 15, 20, 25, 30)


def _files(root: Path) -> dict[int, Path]:
    return {
        power: root / f"level_{index:02d}__jnr_{power}dB" / "closedset_6class_train.mat"
        for index, power in enumerate(POWERS, start=1)
    }


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = (project / args.power_root).resolve()
    # The raw files and any local rechunked representation bearing this prefix
    # are the *same new identity pool*, not historical evidence.  Exclude all
    # of those storage representations while retaining every other experiment.
    pool_prefix = root.name.split("_rechunked", maxsplit=1)[0]
    files = _files(root)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P5g paired power files are missing: {missing}")
    alignment = audit_power_alignment(files)
    reference_seeds, reference_labels = read_identity_metadata(files[30])
    indices = np.arange(reference_seeds.size, dtype=np.int64)
    reference_values, reference_mask = read_selected_parameters(files[30], indices)

    # Retain every local historical train file in the check.  Raw and rechunked
    # duplicates are intentionally both inspected: duplicate prior identities
    # cannot conceal an overlap with this newly generated pool.
    candidate_paths = [Path(DEFAULT_TRAIN_MAT)]
    candidate_paths.extend(project.joinpath("generated_data").rglob("closedset_6class_train.mat"))
    historical = []
    identity_set = set(reference_seeds.tolist())
    for path in sorted({item.resolve() for item in candidate_paths}):
        if root in path.parents or any(parent.name.startswith(pool_prefix) for parent in path.parents):
            continue
        try:
            prior, _ = read_identity_metadata(path)
        except (KeyError, OSError, ValueError) as error:
            historical.append({"path": str(path), "readable": False, "error": type(error).__name__})
            continue
        historical.append({
            "path": str(path), "readable": True, "identity_count": int(prior.size),
            "overlap_count": int(len(identity_set.intersection(prior.tolist()))),
        })

    masks_equal = True; non_jnr_equal = True; jnr_contract = True; level_contract = []
    for level in POWERS:
        values, mask = read_selected_parameters(files[level], indices)
        masks_equal &= bool(np.array_equal(mask, reference_mask))
        active_non_jnr = mask[:, :, 1]
        non_jnr_equal &= bool(np.allclose(values[:, :, 1][active_non_jnr], reference_values[:, :, 1][active_non_jnr], rtol=0.0, atol=1.0e-9))
        expected = np.asarray([30.0, 30.0, 30.0, level, level, level], dtype=np.float32)
        active_jnr = mask[:, :, 0] & reference_labels.astype(bool)
        maximum = float(np.abs(values[:, :, 0] - expected[None, :])[active_jnr].max())
        passed = maximum <= 1.0e-4
        jnr_contract &= passed
        level_contract.append({"power_db": level, "maximum_active_jnr_contract_error_db": maximum, "jnr_contract_passed": passed})
    unreadable = [row for row in historical if not row["readable"]]
    overlaps = [row for row in historical if row.get("overlap_count", 0) != 0]
    result = {
        "power_root": str(root), "alignment": alignment, "identity_sha256": identity_hash(reference_seeds),
        "identity_minimum": int(reference_seeds.min()), "identity_maximum": int(reference_seeds.max()),
        "historical_train_file_audit": {"checked_file_count": len(historical), "unreadable_file_count": len(unreadable), "overlap_file_count": len(overlaps), "overlap_rows": overlaps, "unreadable_rows": unreadable},
        "parameter_contract": {"masks_identical_across_powers": masks_equal, "non_jnr_active_parameters_identical_across_powers": non_jnr_equal, "jnr_targets_follow_fixed_first3_swept_last3_contract": jnr_contract, "levels": level_contract},
        "valid_for_frozen_inference": bool(alignment["samples"] == 3780 and alignment["unique_identities"] == 3780 and alignment["combinations"] == 63 and not unreadable and not overlaps and masks_equal and non_jnr_equal and jnr_contract),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    result["audit_sha256"] = _sha256(result)
    output = project / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"valid_for_frozen_inference": result["valid_for_frozen_inference"], "historical_train_file_audit": result["historical_train_file_audit"], "audit_sha256": result["audit_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
