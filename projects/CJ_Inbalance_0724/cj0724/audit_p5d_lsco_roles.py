"""Independently verify the frozen P5d composition-disjoint role arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724.dual_axis import combo_bits, read_identity_metadata


ROLE_NAMES = ("train", "early", "seen_confirm", "unseen")


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def audit(data_file: str | Path, roles_file: str | Path, receipt_file: str | Path) -> dict:
    identities, labels = read_identity_metadata(data_file)
    roles = np.load(roles_file)
    receipt = json.loads(Path(receipt_file).read_text(encoding="utf-8"))
    combo = np.asarray([int(combo_bits(row), 2) for row in labels], dtype=np.int64)
    folds = {}
    for fold in range(3):
        arrays = {name: np.asarray(roles[f"fold{fold}_{name}"], dtype=np.int64) for name in ROLE_NAMES}
        all_indices = np.concatenate(list(arrays.values()))
        unseen_combinations = np.asarray(receipt["folds"][str(fold)]["unseen_combinations"], dtype=np.int64)
        role_combo = {name: np.unique(combo[index]).tolist() for name, index in arrays.items()}
        disjoint = all(
            not np.intersect1d(arrays[left], arrays[right]).size
            for i, left in enumerate(ROLE_NAMES)
            for right in ROLE_NAMES[i + 1 :]
        )
        partition = np.array_equal(np.sort(all_indices), np.arange(len(identities), dtype=np.int64))
        seen_names = ("train", "early", "seen_confirm")
        seen_clear = all(
            not np.intersect1d(combo[arrays[name]], unseen_combinations).size for name in seen_names
        )
        unseen_exact = np.array_equal(np.unique(combo[arrays["unseen"]]), unseen_combinations)
        receipt_hashes = {
            name: _digest(identities[index]) == receipt["folds"][str(fold)]["roles"][name]["identity_sha256"]
            for name, index in arrays.items()
        }
        folds[str(fold)] = {
            "role_counts": {name: int(len(index)) for name, index in arrays.items()},
            "roles_pairwise_disjoint": disjoint,
            "roles_partition_all_identities": partition,
            "seen_roles_exclude_unseen_combinations": seen_clear,
            "unseen_role_matches_receipt_combinations": unseen_exact,
            "receipt_identity_hashes_match": receipt_hashes,
            "valid": bool(disjoint and partition and seen_clear and unseen_exact and all(receipt_hashes.values())),
            "role_combinations": role_combo,
        }
    return {
        "data_file": str(Path(data_file).resolve()),
        "roles_file": str(Path(roles_file).resolve()),
        "receipt_file": str(Path(receipt_file).resolve()),
        "sample_count": int(len(identities)),
        "identity_sha256_matches_receipt": _digest(identities) == receipt["identity_sha256"],
        "folds": folds,
        "all_folds_valid": all(fold["valid"] for fold in folds.values()),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.data_file, args.roles, args.receipt)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"all_folds_valid": result["all_folds_valid"], "sample_count": result["sample_count"]}, indent=2))


if __name__ == "__main__":
    main()
