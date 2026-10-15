"""Create fixed identity roles shared by all P5b1 power views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724.dual_axis import combo_bits, read_identity_metadata


COUNTS = {"train": 36, "early": 12, "confirm": 12}


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    identities, labels = read_identity_metadata(args.reference_file)
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(combo_bits(label), []).append(index)
    if len(grouped) != 63 or any(len(values) != 60 for values in grouped.values()):
        raise ValueError("P5b1 roles require exactly 63 combinations with 60 identities each")
    rng = np.random.default_rng(args.seed)
    roles = {name: [] for name in COUNTS}
    per_combo = {}
    for combination, values in sorted(grouped.items()):
        order = np.asarray(values, dtype=np.int64)
        rng.shuffle(order)
        cursor = 0
        per_combo[combination] = {}
        for role, count in COUNTS.items():
            selected = order[cursor:cursor + count]
            roles[role].extend(selected.tolist())
            per_combo[combination][role] = int(selected.size)
            cursor += count
    arrays = {}
    receipt_roles = {}
    for role, values in roles.items():
        array = np.asarray(values, dtype=np.int64)
        rng.shuffle(array)
        arrays[role] = array
        receipt_roles[role] = {
            "samples": int(array.size),
            "indices_sha256": array_hash(array),
            "identities_sha256": array_hash(identities[array]),
            "combination_count": int(len({combo_bits(labels[index]) for index in array})),
        }
    merged = np.concatenate(list(arrays.values()))
    if merged.size != identities.size or np.unique(merged).size != identities.size:
        raise RuntimeError("P5b1 identity roles are not a disjoint full partition")
    output_npz = Path(args.output_npz)
    output_json = Path(args.output_json)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    receipt = {
        "seed": int(args.seed),
        "reference_file": str(Path(args.reference_file).resolve()),
        "identity_count": int(identities.size),
        "identity_sha256": array_hash(identities),
        "role_counts_per_combination": COUNTS,
        "roles": receipt_roles,
        "all_roles_disjoint": True,
        "all_identities_assigned": True,
        "same_roles_required_at_all_powers": True,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    output_json.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
