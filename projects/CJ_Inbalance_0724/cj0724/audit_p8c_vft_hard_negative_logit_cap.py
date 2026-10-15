"""Train-only identity and power-pair audit for P8c against all historic generated train pools."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724.dual_axis import audit_power_alignment, combo_bits, discover_power_train_files, read_identity_metadata


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(fresh_ids: np.ndarray, fresh_labels: np.ndarray, reference_files: list[Path]) -> dict:
    combos = [combo_bits(row) for row in fresh_labels]
    counts = {combo: combos.count(combo) for combo in sorted(set(combos))}
    overlaps = {}
    for path in reference_files:
        reference_ids, _ = read_identity_metadata(path)
        count = int(np.intersect1d(fresh_ids, reference_ids).size)
        if count:
            overlaps[str(path)] = count
    return {
        "unique_fresh_identities": int(len(np.unique(fresh_ids))),
        "reference_file_count": int(len(reference_files)),
        "identity_overlap_by_reference": overlaps,
        "identity_overlap_total": int(sum(overlaps.values())),
        "combination_count": int(len(counts)),
        "combination_min": int(min(counts.values())),
        "combination_max": int(max(counts.values())),
        "all_63_combinations_have_400": bool(len(counts) == 63 and set(counts.values()) == {400}),
    }


def _reference_train_files(root: Path, fresh_root: Path) -> list[Path]:
    fresh = fresh_root.resolve()
    files = []
    for path in root.glob("**/closedset_6class_train.mat"):
        if fresh not in path.resolve().parents:
            files.append(path.resolve())
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--historic-generated-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregister.read_text(encoding="utf-8"))
    files = discover_power_train_files(args.data_root)
    paths = [str(path).lower() for path in files.values()]
    if any("test" in path for path in paths):
        raise RuntimeError("P8c audit refuses non-train paths")
    alignment = audit_power_alignment(files)
    fresh_ids, fresh_labels = read_identity_metadata(files[30])
    reference_files = _reference_train_files(args.historic_generated_root, args.data_root)
    detail = inspect(fresh_ids, fresh_labels, reference_files)
    gates = {
        "all_nine_powers_identity_and_label_aligned": len(alignment["levels"]) == 9,
        "only_train_paths_opened": all("test" not in path for path in paths),
        "historic_reference_pools_found": detail["reference_file_count"] > 0,
        "zero_overlap_with_all_historic_generated_train_pools": detail["identity_overlap_total"] == 0,
        "exactly_25200_unique_identities": detail["unique_fresh_identities"] == 25200,
        "all_63_combinations_have_400": detail["all_63_combinations_have_400"],
    }
    output = {
        "experiment": prereg["experiment"], "preregister_sha256": sha256_file(args.preregister),
        "data_root": str(args.data_root.resolve()), "historic_generated_root": str(args.historic_generated_root.resolve()),
        "alignment": alignment, "detail": detail, "gates": gates,
        "valid_for_p8c_training": bool(all(gates.values())),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"valid_for_p8c_training": output["valid_for_p8c_training"], "gates": gates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
