"""Build a train-only, hash-recorded multi-axis long-tail protocol atlas.

P8a deliberately contains no trainable model.  It freezes the sampling contract
needed before a later long-tail mechanism can be evaluated fairly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from cj0724.dual_axis import (
    CLASS_NAMES,
    audit_power_alignment,
    build_identity_roles,
    combo_bits,
    discover_power_train_files,
    indices_sha256,
    read_identity_metadata,
    roles_receipt,
)


TARGET_INDEX = CLASS_NAMES.index("VFT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: dict) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _groups(labels: np.ndarray, eligible: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    grouped: dict[str, list[int]] = {}
    for index in np.asarray(eligible, dtype=np.int64):
        grouped.setdefault(combo_bits(labels[int(index)]), []).append(int(index))
    names = sorted(grouped)
    if len(names) != 63:
        raise ValueError(f"Expected all 63 nonempty combinations, got {len(names)}")
    return names, [np.asarray(grouped[name], dtype=np.int64) for name in names]


def _desired_combo_counts(names: list[str], mode: str, sample_count: int) -> np.ndarray:
    if mode == "balanced":
        return np.full(len(names), float(sample_count) / len(names), dtype=np.float64)
    tail = np.asarray([sum(map(int, name)) % 2 == 0 for name in names], dtype=bool)
    desired = np.empty(len(names), dtype=np.float64)
    desired[tail] = 10.0
    desired[~tail] = float(sample_count - desired[tail].sum()) / float((~tail).sum())
    return desired


def _solve_counts(
    labels: np.ndarray,
    eligible: np.ndarray,
    target_labels: list[int],
    mode: str,
    sample_count: int,
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    names, members = _groups(labels, eligible)
    vectors = np.asarray([[int(bit) for bit in name] for name in names], dtype=np.float64)
    n = len(names)
    # Integer combination counts x plus absolute-deviation variables d+ and d-.
    variables = 3 * n
    objective = np.concatenate([np.zeros(n), np.ones(2 * n)])
    equalities = np.zeros((len(CLASS_NAMES) + 1 + n, variables), dtype=np.float64)
    equalities[: len(CLASS_NAMES), :n] = vectors.T
    equalities[len(CLASS_NAMES), :n] = 1.0
    equalities[len(CLASS_NAMES) + 1 :, :n] = np.eye(n)
    equalities[len(CLASS_NAMES) + 1 :, n : 2 * n] = -np.eye(n)
    equalities[len(CLASS_NAMES) + 1 :, 2 * n :] = np.eye(n)
    desired = _desired_combo_counts(names, mode, sample_count)
    rhs = np.concatenate([np.asarray(target_labels, dtype=np.float64), [float(sample_count)], desired])
    lower = np.concatenate([np.ones(n), np.zeros(2 * n)])
    upper = np.concatenate([np.asarray([len(x) for x in members], dtype=np.float64), np.full(2 * n, np.inf)])
    result = milp(
        c=objective,
        integrality=np.concatenate([np.ones(n), np.zeros(2 * n)]),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(equalities, rhs, rhs),
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Unable to construct {mode} manifest: {result.message}")
    counts = np.rint(result.x[:n]).astype(np.int64)
    if int(counts.sum()) != int(sample_count):
        raise AssertionError("Manifest sample total is incorrect")
    achieved = counts @ vectors
    if not np.array_equal(achieved.astype(np.int64), np.asarray(target_labels, dtype=np.int64)):
        raise AssertionError("Manifest label targets are incorrect")
    return names, members, counts


def _pick_indices(members: list[np.ndarray], counts: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for group, count in zip(members, counts):
        shuffled = group.copy()
        rng.shuffle(shuffled)
        selected.extend(shuffled[: int(count)].tolist())
    array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(array)
    return array


def _visibility_levels(labels: np.ndarray, selected: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    levels = np.full(len(selected), 30, dtype=np.int16)
    eligible = np.logical_and(labels[selected, TARGET_INDEX] == 1, labels[selected, :3].any(axis=1))
    weak = np.asarray((10, 15, 20), dtype=np.int16)
    for position in np.flatnonzero(eligible):
        mixed = (int(selected[position]) * 1103515245 + int(seed) * 12345) & 0x7FFFFFFF
        levels[position] = weak[mixed % len(weak)]
    return levels, eligible


def build_manifest(
    labels: np.ndarray,
    eligible: np.ndarray,
    name: str,
    target_labels: list[int],
    mode: str,
    seed: int,
    visibility_tail: bool,
) -> dict:
    names, members, counts = _solve_counts(labels, eligible, target_labels, mode, 2400)
    selected = _pick_indices(members, counts, seed)
    levels = np.full(len(selected), 30, dtype=np.int16)
    visibility = np.zeros(len(selected), dtype=bool)
    if visibility_tail:
        levels, visibility = _visibility_levels(labels, selected, seed)
    achieved = labels[selected].sum(axis=0).astype(int)
    combo_counts = {combo: int(count) for combo, count in zip(names, counts)}
    payload = {
        "name": name,
        "sample_count": int(len(selected)),
        "target_label_counts": [int(value) for value in target_labels],
        "achieved_label_counts": achieved.tolist(),
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": indices_sha256(selected),
        "source_indices_sha256": indices_sha256(eligible),
        "combination_counts": combo_counts,
        "combination_mode": mode,
        "visibility_tail": bool(visibility_tail),
        "view_level_by_selected_position_db": levels.astype(int).tolist(),
        "visibility_tail_positions": np.flatnonzero(visibility).astype(int).tolist(),
        "visibility_level_counts": {str(level): int((levels == level).sum()) for level in (10, 15, 20, 30)},
    }
    payload["manifest_sha256"] = _payload_hash(payload)
    return payload


def validate_manifest(manifest: dict, labels: np.ndarray) -> dict:
    selected = np.asarray(manifest["selected_indices"], dtype=np.int64)
    levels = np.asarray(manifest["view_level_by_selected_position_db"], dtype=np.int16)
    target = np.asarray(manifest["target_label_counts"], dtype=np.int64)
    actual = labels[selected].sum(axis=0).astype(np.int64)
    combos = manifest["combination_counts"]
    visible_positions = np.asarray(manifest["visibility_tail_positions"], dtype=np.int64)
    checks = {
        "sample_count_2400": len(selected) == 2400,
        "unique_indices": len(np.unique(selected)) == len(selected),
        "all_63_combinations_retained": len(combos) == 63 and min(combos.values()) >= 1,
        "exact_label_targets": bool(np.array_equal(actual, target)),
        "view_levels_allowed": bool(np.isin(levels, np.asarray((10, 15, 20, 30), dtype=np.int16)).all()),
        "visibility_positions_are_weak_only": bool((levels[visible_positions] < 30).all()) if len(visible_positions) else not manifest["visibility_tail"],
    }
    if manifest["visibility_tail"]:
        expected = np.logical_and(labels[selected, TARGET_INDEX] == 1, labels[selected, :3].any(axis=1))
        checks["all_and_only_vft_fixed_rows_are_visibility_tail"] = bool(np.array_equal(np.flatnonzero(expected), visible_positions))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.preregister.read_text(encoding="utf-8"))
    freeze = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    if sha256_file(args.preregister) != freeze["preregister_sha256"]:
        raise RuntimeError("P8a preregistration hash does not match method freeze")
    if sha256_file(Path(__file__)) != freeze["atlas_code_sha256"]:
        raise RuntimeError("P8a code hash does not match method freeze")
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")
    power_root = Path(protocol["input"]["power_root"])
    files = discover_power_train_files(power_root)
    if any("test" in str(path).lower() for path in files.values()):
        raise RuntimeError("P8a only permits train paths")
    alignment = audit_power_alignment(files)
    _, labels = read_identity_metadata(files[30])
    roles = build_identity_roles(labels, int(protocol["identity_roles"]["split_seed"]))
    eligible = roles["optimization"]
    definitions = protocol["manifests"]
    manifests = {
        "balanced": build_manifest(labels, eligible, "balanced", definitions["balanced"]["label_targets"], "balanced", 1300000001, False),
        "combo_lt": build_manifest(labels, eligible, "combo_lt", definitions["combo_lt"]["label_targets"], "combo_lt", 1300000002, False),
        "mixed_lt": build_manifest(labels, eligible, "mixed_lt", definitions["mixed_lt"]["label_targets"], "combo_lt", 1300000003, True),
    }
    checks = {name: validate_manifest(manifest, labels) for name, manifest in manifests.items()}
    output = {
        "experiment": protocol["experiment"],
        "preregister_sha256": sha256_file(args.preregister),
        "method_freeze_sha256": sha256_file(args.method_freeze),
        "train_only_power_alignment": alignment,
        "roles": roles_receipt(roles, labels, int(protocol["identity_roles"]["split_seed"])),
        "manifests": manifests,
        "checks": checks,
        "passed": bool(all(all(values.values()) for values in checks.values())),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
        "evidence_boundary": "P8a is a data-protocol artifact only. It contains no trained model, no prediction, and no long-tail effectiveness claim."
    }
    args.output_root.mkdir(parents=True)
    (args.output_root / "result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": output["passed"], "checks": checks, "test_opened": False, "locked_test_used": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
