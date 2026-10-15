"""Identity-safe data utilities for frequency--visibility experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from torch.utils.data import Dataset

from cj0724.data import complex_rms_normalize


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
POWER_LEVELS = (10, 15, 20, 25, 30, 35, 40, 45, 50)
ROLE_COUNTS_PER_COMBO = {
    "optimization": 280,
    "early_stop": 40,
    "calibration": 40,
    "development": 40,
}


def combo_bits(label: np.ndarray) -> str:
    return "".join(str(int(value)) for value in np.asarray(label).tolist())


def _json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def indices_sha256(indices: Iterable[int]) -> str:
    values = np.asarray(list(indices), dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def discover_power_train_files(power_root: str | Path) -> dict[int, Path]:
    root = Path(power_root)
    manifest_path = root / "power_sweep_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Power-sweep manifest is missing: {manifest_path}")
    files: dict[int, Path] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            level = int(float(row["sweep_JNR_dB"]))
            stale = Path(row["train_path"])
            candidates = [
                stale,
                root / Path(row.get("output_dir", "")).name / stale.name,
                *sorted(root.glob(f"level_*__jnr_{level}dB/{stale.name}")),
            ]
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Training file for {level} dB is missing")
            files[level] = path
    missing = sorted(set(POWER_LEVELS).difference(files))
    if missing:
        raise ValueError(f"Power levels are missing: {missing}")
    return {level: files[level] for level in POWER_LEVELS}


def read_identity_metadata(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(str(path), "r") as handle:
        seeds_raw = np.asarray(handle["receiver_noise_seed"][:, :]).reshape(-1)
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
    seeds = np.rint(seeds_raw).astype(np.int64)
    if not np.array_equal(seeds_raw, seeds.astype(seeds_raw.dtype)):
        raise ValueError("receiver_noise_seed contains non-integral values")
    if labels.shape != (len(seeds), len(CLASS_NAMES)):
        raise ValueError(f"Unexpected label shape: {labels.shape}")
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("receiver_noise_seed is not unique")
    return seeds, np.ascontiguousarray(labels)


def audit_power_alignment(files: dict[int, Path]) -> dict:
    if not files:
        raise ValueError("power file mapping is empty")
    reference_level = 30 if 30 in files else min(files)
    reference_seeds, reference_labels = read_identity_metadata(files[reference_level])
    level_receipts = {}
    for level, path in files.items():
        seeds, labels = read_identity_metadata(path)
        if not np.array_equal(seeds, reference_seeds):
            raise ValueError(f"Identity order differs at {level} dB")
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f"Labels differ at {level} dB")
        level_receipts[str(level)] = {
            "path": str(path.resolve()),
            "samples": int(len(seeds)),
            "identity_sha256": hashlib.sha256(seeds.astype("<i8").tobytes()).hexdigest(),
        }
    combinations = sorted({combo_bits(label) for label in reference_labels})
    payload = {
        "samples": int(len(reference_seeds)),
        "unique_identities": int(len(np.unique(reference_seeds))),
        "combinations": int(len(combinations)),
        "label_counts": reference_labels.sum(axis=0).astype(int).tolist(),
        "levels": level_receipts,
        "locked_test_used": False,
    }
    payload["audit_sha256"] = _json_sha256(payload)
    return payload


def build_identity_roles(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    matrix = np.asarray(labels, dtype=np.uint8)
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(matrix):
        grouped.setdefault(combo_bits(label), []).append(index)
    if len(grouped) != 63:
        raise ValueError(f"Expected 63 combinations, got {len(grouped)}")
    rng = np.random.default_rng(int(seed))
    roles = {name: [] for name in ROLE_COUNTS_PER_COMBO}
    for key in sorted(grouped):
        candidates = np.asarray(grouped[key], dtype=np.int64)
        required = sum(ROLE_COUNTS_PER_COMBO.values())
        if len(candidates) < required:
            raise ValueError(f"Combination {key} has only {len(candidates)} samples")
        rng.shuffle(candidates)
        cursor = 0
        for role, count in ROLE_COUNTS_PER_COMBO.items():
            roles[role].extend(candidates[cursor : cursor + count].tolist())
            cursor += count
    result = {}
    for role, values in roles.items():
        array = np.asarray(values, dtype=np.int64)
        rng.shuffle(array)
        result[role] = array
    merged = np.concatenate(list(result.values()))
    if len(np.unique(merged)) != len(merged):
        raise ValueError("Identity roles overlap")
    return result


def roles_receipt(roles: dict[str, np.ndarray], labels: np.ndarray, seed: int) -> dict:
    payload = {"seed": int(seed), "roles": {}, "locked_test_used": False}
    for role, indices in roles.items():
        counts: dict[str, int] = {}
        for index in indices:
            key = combo_bits(labels[int(index)])
            counts[key] = counts.get(key, 0) + 1
        payload["roles"][role] = {
            "samples": int(len(indices)),
            "indices_sha256": indices_sha256(indices),
            "label_counts": labels[indices].sum(axis=0).astype(int).tolist(),
            "combo_min": int(min(counts.values())),
            "combo_max": int(max(counts.values())),
            "combination_count": int(len(counts)),
        }
    payload["split_sha256"] = _json_sha256(payload)
    return payload


def _group_indices(labels: np.ndarray, eligible_indices: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index in eligible_indices:
        grouped.setdefault(combo_bits(labels[int(index)]), []).append(int(index))
    return {
        key: np.asarray(values, dtype=np.int64) for key, values in sorted(grouped.items())
    }


def build_balanced_manifest(
    labels: np.ndarray, eligible_indices: Iterable[int], seed: int
) -> dict:
    """Select 2400 identities while keeping all 63 combinations nearly equal."""

    eligible = np.asarray(list(eligible_indices), dtype=np.int64)
    grouped = _group_indices(labels, eligible)
    if len(grouped) != 63:
        raise ValueError("Balanced manifest requires all 63 combinations")
    rng = np.random.default_rng(int(seed))
    selected = []
    combo_counts = {}
    for key, candidates in grouped.items():
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        count = 39 if key.count("1") == 1 else 38
        selected.extend(shuffled[:count].tolist())
        combo_counts[key] = count
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    if len(selected_array) != 2400:
        raise AssertionError("Balanced manifest must contain 2400 identities")
    payload = {
        "protocol": "balanced_2400",
        "seed": int(seed),
        "selected_indices": selected_array.tolist(),
        "sample_count": int(len(selected_array)),
        "label_counts": labels[selected_array].sum(axis=0).astype(int).tolist(),
        "combo_counts": combo_counts,
        "source_indices_sha256": indices_sha256(eligible),
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    return payload


def build_target_label_lt_manifest(
    labels: np.ndarray,
    eligible_indices: Iterable[int],
    target_label_index: int,
    seed: int,
    sample_count: int = 2400,
    head_positive_count: int = 1200,
    tail_positive_count: int = 400,
) -> dict:
    """Build an exact one-tail IF=3 manifest while retaining all combinations."""

    eligible = np.asarray(list(eligible_indices), dtype=np.int64)
    grouped = _group_indices(labels, eligible)
    combo_names = sorted(grouped)
    if len(combo_names) != 63:
        raise ValueError("Target label-LT manifest requires all 63 combinations")
    target_label_index = int(target_label_index)
    if target_label_index < 0 or target_label_index >= len(CLASS_NAMES):
        raise IndexError("target_label_index is outside the six labels")
    target = np.full(len(CLASS_NAMES), int(head_positive_count), dtype=np.float64)
    target[target_label_index] = float(tail_positive_count)
    combo_vectors = np.asarray(
        [[int(bit) for bit in name] for name in combo_names], dtype=np.float64
    )
    combo_count = len(combo_names)
    desired_per_combo = float(sample_count) / float(combo_count)

    # x is integer combo count; positive/negative slack minimizes deviation from
    # an otherwise uniform combination distribution.
    variable_count = 3 * combo_count
    exact = np.zeros((len(CLASS_NAMES) + 1, variable_count), dtype=np.float64)
    exact[: len(CLASS_NAMES), :combo_count] = combo_vectors.T
    exact[len(CLASS_NAMES), :combo_count] = 1.0
    exact_target = np.concatenate([target, [float(sample_count)]])
    closeness = np.zeros((combo_count, variable_count), dtype=np.float64)
    closeness[:, :combo_count] = np.eye(combo_count)
    closeness[:, combo_count : 2 * combo_count] = -np.eye(combo_count)
    closeness[:, 2 * combo_count :] = np.eye(combo_count)
    constraints = np.vstack([exact, closeness])
    lower_constraint = np.concatenate(
        [exact_target, np.full(combo_count, desired_per_combo)]
    )
    upper_constraint = lower_constraint.copy()
    available = np.asarray([len(grouped[name]) for name in combo_names], dtype=np.float64)
    upper_combo = np.minimum(available, 80.0)
    lower_bounds = np.concatenate([np.ones(combo_count), np.zeros(2 * combo_count)])
    upper_bounds = np.concatenate(
        [upper_combo, np.full(2 * combo_count, np.inf)]
    )
    rng = np.random.default_rng(int(seed) + target_label_index * 1009)
    objective = np.concatenate(
        [rng.uniform(0.0, 1.0e-6, combo_count), np.ones(2 * combo_count)]
    )
    solution = milp(
        objective,
        integrality=np.concatenate(
            [np.ones(combo_count), np.zeros(2 * combo_count)]
        ),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            constraints, lower_constraint, upper_constraint
        ),
        options={"time_limit": 30.0},
    )
    if not solution.success or solution.x is None:
        raise ValueError(f"Unable to construct target label-LT: {solution.message}")
    selected_counts = np.rint(solution.x[:combo_count]).astype(np.int64)
    selected = []
    combo_counts = {}
    for name, count in zip(combo_names, selected_counts):
        candidates = grouped[name].copy()
        rng.shuffle(candidates)
        selected.extend(candidates[: int(count)].tolist())
        combo_counts[name] = int(count)
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    achieved = labels[selected_array].sum(axis=0).astype(int)
    if len(selected_array) != sample_count or not np.array_equal(achieved, target.astype(int)):
        raise AssertionError("Constructed label-LT manifest misses its exact targets")
    payload = {
        "protocol": "target_label_lt_if3",
        "seed": int(seed),
        "target_label_index": target_label_index,
        "target_label": CLASS_NAMES[target_label_index],
        "selected_indices": selected_array.tolist(),
        "sample_count": int(len(selected_array)),
        "target_label_counts": target.astype(int).tolist(),
        "achieved_label_counts": achieved.tolist(),
        "combo_counts": combo_counts,
        "source_indices_sha256": indices_sha256(eligible),
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    return payload


class PowerViewDataset(Dataset):
    """Lazy dataset selecting one deterministic power view per identity and epoch."""

    def __init__(
        self,
        level_files: dict[int, Path],
        labels: np.ndarray,
        indices: Iterable[int],
        levels: Iterable[int],
        seed: int,
    ) -> None:
        self.level_files = {int(level): Path(path) for level, path in level_files.items()}
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.levels = tuple(int(level) for level in levels)
        self.seed = int(seed)
        self.epoch = 0
        if not self.levels or any(level not in self.level_files for level in self.levels):
            raise ValueError("Dataset levels are empty or unavailable")
        self._handles: dict[int, h5py.File] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(len(self.indices))

    def _get_handle(self, level: int) -> h5py.File:
        if level not in self._handles:
            self._handles[level] = h5py.File(str(self.level_files[level]), "r")
        return self._handles[level]

    def _choose_level(self, sample_index: int) -> int:
        if len(self.levels) == 1:
            return self.levels[0]
        mixed = (
            int(sample_index) * 1103515245
            + self.seed * 12345
            + self.epoch * 2654435761
        ) & 0x7FFFFFFF
        return self.levels[mixed % len(self.levels)]

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        sample_index = int(self.indices[int(position)])
        level = self._choose_level(sample_index)
        handle = self._get_handle(level)
        raw = np.asarray(handle["data_legacy"][:, sample_index], dtype=np.float32)
        iq = complex_rms_normalize(raw.reshape(2, -1))
        label = self.labels[sample_index]
        weak = np.zeros(len(CLASS_NAMES), dtype=bool)
        fixed_present = bool(label[:3].any())
        swept_present = bool(label[3:].any())
        if level < 30 and fixed_present:
            weak[3:] = label[3:].astype(bool)
        elif level > 30 and swept_present:
            weak[:3] = label[:3].astype(bool)
        return {
            "iq": torch.from_numpy(iq),
            "label": torch.from_numpy(
                label.astype(np.float32, copy=True)
            ),
            "weak_mask": torch.from_numpy(weak.copy()),
            "sample_index": sample_index,
            "level": int(level),
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        self.close()


class PairedPowerViewDataset(PowerViewDataset):
    """Return an equal-power reference and a deterministic weak view together."""

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        sample_index = int(self.indices[int(position)])
        weak_level = self._choose_level(sample_index)
        reference_handle = self._get_handle(30)
        weak_handle = self._get_handle(weak_level)
        reference_raw = np.asarray(
            reference_handle["data_legacy"][:, sample_index], dtype=np.float32
        )
        weak_raw = np.asarray(
            weak_handle["data_legacy"][:, sample_index], dtype=np.float32
        )
        label = self.labels[sample_index]
        weak = np.zeros(len(CLASS_NAMES), dtype=bool)
        fixed_present = bool(label[:3].any())
        swept_present = bool(label[3:].any())
        if weak_level < 30 and fixed_present:
            weak[3:] = label[3:].astype(bool)
        elif weak_level > 30 and swept_present:
            weak[:3] = label[:3].astype(bool)
        return {
            "iq_reference": torch.from_numpy(
                complex_rms_normalize(reference_raw.reshape(2, -1))
            ),
            "iq_weak": torch.from_numpy(
                complex_rms_normalize(weak_raw.reshape(2, -1))
            ),
            "label": torch.from_numpy(label.astype(np.float32, copy=True)),
            "weak_mask": torch.from_numpy(weak.copy()),
            "sample_index": sample_index,
            "level": int(weak_level),
        }
