"""Deterministic combination-stratified development splits."""

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, Sequence

import numpy as np


def combo_key(row: Sequence[int]) -> str:
    values = np.asarray(row).reshape(-1)
    if values.size == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("combo labels must be a nonempty binary vector")
    return "".join(str(int(value)) for value in values)


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError("labels must have shape [N, C], got {}".format(array.shape))
    if not np.isin(array, [0, 1]).all():
        raise ValueError("labels must be binary")
    return array.astype(np.uint8, copy=False)


def group_indices_by_combo(
    labels: np.ndarray, eligible_indices: Iterable[int]
) -> Dict[str, np.ndarray]:
    matrix = _validate_labels(labels)
    indices = np.asarray(list(eligible_indices), dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("eligible_indices must be one-dimensional")
    if indices.size and (indices.min() < 0 or indices.max() >= len(matrix)):
        raise IndexError("eligible_indices contain an out-of-range sample index")
    groups = {}
    for index in indices:
        key = combo_key(matrix[int(index)])
        groups.setdefault(key, []).append(int(index))
    return {
        key: np.asarray(group_indices, dtype=np.int64)
        for key, group_indices in groups.items()
    }


@dataclass(frozen=True)
class RoleCounts:
    optimization: int = 280
    early_stop: int = 40
    calibration: int = 40
    threshold_selection: int = 40

    @property
    def total(self) -> int:
        return int(
            self.optimization
            + self.early_stop
            + self.calibration
            + self.threshold_selection
        )


@dataclass
class SplitRoles:
    optimization: np.ndarray
    early_stop: np.ndarray
    calibration: np.ndarray
    threshold_selection: np.ndarray
    seed: int
    counts: RoleCounts
    sha256: str

    def assert_disjoint(self) -> None:
        roles = [
            self.optimization,
            self.early_stop,
            self.calibration,
            self.threshold_selection,
        ]
        sets = [set(int(value) for value in role.tolist()) for role in roles]
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                overlap = sets[left] & sets[right]
                if overlap:
                    raise AssertionError(
                        "split roles overlap at {} sample indices".format(len(overlap))
                    )


def _manifest_hash(
    roles: Sequence[np.ndarray], seed: int, counts: RoleCounts
) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(
        "{},{},{},{}".format(
            counts.optimization,
            counts.early_stop,
            counts.calibration,
            counts.threshold_selection,
        ).encode("ascii")
    )
    for role in roles:
        digest.update(np.asarray(role, dtype="<i8").tobytes())
    return digest.hexdigest()


def build_stratified_roles(
    labels: np.ndarray,
    eligible_indices: Iterable[int],
    seed: int,
    counts: RoleCounts = RoleCounts(),
) -> SplitRoles:
    matrix = _validate_labels(labels)
    groups = group_indices_by_combo(matrix, eligible_indices)
    if not groups:
        raise ValueError("No eligible samples were provided")
    if min(
        counts.optimization,
        counts.early_stop,
        counts.calibration,
        counts.threshold_selection,
    ) < 0:
        raise ValueError("Role counts must be nonnegative")

    rng = np.random.RandomState(int(seed))
    role_lists = [[], [], [], []]
    for key in sorted(groups):
        group = groups[key].copy()
        if len(group) < counts.total:
            raise ValueError(
                "Combination {} has {} samples but split requires {}".format(
                    key, len(group), counts.total
                )
            )
        rng.shuffle(group)
        boundaries = np.cumsum(
            [
                0,
                counts.optimization,
                counts.early_stop,
                counts.calibration,
                counts.threshold_selection,
            ]
        )
        for role_index in range(4):
            start = int(boundaries[role_index])
            end = int(boundaries[role_index + 1])
            role_lists[role_index].extend(group[start:end].tolist())

    roles = [np.asarray(values, dtype=np.int64) for values in role_lists]
    for role in roles:
        rng.shuffle(role)
    split = SplitRoles(
        optimization=roles[0],
        early_stop=roles[1],
        calibration=roles[2],
        threshold_selection=roles[3],
        seed=int(seed),
        counts=counts,
        sha256=_manifest_hash(roles, seed, counts),
    )
    split.assert_disjoint()
    return split

