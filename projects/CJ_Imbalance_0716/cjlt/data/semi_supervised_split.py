"""Nested labeled subsets and label-hiding wrappers for semi-supervised runs."""

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


@dataclass
class NestedLabeledSplits:
    ratios: Tuple[float, ...]
    labeled: Dict[float, np.ndarray]
    unlabeled: Dict[float, np.ndarray]
    sha256: str


def _combination_code(labels):
    powers = np.left_shift(1, np.arange(labels.shape[1], dtype=np.int64))
    return (labels.astype(np.int64) * powers[None, :]).sum(axis=1)


def build_nested_labeled_splits(
    labels,
    optimization_indices: Iterable[int],
    ratios: Sequence[float],
    seed=2026,
):
    target = np.asarray(labels, dtype=np.uint8)
    indices = np.asarray(list(optimization_indices), dtype=np.int64)
    if target.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    if indices.ndim != 1:
        raise ValueError("optimization_indices must be one-dimensional")
    if indices.size and (indices.min() < 0 or indices.max() >= len(target)):
        raise IndexError("optimization_indices contain an out-of-range index")
    ratio_values = tuple(sorted(set(float(value) for value in ratios)))
    if not ratio_values or ratio_values[-1] > 1.0 or ratio_values[0] <= 0.0:
        raise ValueError("ratios must be in (0, 1]")

    rng = np.random.RandomState(int(seed))
    code = _combination_code(target[indices])
    ordered_by_group = {}
    for value in sorted(np.unique(code)):
        group = indices[code == value].copy()
        rng.shuffle(group)
        ordered_by_group[int(value)] = group

    labeled = {}
    unlabeled = {}
    universe = set(int(value) for value in indices)
    for ratio in ratio_values:
        selected_parts = []
        for group in ordered_by_group.values():
            count = len(group) if ratio >= 1.0 else int(np.ceil(len(group) * ratio))
            selected_parts.append(group[:count])
        selected = np.sort(np.concatenate(selected_parts)).astype(np.int64)
        selected_set = set(int(value) for value in selected)
        remaining = np.asarray(sorted(universe.difference(selected_set)), dtype=np.int64)
        labeled[ratio] = selected
        unlabeled[ratio] = remaining

    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("ascii"))
    for ratio in ratio_values:
        digest.update(("{:.8f}".format(ratio)).encode("ascii"))
        digest.update(np.asarray(labeled[ratio], dtype="<i8").tobytes())
        digest.update(np.asarray(unlabeled[ratio], dtype="<i8").tobytes())
    return NestedLabeledSplits(
        ratios=ratio_values,
        labeled=labeled,
        unlabeled=unlabeled,
        sha256=digest.hexdigest(),
    )


class HiddenLabelDataset(Dataset):
    """Expose signals and identity while withholding all training truth."""

    _HIDDEN_KEYS = {"label", "parameter_values", "parameter_mask"}

    def __init__(self, labeled_dataset):
        self.labeled_dataset = labeled_dataset

    def __len__(self):
        return len(self.labeled_dataset)

    def __getitem__(self, index):
        item = self.labeled_dataset[index]
        return {
            key: value for key, value in item.items() if key not in self._HIDDEN_KEYS
        }

    def close(self):
        close = getattr(self.labeled_dataset, "close", None)
        if close is not None:
            close()

