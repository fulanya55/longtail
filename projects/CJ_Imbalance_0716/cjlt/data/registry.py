"""Class registry and label selection rules for the 0716 dataset."""

from typing import List

import numpy as np


SIX_CLASS_NAMES = ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"]
FOUR_CLASS_NAMES = ["FMB", "SinFM", "MFT", "ISRJ"]
FOUR_CLASS_INDICES = [0, 2, 3, 5]
FOUR_CLASS_EXCLUDED_INDICES = [1, 4]
FIVE_CLASS_NAMES = ["FMB", "SinFM", "MFT", "VFT", "ISRJ"]
FIVE_CLASS_INDICES = [0, 2, 3, 4, 5]
FIVE_CLASS_EXCLUDED_INDICES = [1]


def _validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in ("four", "five", "six"):
        raise ValueError("Unknown class mode: {!r}. Expected 'four', 'five', or 'six'.".format(mode))
    return normalized


def _validate_labels6(labels6: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels6)
    if labels.ndim != 2 or labels.shape[1] != 6:
        raise ValueError("labels6 must have shape [N, 6], got {}".format(labels.shape))
    return labels


def get_class_names(mode: str) -> List[str]:
    normalized = _validate_mode(mode)
    if normalized == "four":
        return list(FOUR_CLASS_NAMES)
    if normalized == "five":
        return list(FIVE_CLASS_NAMES)
    return list(SIX_CLASS_NAMES)


def get_selected_indices(mode: str) -> List[int]:
    normalized = _validate_mode(mode)
    if normalized == "four":
        return list(FOUR_CLASS_INDICES)
    if normalized == "five":
        return list(FIVE_CLASS_INDICES)
    return list(range(6))


def select_labels(labels6: np.ndarray, mode: str) -> np.ndarray:
    labels = _validate_labels6(labels6)
    return labels[:, get_selected_indices(mode)].copy()


def valid_sample_mask(labels6: np.ndarray, mode: str) -> np.ndarray:
    labels = _validate_labels6(labels6)
    normalized = _validate_mode(mode)
    if normalized == "six":
        return labels.sum(axis=1) > 0

    excluded = FOUR_CLASS_EXCLUDED_INDICES if normalized == "four" else FIVE_CLASS_EXCLUDED_INDICES
    selected = FOUR_CLASS_INDICES if normalized == "four" else FIVE_CLASS_INDICES
    has_no_excluded_component = labels[:, excluded].sum(axis=1) == 0
    has_selected_component = labels[:, selected].sum(axis=1) > 0
    return np.logical_and(has_no_excluded_component, has_selected_component)
