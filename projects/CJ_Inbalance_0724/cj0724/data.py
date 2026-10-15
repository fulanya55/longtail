"""Lazy MATLAB-v7.3 reader for the six-class 0716 training file."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SIGNAL_LENGTH = 3000
NUM_CLASSES = 6


def complex_rms_normalize(iq: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    """Return float32 [I, Q] scaled by sqrt(mean(I^2 + Q^2))."""
    signal = np.asarray(iq, dtype=np.float32)
    if signal.shape != (2, SIGNAL_LENGTH):
        raise ValueError(f"Expected I/Q shape (2, {SIGNAL_LENGTH}), got {signal.shape}")
    rms = float(np.sqrt(np.mean(signal[0] ** 2 + signal[1] ** 2)))
    if rms <= epsilon:
        return np.zeros_like(signal, dtype=np.float32)
    return np.ascontiguousarray(signal / rms, dtype=np.float32)


def read_labels(mat_path: str | Path) -> np.ndarray:
    """Load the small label matrix once as sample-major uint8 [N, 6]."""
    path = Path(mat_path)
    with h5py.File(path, "r") as handle:
        if "label_selected" not in handle:
            raise KeyError(f"label_selected is missing from {path}")
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
    if labels.ndim != 2 or labels.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected labels [N, {NUM_CLASSES}], got {labels.shape}")
    return np.ascontiguousarray(labels)


def make_train_val_split(
    sample_count: int,
    validation_fraction: float = 0.1,
    seed: int = 20260724,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic disjoint development split from the train MAT file."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    order = np.random.default_rng(int(seed)).permutation(int(sample_count))
    val_count = max(1, int(round(sample_count * validation_fraction)))
    val_indices = np.sort(order[:val_count])
    train_indices = np.sort(order[val_count:])
    if max_train_samples is not None:
        train_indices = np.sort(train_indices[: int(max_train_samples)])
    if max_val_samples is not None:
        val_indices = np.sort(val_indices[: int(max_val_samples)])
    return train_indices.astype(np.int64), val_indices.astype(np.int64)


class V73IQDataset(Dataset):
    """Worker-safe lazy dataset yielding normalized `[2, 3000]` I/Q tensors."""

    def __init__(
        self,
        mat_path: str | Path,
        indices: Iterable[int],
        parameter_columns: Iterable[int] | None = None,
    ) -> None:
        self.path = Path(mat_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"MAT-v7.3 file not found: {self.path}")
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.labels = read_labels(self.path)
        self.parameter_columns = (
            None
            if parameter_columns is None
            else np.asarray(list(parameter_columns), dtype=np.int64)
        )
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= len(self.labels)):
            raise IndexError("indices contain values outside the label matrix")
        if self.parameter_columns is not None:
            if self.parameter_columns.ndim != 1:
                raise ValueError("parameter_columns must be one-dimensional")
            if self.parameter_columns.size and (
                self.parameter_columns.min() < 0 or self.parameter_columns.max() >= 79
            ):
                raise IndexError("parameter_columns contain values outside [0, 78]")
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
            shape = tuple(self._handle["data_legacy"].shape)
            if shape[0] != 2 * SIGNAL_LENGTH:
                raise ValueError(f"Expected data_legacy [6000, N], got {shape}")
        return self._handle

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        sample_index = int(self.indices[int(position)])
        raw = np.asarray(self._get_handle()["data_legacy"][:, sample_index], dtype=np.float32)
        iq = complex_rms_normalize(raw.reshape(2, SIGNAL_LENGTH))
        item = {
            "iq": torch.from_numpy(iq),
            "label": torch.from_numpy(self.labels[sample_index].astype(np.float32, copy=True)),
            "sample_index": sample_index,
        }
        if self.parameter_columns is not None:
            handle = self._get_handle()
            values = np.asarray(
                handle["parameter_values"][self.parameter_columns, sample_index],
                dtype=np.float32,
            )
            mask = np.asarray(
                handle["parameter_mask"][self.parameter_columns, sample_index],
                dtype=np.uint8,
            ).astype(bool)
            item["parameter_values"] = torch.from_numpy(values.copy())
            item["parameter_mask"] = torch.from_numpy(mask.copy())
        return item

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()
