"""MATLAB v7.3/HDF5 reader for the 0716 compound-jamming dataset."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cjlt.data.registry import get_selected_indices


PathLike = Union[str, Path]
SIGNAL_LENGTH = 3000
NUM_SAMPLES = 25200


def _as_path(path: PathLike) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError("MAT v7.3 file not found: {}".format(resolved))
    return resolved


def _validate_sample_index(sample_index: int, sample_count: int) -> int:
    index = int(sample_index)
    if index < 0 or index >= int(sample_count):
        raise IndexError(
            "sample_index {} outside valid range [0, {})".format(index, sample_count)
        )
    return index


def inspect_v73_layout(path: PathLike) -> Dict[str, Dict[str, object]]:
    mat_path = _as_path(path)
    keys = (
        "data_legacy",
        "iq_complex",
        "label_selected",
        "known_label",
        "parameter_values",
        "parameter_mask",
    )
    layout = {}
    with h5py.File(str(mat_path), "r") as handle:
        for key in keys:
            if key not in handle:
                raise KeyError("Required dataset {!r} missing from {}".format(key, mat_path))
            dataset = handle[key]
            layout[key] = {
                "shape": tuple(int(value) for value in dataset.shape),
                "dtype": str(dataset.dtype),
            }
    return layout


def read_labels(path: PathLike) -> np.ndarray:
    mat_path = _as_path(path)
    with h5py.File(str(mat_path), "r") as handle:
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
    if labels.ndim != 2 or labels.shape[1] != 6:
        raise ValueError("Expected sample-major labels [N, 6], got {}".format(labels.shape))
    return np.ascontiguousarray(labels)


def read_parameters(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    mat_path = _as_path(path)
    with h5py.File(str(mat_path), "r") as handle:
        values = np.asarray(handle["parameter_values"][:, :], dtype=np.float32).T
        mask = np.asarray(handle["parameter_mask"][:, :], dtype=np.uint8).T.astype(bool)
    if values.shape != mask.shape or values.ndim != 2 or values.shape[1] != 79:
        raise ValueError(
            "Expected parameter values/mask [N, 79], got {} and {}".format(
                values.shape, mask.shape
            )
        )
    return np.ascontiguousarray(values), np.ascontiguousarray(mask)


def complex_rms_normalize(
    signal: np.ndarray, epsilon: float = 1e-12
) -> Tuple[np.ndarray, float]:
    iq = np.asarray(signal, dtype=np.float32)
    if iq.ndim != 2 or iq.shape[0] != 2:
        raise ValueError("signal must have shape [2, L], got {}".format(iq.shape))
    raw_power = float(np.mean(iq[0] ** 2 + iq[1] ** 2))
    raw_rms = float(np.sqrt(max(raw_power, 0.0)))
    if raw_rms == 0.0:
        return np.zeros_like(iq, dtype=np.float32), 0.0
    normalized = iq / float(np.sqrt(raw_power + float(epsilon)))
    return normalized.astype(np.float32, copy=False), raw_rms


def _read_signal_from_handle(
    handle: h5py.File, sample_index: int, normalization: str
) -> Tuple[np.ndarray, float]:
    sample_count = int(handle["data_legacy"].shape[1])
    index = _validate_sample_index(sample_index, sample_count)
    raw = np.asarray(handle["data_legacy"][:, index], dtype=np.float32)
    if raw.shape != (2 * SIGNAL_LENGTH,):
        raise ValueError(
            "Expected data_legacy sample shape ({},), got {}".format(
                2 * SIGNAL_LENGTH, raw.shape
            )
        )
    signal = raw.reshape(2, SIGNAL_LENGTH)
    raw_power = float(np.mean(signal[0] ** 2 + signal[1] ** 2))
    raw_rms = float(np.sqrt(max(raw_power, 0.0)))
    if normalization == "none":
        return np.ascontiguousarray(signal, dtype=np.float32), raw_rms
    if normalization == "complex_rms":
        return complex_rms_normalize(signal)
    raise ValueError(
        "Unknown normalization {!r}; expected 'none' or 'complex_rms'.".format(
            normalization
        )
    )


def read_signal(
    path: PathLike, sample_index: int, normalization: str = "complex_rms"
) -> Tuple[np.ndarray, float]:
    mat_path = _as_path(path)
    with h5py.File(str(mat_path), "r") as handle:
        return _read_signal_from_handle(handle, sample_index, normalization)


def read_complex_signal(path: PathLike, sample_index: int) -> np.ndarray:
    mat_path = _as_path(path)
    with h5py.File(str(mat_path), "r") as handle:
        sample_count = int(handle["iq_complex"].shape[1])
        index = _validate_sample_index(sample_index, sample_count)
        stored = np.asarray(handle["iq_complex"][:, index])
    if stored.dtype.fields is None or "real" not in stored.dtype.fields or "imag" not in stored.dtype.fields:
        raise TypeError("iq_complex must use compound fields named 'real' and 'imag'")
    return np.asarray(stored["real"] + 1j * stored["imag"], dtype=np.complex128)


class V73SignalDataset(Dataset):
    """Lazy worker-safe reader returning raw I/Q tensors and sample metadata."""

    def __init__(
        self,
        path: PathLike,
        indices: Iterable[int],
        class_mode: str = "six",
        normalization: str = "complex_rms",
        include_parameters: bool = False,
    ) -> None:
        self.path = _as_path(path)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.class_mode = str(class_mode)
        self.selected_indices = get_selected_indices(self.class_mode)
        self.normalization = str(normalization)
        if self.normalization not in ("none", "complex_rms"):
            raise ValueError("Unknown normalization: {!r}".format(normalization))
        self.include_parameters = bool(include_parameters)
        self.labels6 = read_labels(self.path)
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if self.indices.size:
            _validate_sample_index(int(self.indices.min()), len(self.labels6))
            _validate_sample_index(int(self.indices.max()), len(self.labels6))
        self._handle: Optional[h5py.File] = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(str(self.path), "r")
        return self._handle

    def __getitem__(self, position: int) -> Dict[str, object]:
        pos = int(position)
        if pos < 0:
            pos += len(self)
        if pos < 0 or pos >= len(self):
            raise IndexError("dataset position {} outside [0, {})".format(position, len(self)))
        sample_index = int(self.indices[pos])
        handle = self._get_handle()
        signal, raw_rms = _read_signal_from_handle(
            handle, sample_index, self.normalization
        )
        label = self.labels6[sample_index, self.selected_indices].astype(
            np.float32, copy=True
        )
        item: Dict[str, object] = {
            "signal": torch.from_numpy(signal),
            "label": torch.from_numpy(label),
            "sample_index": sample_index,
            "raw_rms": raw_rms,
        }
        if self.include_parameters:
            values = np.asarray(
                handle["parameter_values"][:, sample_index], dtype=np.float32
            )
            mask = np.asarray(
                handle["parameter_mask"][:, sample_index], dtype=np.uint8
            ).astype(bool)
            item["parameter_values"] = torch.from_numpy(values)
            item["parameter_mask"] = torch.from_numpy(mask)
        return item

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self) -> Dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()

