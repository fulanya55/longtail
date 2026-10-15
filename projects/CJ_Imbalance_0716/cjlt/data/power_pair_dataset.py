"""Lazy worker-safe dataset for identity-matched power views."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cjlt.data.power_pairs import (
    PowerPairRecord,
    discover_power_sweep_train_files,
)
from cjlt.data.registry import get_selected_indices
from cjlt.data.v73_dataset import _read_signal_from_handle


PathLike = Union[str, Path]


class PowerPairDataset(Dataset):
    def __init__(
        self,
        root: PathLike,
        records: Iterable[PowerPairRecord],
        class_mode="six",
        normalization="complex_rms",
        include_parameters=False,
    ):
        self.root = Path(root)
        self.records = list(records)
        self.class_mode = str(class_mode)
        self.selected_indices = get_selected_indices(self.class_mode)
        self.normalization = str(normalization)
        self.include_parameters = bool(include_parameters)
        if self.normalization not in ("none", "complex_rms"):
            raise ValueError("Unknown normalization: {!r}".format(normalization))
        self.level_files = discover_power_sweep_train_files(self.root)
        used_levels = set()
        for record in self.records:
            used_levels.add(int(record.reference_level))
            used_levels.add(int(record.paired_level))
        missing = sorted(used_levels.difference(self.level_files))
        if missing:
            raise ValueError("Power-pair records use missing levels: {}".format(missing))
        self._handles: Dict[int, h5py.File] = {}

    def __len__(self):
        return len(self.records)

    def _get_handle(self, level):
        level_value = int(level)
        if level_value not in self._handles:
            self._handles[level_value] = h5py.File(
                str(self.level_files[level_value]), "r"
            )
        return self._handles[level_value]

    def __getitem__(self, position):
        index = int(position)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset position outside valid range")
        record = self.records[index]
        reference_signal, reference_rms = _read_signal_from_handle(
            self._get_handle(record.reference_level),
            record.reference_index,
            self.normalization,
        )
        paired_signal, paired_rms = _read_signal_from_handle(
            self._get_handle(record.paired_level),
            record.paired_index,
            self.normalization,
        )
        selected = self.selected_indices
        label = record.labels[selected].astype(np.float32, copy=True)
        stronger = record.stronger_level_by_label[selected].astype(
            np.int16, copy=True
        )
        item = {
            "reference_signal": torch.from_numpy(reference_signal),
            "paired_signal": torch.from_numpy(paired_signal),
            "label": torch.from_numpy(label),
            "stronger_level_by_label": torch.from_numpy(stronger),
            "reference_level": int(record.reference_level),
            "paired_level": int(record.paired_level),
            "relative_power_gap": float(record.relative_power_gap),
            "identity": int(record.identity),
            "reference_index": int(record.reference_index),
            "paired_index": int(record.paired_index),
            "reference_raw_rms": float(reference_rms),
            "paired_raw_rms": float(paired_rms),
        }
        if self.include_parameters:
            reference_handle = self._get_handle(record.reference_level)
            paired_handle = self._get_handle(record.paired_level)
            for prefix, handle, sample_index in (
                ("reference", reference_handle, record.reference_index),
                ("paired", paired_handle, record.paired_index),
            ):
                values = np.asarray(
                    handle["parameter_values"][:, sample_index], dtype=np.float32
                )
                mask = np.asarray(
                    handle["parameter_mask"][:, sample_index], dtype=np.uint8
                ).astype(bool)
                item["{}_parameter_values".format(prefix)] = torch.from_numpy(values)
                item["{}_parameter_mask".format(prefix)] = torch.from_numpy(mask)
        return item

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        self.close()
