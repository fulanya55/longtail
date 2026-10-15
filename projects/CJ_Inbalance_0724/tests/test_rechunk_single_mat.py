from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from cj0724.rechunk_independent_power_data import REQUIRED_DATASETS, convert_file


def test_convert_single_mat_preserves_required_arrays(tmp_path: Path) -> None:
    source = tmp_path / "source.mat"
    target = tmp_path / "target.mat"
    arrays = {
        "data_legacy": np.arange(60, dtype=np.float32).reshape(12, 5),
        "label_selected": np.arange(30, dtype=np.uint8).reshape(6, 5),
        "parameter_values": np.arange(20, dtype=np.float32).reshape(4, 5),
        "parameter_mask": np.ones((4, 5), dtype=np.uint8),
        "receiver_noise_seed": np.arange(5, dtype=np.int64).reshape(1, 5),
    }
    with h5py.File(source, "w") as handle:
        for name in REQUIRED_DATASETS:
            handle.create_dataset(name, data=arrays[name])
    receipt = convert_file(source, target)
    assert receipt["chunks"] == (12, 1)
    with h5py.File(target, "r") as handle:
        for name, expected in arrays.items():
            np.testing.assert_array_equal(handle[name][:], expected)
