import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from cjlt.data.registry import valid_sample_mask
from cjlt.data.v73_dataset import (
    V73SignalDataset,
    complex_rms_normalize,
    inspect_v73_layout,
    read_complex_signal,
    read_labels,
    read_parameters,
    read_signal,
)


TRAIN_PATH = Path(
    r"C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"
)


def test_inspect_v73_layout_matches_0716_train_file():
    layout = inspect_v73_layout(TRAIN_PATH)
    assert layout["data_legacy"]["shape"] == (6000, 25200)
    assert layout["iq_complex"]["shape"] == (3000, 25200)
    assert layout["label_selected"]["shape"] == (6, 25200)
    assert layout["parameter_values"]["shape"] == (79, 25200)
    assert layout["parameter_mask"]["shape"] == (79, 25200)


def test_read_labels_and_parameters_return_sample_major_arrays():
    labels = read_labels(TRAIN_PATH)
    values, mask = read_parameters(TRAIN_PATH)

    assert labels.shape == (25200, 6)
    assert labels.dtype == np.uint8
    assert values.shape == (25200, 79)
    assert values.dtype == np.float32
    assert mask.shape == (25200, 79)
    assert mask.dtype == np.bool_
    assert np.isfinite(values[mask]).all()
    assert np.isnan(values[~mask]).all()


def test_real_four_class_filter_yields_6000_clean_samples():
    labels = read_labels(TRAIN_PATH)
    mask = valid_sample_mask(labels, "four")
    selected = labels[mask][:, [0, 2, 3, 5]]

    assert int(mask.sum()) == 6000
    assert np.unique(selected, axis=0).shape[0] == 15
    assert selected.sum(axis=0).tolist() == [3200, 3200, 3200, 3200]


def test_legacy_and_compound_complex_representations_are_identical():
    indices = [0, 399, 400, 8399, 12799, 25199]
    with h5py.File(str(TRAIN_PATH), "r") as handle:
        for sample_index in indices:
            raw_legacy = np.asarray(handle["data_legacy"][:, sample_index])
            raw_compound = np.asarray(handle["iq_complex"][:, sample_index])
            raw_reconstructed = raw_legacy[:3000] + 1j * raw_legacy[3000:]
            stored_complex = raw_compound["real"] + 1j * raw_compound["imag"]
            np.testing.assert_allclose(raw_reconstructed, stored_complex, rtol=0.0, atol=0.0)

            legacy, _ = read_signal(TRAIN_PATH, sample_index, normalization="none")
            compound = read_complex_signal(TRAIN_PATH, sample_index)
            reconstructed = legacy[0].astype(np.float64) + 1j * legacy[1].astype(np.float64)
            np.testing.assert_allclose(reconstructed, compound, rtol=1e-6, atol=1e-6)


def test_complex_rms_normalization_preserves_shape_and_unit_power():
    signal, raw_rms = read_signal(TRAIN_PATH, 0, normalization="complex_rms")

    assert signal.shape == (2, 3000)
    assert signal.dtype == np.float32
    assert raw_rms > 0.0
    normalized_power = np.mean(signal[0] ** 2 + signal[1] ** 2)
    assert normalized_power == pytest.approx(1.0, rel=1e-5, abs=1e-6)


def test_complex_rms_normalization_handles_zero_signal():
    signal = np.zeros((2, 16), dtype=np.float32)
    normalized, raw_rms = complex_rms_normalize(signal)
    assert raw_rms == 0.0
    assert np.array_equal(normalized, signal)
    assert np.isfinite(normalized).all()


def test_dataset_returns_tensors_and_clears_handle_when_pickled():
    labels = read_labels(TRAIN_PATH)
    valid_indices = np.flatnonzero(valid_sample_mask(labels, "four"))[:3]
    dataset = V73SignalDataset(
        TRAIN_PATH,
        indices=valid_indices,
        class_mode="four",
        normalization="complex_rms",
        include_parameters=True,
    )

    item = dataset[0]
    assert item["signal"].shape == (2, 3000)
    assert item["signal"].dtype == torch.float32
    assert item["label"].shape == (4,)
    assert item["label"].dtype == torch.float32
    assert item["parameter_values"].shape == (79,)
    assert item["parameter_mask"].dtype == torch.bool
    assert item["sample_index"] == int(valid_indices[0])
    assert item["raw_rms"] > 0.0
    assert dataset._handle is not None

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._handle is None
    assert restored[1]["sample_index"] == int(valid_indices[1])
    restored.close()
    dataset.close()


def test_reader_rejects_bad_index_and_unknown_normalization():
    with pytest.raises(IndexError):
        read_signal(TRAIN_PATH, 25200)
    with pytest.raises(ValueError, match="normalization"):
        read_signal(TRAIN_PATH, 0, normalization="minmax")
