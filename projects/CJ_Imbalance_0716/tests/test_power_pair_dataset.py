from pathlib import Path

import numpy as np

from cjlt.data.power_pair_dataset import PowerPairDataset
from cjlt.data.power_pairs import (
    build_role_bound_power_pair_index,
    load_level_identity,
)
from cjlt.data.registry import get_selected_indices


DATA_ROOT = Path(
    r"C:\LJQ_code\干扰数据\复合干扰数据0716"
)
BASE_TRAIN = DATA_ROOT / "closedset_6class_train.mat"
POWER_ROOT = DATA_ROOT / "power_sweep_9level_v1"


def test_role_bound_power_index_maps_base_indices_by_receiver_identity():
    base_indices = [0, 2000, 24800]
    pair_index = build_role_bound_power_pair_index(
        BASE_TRAIN,
        POWER_ROOT,
        base_indices=base_indices,
        levels=[10, 30, 50],
        reference_level=30,
    )
    base_seeds, _ = load_level_identity(BASE_TRAIN)

    assert pair_index.base_indices == tuple(base_indices)
    assert pair_index.reference_indices == tuple(base_indices)
    assert {record.identity for record in pair_index.records} == {
        int(base_seeds[index]) for index in base_indices
    }
    assert len(pair_index.records) == 6


def test_power_pair_dataset_returns_selected_labels_and_direction_metadata():
    pair_index = build_role_bound_power_pair_index(
        BASE_TRAIN,
        POWER_ROOT,
        base_indices=[24800],
        levels=[10, 30],
        reference_level=30,
    )
    dataset = PowerPairDataset(
        POWER_ROOT,
        pair_index.records,
        class_mode="four",
        normalization="complex_rms",
    )

    item = dataset[0]

    assert item["reference_signal"].shape == (2, 3000)
    assert item["paired_signal"].shape == (2, 3000)
    assert item["label"].tolist() == [1.0, 1.0, 1.0, 1.0]
    selected = get_selected_indices("four")
    expected = pair_index.records[0].stronger_level_by_label[selected]
    assert np.array_equal(item["stronger_level_by_label"].numpy(), expected)
    assert item["reference_level"] == 30
    assert item["paired_level"] == 10
    assert item["identity"] == pair_index.records[0].identity
    dataset.close()
