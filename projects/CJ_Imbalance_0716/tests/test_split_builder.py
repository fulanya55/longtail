from pathlib import Path

import numpy as np
import pytest

from cjlt.data.registry import select_labels, valid_sample_mask
from cjlt.data.split_builder import (
    RoleCounts,
    build_stratified_roles,
    combo_key,
    group_indices_by_combo,
)
from cjlt.data.v73_dataset import read_labels


TRAIN_PATH = Path(
    r"C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"
)


def test_combo_key_encodes_complete_multihot_label():
    assert combo_key([1, 0, 1, 0]) == "1010"
    with pytest.raises(ValueError, match="binary"):
        combo_key([1, 2, 0])


def test_four_class_split_has_15_combinations_and_expected_role_sizes():
    labels6 = read_labels(TRAIN_PATH)
    eligible = np.flatnonzero(valid_sample_mask(labels6, "four"))
    labels4 = select_labels(labels6, "four")

    split = build_stratified_roles(
        labels4,
        eligible_indices=eligible,
        seed=2026,
        counts=RoleCounts(),
    )

    assert len(group_indices_by_combo(labels4, eligible)) == 15
    assert len(split.optimization) == 4200
    assert len(split.early_stop) == 600
    assert len(split.calibration) == 600
    assert len(split.threshold_selection) == 600
    split.assert_disjoint()


def test_six_class_split_has_63_combinations_and_expected_role_sizes():
    labels6 = read_labels(TRAIN_PATH)
    eligible = np.flatnonzero(valid_sample_mask(labels6, "six"))

    split = build_stratified_roles(labels6, eligible, seed=2026)

    assert len(group_indices_by_combo(labels6, eligible)) == 63
    assert len(split.optimization) == 17640
    assert len(split.early_stop) == 2520
    assert len(split.calibration) == 2520
    assert len(split.threshold_selection) == 2520
    split.assert_disjoint()


def test_split_is_reproducible_and_changes_with_seed():
    labels6 = read_labels(TRAIN_PATH)
    eligible = np.flatnonzero(valid_sample_mask(labels6, "four"))
    labels4 = select_labels(labels6, "four")

    split_a = build_stratified_roles(labels4, eligible, seed=2026)
    split_b = build_stratified_roles(labels4, eligible, seed=2026)
    split_c = build_stratified_roles(labels4, eligible, seed=2027)

    assert split_a.sha256 == split_b.sha256
    assert np.array_equal(split_a.optimization, split_b.optimization)
    assert split_a.sha256 != split_c.sha256
    assert not np.array_equal(split_a.optimization, split_c.optimization)


def test_split_rejects_group_with_insufficient_samples():
    labels = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="requires"):
        build_stratified_roles(
            labels,
            eligible_indices=np.arange(3),
            seed=1,
            counts=RoleCounts(optimization=1, early_stop=1, calibration=1, threshold_selection=1),
        )
