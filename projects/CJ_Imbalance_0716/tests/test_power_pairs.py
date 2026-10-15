from pathlib import Path

import numpy as np

from cjlt.data.power_pairs import (
    build_power_pair_index,
    discover_power_sweep_train_files,
    load_level_identity,
)


POWER_ROOT = Path(
    r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1"
)


def test_discover_power_sweep_train_files_reads_all_nine_levels():
    files = discover_power_sweep_train_files(POWER_ROOT)
    assert sorted(files) == [10, 15, 20, 25, 30, 35, 40, 45, 50]
    assert all(path.name == "closedset_6class_train.mat" for path in files.values())
    assert all(path.is_file() for path in files.values())


def test_receiver_seed_and_labels_are_paired_across_levels():
    files = discover_power_sweep_train_files(POWER_ROOT)
    seed10, labels10 = load_level_identity(files[10])
    seed30, labels30 = load_level_identity(files[30])
    seed50, labels50 = load_level_identity(files[50])

    assert len(np.unique(seed10)) == 25200
    assert np.array_equal(seed10, seed30)
    assert np.array_equal(seed10, seed50)
    assert np.array_equal(labels10, labels30)
    assert np.array_equal(labels10, labels50)


def test_build_power_pair_index_tracks_label_specific_strength_direction():
    # 0: FMB only, 2000: ISRJ only, 24800: all six classes.
    index = build_power_pair_index(
        POWER_ROOT,
        levels=[10, 30, 50],
        reference_level=30,
        sample_indices=[0, 2000, 24800],
    )

    assert index.reference_level == 30
    assert index.levels == (10, 30, 50)
    assert len(index.records) == 6
    assert len(index.sha256) == 64

    by_key = {
        (record.reference_index, record.paired_level): record
        for record in index.records
    }

    fmb_only_10 = by_key[(0, 10)]
    assert fmb_only_10.labels.tolist() == [1, 0, 0, 0, 0, 0]
    assert fmb_only_10.stronger_level_by_label.tolist() == [0, 0, 0, 0, 0, 0]

    isrj_only_10 = by_key[(2000, 10)]
    assert isrj_only_10.labels.tolist() == [0, 0, 0, 0, 0, 1]
    assert isrj_only_10.stronger_level_by_label.tolist() == [0, 0, 0, 0, 0, 30]

    all_classes_10 = by_key[(24800, 10)]
    assert all_classes_10.stronger_level_by_label.tolist() == [10, 10, 10, 30, 30, 30]
    assert all_classes_10.relative_power_gap == 20.0

    all_classes_50 = by_key[(24800, 50)]
    assert all_classes_50.stronger_level_by_label.tolist() == [30, 30, 30, 50, 50, 50]
    assert all_classes_50.relative_power_gap == 20.0


def test_power_pair_index_is_reproducible():
    first = build_power_pair_index(
        POWER_ROOT,
        levels=[10, 30, 50],
        reference_level=30,
        sample_indices=[0, 2000, 24800],
    )
    repeat = build_power_pair_index(
        POWER_ROOT,
        levels=[10, 30, 50],
        reference_level=30,
        sample_indices=[0, 2000, 24800],
    )
    assert first.sha256 == repeat.sha256
