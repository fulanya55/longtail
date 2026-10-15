from pathlib import Path

import numpy as np
import pytest

from cjlt.data.longtail_manifest import (
    build_combo_longtail,
    build_label_longtail,
)
from cjlt.data.registry import select_labels, valid_sample_mask
from cjlt.data.split_builder import build_stratified_roles
from cjlt.data.v73_dataset import read_labels


TRAIN_PATH = Path(
    r"C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"
)


@pytest.fixture(scope="module")
def four_class_optimization():
    labels6 = read_labels(TRAIN_PATH)
    labels4 = select_labels(labels6, "four")
    eligible = np.flatnonzero(valid_sample_mask(labels6, "four"))
    split = build_stratified_roles(labels4, eligible, seed=2026)
    return labels4, split.optimization, split.sha256


def test_combo_longtail_matches_exponential_targets(four_class_optimization):
    labels4, optimization, split_hash = four_class_optimization
    manifest = build_combo_longtail(
        labels4,
        optimization,
        imbalance_factor=10,
        seed=2026,
        source_split_sha256=split_hash,
    )

    counts = np.asarray(list(manifest.combo_counts.values()), dtype=np.int64)
    assert manifest.protocol == "combo_lt"
    assert len(counts) == 15
    assert counts.max() == 280
    assert counts.min() == 28
    assert len(manifest.selected_indices) == int(counts.sum())
    assert len(np.unique(manifest.selected_indices)) == len(manifest.selected_indices)
    assert manifest.imbalance_factor == 10.0
    assert manifest.source_split_sha256 == split_hash
    assert len(manifest.sha256) == 64


def test_combo_longtail_is_reproducible_but_seed_changes_selected_samples(
    four_class_optimization,
):
    labels4, optimization, split_hash = four_class_optimization
    first = build_combo_longtail(labels4, optimization, 30, 2026, split_hash)
    repeat = build_combo_longtail(labels4, optimization, 30, 2026, split_hash)
    changed = build_combo_longtail(labels4, optimization, 30, 2027, split_hash)

    assert first.sha256 == repeat.sha256
    assert np.array_equal(first.selected_indices, repeat.selected_indices)
    assert first.sha256 != changed.sha256
    assert not np.array_equal(first.selected_indices, changed.selected_indices)


def test_label_longtail_reaches_requested_marginals_within_tolerance(
    four_class_optimization,
):
    labels4, optimization, split_hash = four_class_optimization
    target = np.asarray([1000, 800, 600, 400], dtype=np.int64)
    manifest = build_label_longtail(
        labels4,
        optimization,
        target_label_counts=target,
        seed=2026,
        tolerance=0.05,
        source_split_sha256=split_hash,
    )

    achieved = np.asarray(manifest.achieved_label_counts, dtype=np.int64)
    error = np.abs(achieved - target) / target
    assert manifest.protocol == "label_lt"
    assert np.array_equal(manifest.target_label_counts, target)
    assert np.all(error <= 0.05)
    assert np.allclose(manifest.label_count_error, error)
    assert len(np.unique(manifest.selected_indices)) == len(manifest.selected_indices)
    assert sum(manifest.combo_counts.values()) == len(manifest.selected_indices)


def test_label_longtail_reports_infeasible_target_instead_of_claiming_success(
    four_class_optimization,
):
    labels4, optimization, split_hash = four_class_optimization
    with pytest.raises(ValueError, match="exceeds available"):
        build_label_longtail(
            labels4,
            optimization,
            target_label_counts=[3000, 800, 600, 400],
            seed=2026,
            tolerance=0.05,
            source_split_sha256=split_hash,
        )


def test_label_longtail_milp_is_reproducible(four_class_optimization):
    labels4, optimization, split_hash = four_class_optimization
    target = np.asarray([800, 600, 450, 300], dtype=np.int64)
    first = build_label_longtail(
        labels4, optimization, target, 2026, 0.10, split_hash
    )
    repeat = build_label_longtail(
        labels4, optimization, target, 2026, 0.10, split_hash
    )
    assert first.sha256 == repeat.sha256
    assert np.array_equal(first.selected_indices, repeat.selected_indices)


def test_label_longtail_can_fix_total_sample_count(four_class_optimization):
    labels4, optimization, split_hash = four_class_optimization
    target = np.asarray([800, 650, 520, 400], dtype=np.int64)
    manifest = build_label_longtail(
        labels4,
        optimization,
        target_label_counts=target,
        seed=2026,
        tolerance=0.05,
        source_split_sha256=split_hash,
        target_sample_count=1200,
    )

    assert manifest.target_sample_count == 1200
    assert manifest.achieved_sample_count == 1200
    assert manifest.sample_count_error == 0.0
    assert len(manifest.selected_indices) == 1200
    assert manifest.to_dict()["target_sample_count"] == 1200


def test_label_longtail_rejects_invalid_sample_budget(four_class_optimization):
    labels4, optimization, split_hash = four_class_optimization
    with pytest.raises(ValueError, match="exceeds eligible"):
        build_label_longtail(
            labels4,
            optimization,
            target_label_counts=[800, 650, 520, 400],
            seed=2026,
            tolerance=0.05,
            source_split_sha256=split_hash,
            target_sample_count=len(optimization) + 1,
        )
