from __future__ import annotations

import numpy as np

from experiments.analyze_route_a_isrj_frozen_colabel_audit import _metric, _strata


def test_strata_cover_each_sample_once_in_joint_partition() -> None:
    targets = np.zeros((8, 6), dtype=np.uint8)
    targets[:, 3] = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    targets[:, 4] = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = _strata(targets)
    joint = sum(groups[f"mft_{mft}__vft_{vft}"].astype(int) for mft in (0, 1) for vft in (0, 1))
    assert np.array_equal(joint, np.ones(8, dtype=int))


def test_metric_uses_weak_positive_not_strong_positive_for_ranking() -> None:
    targets = np.zeros((24, 6), dtype=np.uint8)
    targets[:12, 5] = 1
    weak = np.zeros_like(targets, dtype=bool)
    weak[:10, 5] = True
    probability = np.zeros((24, 6), dtype=np.float64)
    probability[:10, 5] = 0.9
    probability[10:12, 5] = 0.1
    prediction = probability >= 0.5
    metric = _metric(targets, weak, probability, prediction, np.ones(24, dtype=bool))
    assert metric["weak_positive_count"] == 10
    assert metric["strong_positive_count"] == 2
    assert metric["weak_vs_negative_auroc"] == 1.0
