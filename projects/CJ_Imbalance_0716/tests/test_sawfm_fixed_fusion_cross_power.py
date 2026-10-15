import numpy as np

from experiments.run_route_a_sawfm_fixed_fusion_cross_power import (
    _fuse,
    _power_component_matrix,
    _stratified_metrics,
)


def test_fixed_fusion_changes_only_sawfm_logit():
    base = np.arange(18, dtype=np.float32).reshape(3, 6)
    expert = np.asarray([-2.0, 1.0, 4.0], dtype=np.float32)
    fused = _fuse(base, expert, 0.80)
    assert np.allclose(fused[:, 0], base[:, 0])
    assert np.allclose(fused[:, 2:], base[:, 2:])
    assert np.allclose(fused[:, 1], 0.20 * base[:, 1] + 0.80 * expert)


def test_power_component_matrix_uses_fixed_30_and_swept_level():
    labels = np.asarray([[1, 1, 0, 1, 0, 1]], dtype=np.uint8)
    values = _power_component_matrix(labels, 45)
    assert np.allclose(values[0, [0, 1]], [30.0, 30.0])
    assert np.isnan(values[0, 2])
    assert np.allclose(values[0, [3, 5]], [45.0, 45.0])
    assert np.isnan(values[0, 4])


def test_stratified_metrics_aligns_matched_pairs():
    labels = np.asarray(
        [
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    positive = np.asarray([3.0, 2.0, 4.0, 1.0])
    negative = np.asarray([-1.0, 0.0, -2.0, 0.5])
    rows = _stratified_metrics(positive, negative, labels, min_pairs=2)
    assert rows
    assert all(row["pair_count"] >= 2 for row in rows)
    assert all(0.0 <= row["auroc"] <= 1.0 for row in rows)
