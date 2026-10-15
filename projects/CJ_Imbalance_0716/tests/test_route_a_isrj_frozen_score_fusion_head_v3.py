import numpy as np

from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import (
    ISRJ_INDEX,
    _fusion_features,
    _replace_isrj,
    _sample_weights,
)


def test_fusion_features_have_fixed_four_columns_and_finite_values():
    logits = np.zeros((3, 6), dtype=np.float32)
    logits[:, ISRJ_INDEX] = [-2.0, 0.0, 2.0]
    features = _fusion_features(logits, np.asarray([-1.0, 0.0, 1.0], dtype=np.float32))
    assert features.shape == (3, 4)
    assert np.isfinite(features).all()
    assert np.allclose(features[:, 0], logits[:, ISRJ_INDEX])


def test_sample_weights_only_upweight_weak_isrj_positives():
    targets = np.zeros((4, 6), dtype=np.uint8)
    targets[:3, ISRJ_INDEX] = 1
    weak = np.zeros((4, 6), dtype=bool)
    weak[[0, 2], ISRJ_INDEX] = True
    weights = _sample_weights(targets, weak, 4.0)
    assert weights.tolist() == [4.0, 1.0, 4.0, 1.0]


def test_replace_isrj_preserves_all_other_logits():
    logits = np.arange(24, dtype=np.float32).reshape(4, 6)
    output = _replace_isrj(logits, np.asarray([1, 2, 3, 4], dtype=np.float32))
    assert np.array_equal(output[:, :ISRJ_INDEX], logits[:, :ISRJ_INDEX])
    assert output[:, ISRJ_INDEX].tolist() == [1.0, 2.0, 3.0, 4.0]

