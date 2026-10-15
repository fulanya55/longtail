import numpy as np

from experiments.run_route_a_cooccurrence_safety_head_v3 import _features, _sample_weights, _select_threshold


def test_features_are_nine_dimensional_and_alignment_sensitive():
    logits = np.arange(24, dtype=np.float32).reshape(4, 6) / 10.0
    aligned = _features(logits, 3)
    shuffled = _features(logits, 3, np.array([3, 2, 1, 0]))
    assert aligned.shape == (4, 9)
    assert np.array_equal(aligned[:, 0], shuffled[:, 0])
    assert not np.array_equal(aligned[:, 1:], shuffled[:, 1:])


def test_order_balancing_increases_rare_high_order_negative_weight():
    labels = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=np.uint8)
    weak = np.zeros_like(labels, dtype=bool)
    weights = _sample_weights(labels, weak, 5, True, 4.0)
    assert weights[0] > weights[2]


def test_threshold_obeys_overall_and_hard_negative_constraints():
    target = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    weak = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    probability = np.array([0.8, 0.9, 0.7, 0.6, 0.2, 0.1])
    order = np.array([2, 3, 5, 4, 2, 1])
    threshold, receipt = _select_threshold(target, weak, probability, order, 0.25, 0.0)
    prediction = probability >= threshold
    assert receipt["feasible"]
    assert prediction[target == 0].mean() <= 0.25
    assert prediction[(target == 0) & (order >= 4)].mean() <= 0.0
