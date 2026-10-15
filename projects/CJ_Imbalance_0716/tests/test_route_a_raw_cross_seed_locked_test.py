import numpy as np

from experiments.run_route_a_raw_cross_seed_locked_test_once import (
    _metrics,
    _sigmoid,
)


def test_sigmoid_matches_frozen_platt_formula():
    logits = np.asarray([-2.0, 0.0, 2.0])
    coefficient = 0.8
    intercept = -0.3
    expected = 1.0 / (1.0 + np.exp(-(coefficient * logits + intercept)))
    assert np.allclose(_sigmoid(coefficient * logits + intercept), expected)


def test_metrics_respects_fixed_per_label_thresholds_and_weak_mask():
    probability = np.asarray([
        [0.9, 0.8, 0.1, 0.1, 0.1, 0.1],
        [0.7, 0.4, 0.1, 0.1, 0.1, 0.1],
        [0.2, 0.7, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.3, 0.9, 0.9, 0.9, 0.9],
    ])
    targets = np.asarray([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)
    weak = np.zeros_like(targets, dtype=bool)
    weak[0, 1] = True
    thresholds = np.asarray([0.5, 0.6, 0.5, 0.5, 0.5, 0.5])
    metrics, prediction = _metrics(probability, targets, thresholds, weak)
    assert np.array_equal(prediction, targets)
    assert metrics["per_label"]["SawFM"]["weak_recall"] == 1.0
    assert metrics["exact_match"] == 1.0
