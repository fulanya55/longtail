import numpy as np

from cj0724.b4_matched_hard_negatives import (
    VARIANTS,
    binary_metrics,
    companion_signatures,
    hard_negative_weights,
)


def test_companion_signature_ignores_target_bit():
    labels = np.asarray([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    signature = companion_signatures(labels, 1)
    assert signature[0] == signature[1]
    assert signature[0] != signature[2]


def test_all_hard_negative_weights_are_finite_and_class_balanced():
    labels = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]] * 3, dtype=np.uint8)
    e5 = np.linspace(0.05, 0.95, len(labels))
    physics = e5[::-1]
    powers = np.repeat([10, 30, 50], 4)
    for variant in VARIANTS:
        weights = hard_negative_weights(labels, e5, physics, powers, 0, variant, 7)
        assert np.isfinite(weights).all()
        assert (weights > 0).all()
        positive = labels[:, 0].astype(bool)
        assert np.isclose(weights[positive].sum(), weights[~positive].sum())


def test_binary_metrics_reports_expected_fpr_and_recall():
    result = binary_metrics(np.asarray([1, 1, 0, 0]), np.asarray([0.9, 0.4, 0.8, 0.1]), 0.5)
    assert result["recall"] == 0.5
    assert result["fpr"] == 0.5
