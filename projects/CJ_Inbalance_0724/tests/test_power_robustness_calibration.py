import numpy as np

from cj0724.power_robustness_calibration import fbeta_thresholds, gate_features


def test_gate_features_are_deployable_from_logits_only():
    logits = np.zeros((4, 6), dtype=np.float32)
    feature = gate_features(logits)
    assert feature.shape == (4, 10)
    assert np.isfinite(feature).all()


def test_fbeta_thresholds_return_one_value_per_class():
    labels = np.tile(np.eye(6, dtype=np.uint8), (3, 1))
    probability = labels * 0.8 + (1 - labels) * 0.2
    thresholds = fbeta_thresholds(labels, probability, beta=2.0)
    assert thresholds.shape == (6,)
    assert np.all((thresholds >= 0.02) & (thresholds <= 0.95))
