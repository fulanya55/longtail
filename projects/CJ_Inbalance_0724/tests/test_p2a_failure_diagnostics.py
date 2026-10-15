import numpy as np

from cj0724.p2a_failure_diagnostics import per_parameter_nmae


def test_per_parameter_nmae_uses_predicted_positive_valid_entries_only():
    labels = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=np.uint8)
    mask = np.zeros((2, 6, 2), dtype=np.uint8)
    mask[:, 0, 0] = 1
    probabilities = np.array([[0.9, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]])
    thresholds = np.full(6, 0.5)
    prediction = np.zeros((2, 6, 2), dtype=np.float32)
    target = np.zeros((2, 6, 2), dtype=np.float32)
    prediction[0, 0, 0] = 2.0
    prediction[1, 0, 0] = 100.0
    scale = np.ones((6, 2), dtype=np.float32)
    result = per_parameter_nmae(labels, mask, probabilities, thresholds, prediction, target, scale)
    assert result[0, 0] == 2.0
    assert np.isnan(result[0, 1])
