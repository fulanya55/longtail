import numpy as np

from cj0724.oracle_swept_isolation_experiment import select_fpr_thresholds


def test_select_fpr_thresholds_respects_cap_on_calibration_data():
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0]], dtype=np.uint8)
    probability = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.9], [0.2, 0.8], [0.1, 0.1]])
    thresholds = select_fpr_thresholds(labels, probability, 0.0)
    predicted = probability >= thresholds[None, :]
    for index in range(2):
        assert predicted[labels[:, index] == 0, index].mean() == 0.0
        assert predicted[labels[:, index] == 1, index].mean() == 1.0
