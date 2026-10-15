import numpy as np

from cj0724.p2n_analytic_saw_full_system import attach_analytic_predictions


def test_attach_analytic_predictions_changes_only_saw_positive_rows():
    prediction = np.zeros((3, 6, 2), dtype=np.float32)
    labels = np.zeros((3, 6), dtype=np.uint8)
    labels[[0, 2], 1] = 1
    physical = {
        "sample_index": np.asarray([5, 7]),
        "power": np.asarray([25, 50]),
        "analytic_separated": np.asarray([1.0, 2.0]),
    }
    output = attach_analytic_predictions(
        prediction, labels, np.asarray([5, 6, 7]), np.asarray([25, 35, 50]), physical,
    )
    assert output[0, 1, 1] == 1.0
    assert output[2, 1, 1] == 2.0
    assert output[1, 1, 1] == 0.0
