import numpy as np

from cj0724.p6a_predicted_gate_evaluate import _metrics


def test_predicted_gate_metrics_accept_float_dataset_labels_and_keep_miss_penalty():
    record = {
        "sample_indices": np.array([0, 1]),
        "labels": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "probabilities": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        "prediction": np.array([[[1.2], [0.0]], [[4.0], [6.0]]], dtype=np.float32),
        "target": np.array([[[1.0], [np.nan]], [[3.0], [5.0]]], dtype=np.float32),
        "mask": np.array([[[True], [False]], [[True], [True]]]),
    }
    metrics = _metrics(record, np.array([0.5, 0.5], dtype=np.float32), np.ones((2, 1), dtype=np.float32))
    assert metrics["valid_parameter_cells"] == 3
    assert metrics["gated_parameter_cells"] == 2
    assert np.isfinite(metrics["predicted_gate_parameter_nmae"])
    assert metrics["miss_penalty_parameter_nmae"] > metrics["predicted_gate_parameter_nmae"]
