import numpy as np

from cj0724.p1k_finalize_saved_predictions import fast_metric


def _record():
    labels = np.zeros((4, 6), dtype=np.uint8)
    labels[:, 0] = 1
    probability = np.zeros((4, 6), dtype=np.float32)
    probability[:, 0] = np.array([0.9, 0.9, 0.1, 0.9])
    prediction = np.zeros((4, 6, 2), dtype=np.float32)
    target = np.zeros_like(prediction)
    prediction[:, 0, 0] = 0.5
    mask = np.zeros_like(prediction, dtype=np.uint8)
    mask[:, 0, 0] = 1
    return {
        "labels": labels, "probabilities": probability, "thresholds": np.full(6, 0.5),
        "prediction": prediction, "target": target, "mask": mask, "scale": np.ones((6, 2)),
    }


def test_fast_metric_matches_expected_gate_and_miss_penalty():
    record = _record()
    assert np.isclose(fast_metric(record, "predicted_gate_parameter_nmae"), 0.5)
    assert np.isclose(fast_metric(record, "miss_penalty_parameter_nmae"), 0.625)
