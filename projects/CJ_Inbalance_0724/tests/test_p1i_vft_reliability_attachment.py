import numpy as np

from cj0724.p1i_vft_reliability_attachment import (
    candidate_prediction,
    parameter_metrics,
    select_reliability_cutoff,
)


def _payload():
    labels = np.zeros((8, 6), dtype=np.uint8)
    labels[:, 4] = 1
    return {
        "labels": labels,
        "probability": np.tile(np.array([[0, 0, 0, 0, 0.9, 0]], dtype=np.float32), (8, 1)),
        "valid": np.ones(8, dtype=bool),
        "target": np.arange(8, dtype=np.float32) * 100 + 1000,
        "base_prediction": np.full(8, 2000, dtype=np.float32),
        "expert_prediction": np.arange(8, dtype=np.float32) * 100 + 1010,
        "expert_score": np.linspace(0.1, 0.9, 8).astype(np.float32),
    }


def test_candidate_replaces_only_accepted_rows():
    payload = _payload()
    prediction, accept = candidate_prediction(payload, 0.5)
    assert np.array_equal(prediction[~accept], payload["base_prediction"][~accept])
    assert np.array_equal(prediction[accept], payload["expert_prediction"][accept])


def test_parameter_metrics_include_miss_penalty_and_worst_combo():
    payload = _payload()
    metrics = parameter_metrics(payload, payload["expert_prediction"], 0.5, 1000.0)
    assert metrics["predicted_gate_nmae"] < 0.02
    assert metrics["miss_penalty_nmae"] < 0.02
    assert np.isfinite(metrics["worst_combo_predicted_gate_nmae"])


def test_selection_returns_a_frozen_candidate_table():
    result = select_reliability_cutoff(_payload(), 0.5, 1000.0)
    assert "chosen" in result
    assert len(result["candidates"]) >= 2
