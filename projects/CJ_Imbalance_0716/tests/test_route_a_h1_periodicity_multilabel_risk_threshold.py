import numpy as np

from experiments.run_route_a_h1_periodicity_multilabel_risk_threshold import (
    _fit_risk_model,
    _risk_scores,
)


def test_risk_model_uses_only_non_isrj_probabilities():
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(0.0, 1.0, size=(240, 6))
    targets = np.zeros((240, 6), dtype=np.uint8)
    probabilities[:, 5] = 0.15 + 0.7 * probabilities[:, 0]
    model, receipt = _fit_risk_model(probabilities, targets, 0.5)
    score_a = _risk_scores(model, probabilities)
    changed = probabilities.copy()
    changed[:, 5] = 1.0 - changed[:, 5]
    score_b = _risk_scores(model, changed)
    assert np.allclose(score_a, score_b)
    assert receipt["feature_labels"] == ["FMB", "MFT", "SawFM", "SinFM", "VFT"]

