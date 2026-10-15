import numpy as np

from cj0724.p3on_resolution_abstention import ordinal_confidence, resolution_score, selection_threshold


def test_p3on_selection_threshold_uses_selection_not_development_rank():
    score = np.arange(10, dtype=np.float32)
    assert selection_threshold(score, 0.8) == 2.0


def test_p3on_resolution_score_is_observable_and_finite():
    logits = np.zeros((2, 7), dtype=np.float32)
    tokens = np.zeros((2, 16, 4), dtype=np.float32)
    tokens[:, :2, 1] = [[1.0, 0.5], [0.8, 0.4]]
    tokens[:, :2, 3] = [[0.2, 0.5], [0.1, 0.12]]
    assert np.isfinite(ordinal_confidence(logits)).all()
    assert np.isfinite(resolution_score(logits, tokens)).all()
