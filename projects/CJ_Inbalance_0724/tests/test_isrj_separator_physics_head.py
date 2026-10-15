import numpy as np

from cj0724.isrj_separator_physics_head import clipped_logit, feature_view, target_metrics


def test_isrj_feature_views_preserve_expected_dimensions():
    features = np.zeros((4, 310), dtype=np.float32)
    assert feature_view(features, "periodicity_full").shape == (4, 310)
    assert feature_view(features, "envelope_periodicity").shape == (4, 130)


def test_clipped_logit_and_target_metrics_are_finite():
    probability = np.array([0.0, 0.2, 0.8, 1.0])
    assert np.isfinite(clipped_logit(probability)).all()
    metrics = target_metrics(np.array([0, 0, 1, 1]), probability, 0.5)
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0
