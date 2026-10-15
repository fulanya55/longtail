import numpy as np

from cj0724.p2o_failure_atlas import metrics, pearson


def test_pearson_and_metrics_detect_harmonic_outlier() -> None:
    target = np.asarray([5.0, 6.0, 7.0, 8.0]) * 1.0e-6
    prediction = target.copy()
    prediction[-1] *= 2.0
    result = metrics(prediction, target)
    assert pearson(target, target) > 0.999
    assert result["max_ae_us"] == 8.0
    assert result["harmonic_like_error_rate"] == 0.25


def test_metrics_are_exact_for_perfect_prediction() -> None:
    target = np.asarray([10.0, 12.0, 14.0]) * 1.0e-6
    result = metrics(target, target)
    assert result["mae_us"] == 0.0
    assert result["pearson"] > 0.999
    assert result["error_ge_0p5us_rate"] == 0.0
