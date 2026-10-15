import numpy as np

from cj0724.p2u_vft_wide_grid_ood import regression_metrics


def test_regression_metrics_perfect_prediction() -> None:
    target = np.asarray([-5000.0, 1000.0, 12000.0])
    result = regression_metrics(target.copy(), target)
    assert result["mae_hz"] == 0.0
    assert result["mae_reduction_vs_constant"] == 1.0
    assert result["pearson"] > 0.999


def test_regression_metrics_constant_prediction_has_zero_correlation() -> None:
    result = regression_metrics(np.full(4, 4000.0), np.asarray([1000.0, 2000.0, 3000.0, 4000.0]))
    assert result["pearson"] == 0.0
