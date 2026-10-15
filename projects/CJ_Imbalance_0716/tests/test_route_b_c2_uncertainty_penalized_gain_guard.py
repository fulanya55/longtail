import numpy as np
import pytest

from experiments.run_route_b_c2_uncertainty_penalized_gain_guard import (
    _risk_adjusted_score,
)


def test_risk_adjusted_score_interpolates_mean_and_lower_bound():
    mean = np.asarray([1.0, 2.0])
    lower = np.asarray([-1.0, 1.0])
    np.testing.assert_allclose(
        _risk_adjusted_score(mean, lower, 0.10),
        np.asarray([0.8, 1.9]),
    )


def test_risk_adjusted_score_rejects_invalid_weight_or_shape():
    with pytest.raises(ValueError, match="risk_weight"):
        _risk_adjusted_score(np.ones(2), np.ones(2), 1.1)
    with pytest.raises(ValueError, match="match"):
        _risk_adjusted_score(np.ones(2), np.ones(3), 0.1)
