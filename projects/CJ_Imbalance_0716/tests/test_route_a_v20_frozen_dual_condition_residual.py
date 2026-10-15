import numpy as np

from experiments.run_route_a_v20_frozen_dual_condition_residual import GRID, _prob


def test_dual_residual_grid_and_fixed_calibration_contract():
    assert np.allclose(GRID, np.arange(0., 2.01, .2))
    logits = np.zeros((2, 6), dtype=np.float32)
    calibration = {"parameters": [{"coefficient": 1., "intercept": 0.} for _ in range(6)]}
    probability, prediction = _prob(logits, calibration, np.full(6, .5))
    assert np.allclose(probability, .5)
    assert prediction.dtype == bool
    assert prediction.all()
