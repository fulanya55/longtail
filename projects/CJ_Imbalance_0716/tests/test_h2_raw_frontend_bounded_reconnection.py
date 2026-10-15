import numpy as np

from experiments.run_route_a_h2_raw_frontend_bounded_reconnection import (
    _bounded_residual,
    _rescue_residual,
)


def test_signed_residual_is_hard_bounded_and_confidence_damped():
    h2 = np.asarray([0.0, 1.0, 5.0, -5.0])
    raw = np.asarray([10.0, 10.0, 10.0, -10.0])
    fused = _bounded_residual(h2, raw, 0.20)
    delta = fused - h2
    assert np.max(np.abs(delta)) <= 0.20 + 1e-12
    assert abs(delta[2]) < abs(delta[1]) < abs(delta[0])


def test_rescue_only_changes_low_h2_high_raw_samples():
    h2 = np.asarray([-2.0, -2.0, 0.5])
    raw = np.asarray([3.0, 0.0, 3.0])
    fused = _rescue_residual(h2, raw, gate=0.0, raw_threshold=0.80, alpha=0.20)
    assert np.allclose(fused - h2, [0.20, 0.0, 0.0])
