import numpy as np

from cj0724.p2q_high_lag_strongest_development import BOUNDARY_SECONDS, high_lag_strongest


def test_high_lag_rule_preserves_low_and_replaces_high() -> None:
    legacy = np.asarray([10e-6, 20e-6, 8e-6])
    strongest = np.asarray([70e-6, BOUNDARY_SECONDS, 120e-6])
    result = high_lag_strongest(legacy, strongest)
    np.testing.assert_allclose(result, [10e-6, BOUNDARY_SECONDS, 120e-6])


def test_high_lag_rule_rejects_shape_mismatch() -> None:
    try:
        high_lag_strongest(np.zeros(2), np.zeros(3))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
