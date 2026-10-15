import numpy as np
import pytest

from cj0724.p2h_sawfm_affine_calibration import apply_calibrators, fit_calibrators


def test_bounded_affine_recovers_linear_mapping():
    prediction = np.linspace(20.0e-6, 60.0e-6, 100)
    target = 1.1 * prediction + 5.0e-6
    coefficients = fit_calibrators(prediction, target)
    assert coefficients["affine_a"] == pytest.approx(1.1, rel=1.0e-4)
    assert coefficients["affine_b"] == pytest.approx(5.0e-6, rel=1.0e-4)
    calibrated = apply_calibrators(prediction, coefficients)["C2_bounded_affine"]
    assert np.max(np.abs(calibrated - target)) < 1.0e-10
