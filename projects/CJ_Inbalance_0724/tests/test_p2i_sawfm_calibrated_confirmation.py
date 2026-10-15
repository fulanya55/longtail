from cj0724.p2i_sawfm_calibrated_confirmation import COEFFICIENTS


def test_frozen_coefficients_are_bounded():
    assert set(COEFFICIENTS) == {2041, 2042, 2043}
    for slope, intercept in COEFFICIENTS.values():
        assert 0.5 <= slope <= 1.5
        assert -20.0e-6 <= intercept <= 20.0e-6
