import numpy as np

from experiments.run_route_b_c2_cross_seed_expected_gain_guard import (
    _feature_matrix,
    _identity_exclusion_mask,
)


def test_identity_exclusion_keeps_candidates_and_removes_held_identities():
    identity = np.asarray([1, 2, 3, 4])
    candidate = np.asarray([True, True, False, True])
    actual = _identity_exclusion_mask(identity, np.asarray([2, 4]), candidate)
    np.testing.assert_array_equal(actual, np.asarray([True, False, False, False]))


def test_feature_matrix_has_seven_finite_observable_columns():
    features = _feature_matrix(
        q=np.asarray([0.6, 0.7]),
        c2=np.asarray([1.0, 2.0]),
        local=np.asarray([1.5, 1.0]),
        mean=1.0,
        scale=2.0,
        class_probability=np.asarray([0.8, 0.9]),
        raw_rms=np.asarray([0.0, 2.0]),
        level=np.asarray([10, 50]),
    )
    assert features.shape == (2, 7)
    assert np.isfinite(features).all()
