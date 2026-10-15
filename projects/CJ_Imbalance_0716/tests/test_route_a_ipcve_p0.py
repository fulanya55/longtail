import numpy as np

from experiments.run_route_a_ipcve_p0_frozen_observability import (
    PROJECTION_DIM,
    _bootstrap_mean_ci,
    _projection,
)


def test_fixed_source_projection_is_deterministic_and_orthonormal():
    first = _projection("global_128", 128)
    second = _projection("global_128", 128)
    assert first.shape == (128, PROJECTION_DIM)
    assert np.array_equal(first, second)
    assert np.allclose(first.T @ first, np.eye(PROJECTION_DIM), atol=1e-5)


def test_seed_bootstrap_ci_is_finite_and_ordered():
    lower, upper = _bootstrap_mean_ci(np.array([0.01, 0.02, 0.03, 0.04, 0.05]))
    assert np.isfinite(lower) and np.isfinite(upper)
    assert lower <= upper
