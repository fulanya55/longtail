import numpy as np

from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import (
    ISRJ_INDEX, _bounded_isrj_residual, _clipped_logit,
)


def test_isrj_residual_changes_only_isrj_and_is_bounded():
    logits = np.zeros((4, 6), dtype=np.float32)
    output, residual, uncertainty = _bounded_isrj_residual(
        logits, np.asarray([-10.0, -1.0, 1.0, 10.0], dtype=np.float32))
    other = [index for index in range(6) if index != ISRJ_INDEX]
    np.testing.assert_array_equal(output[:, other], logits[:, other])
    assert np.max(np.abs(residual)) <= .5 + 1e-7
    np.testing.assert_allclose(uncertainty, 1.0)


def test_clipped_logit_is_finite_and_monotonic():
    values = _clipped_logit(np.asarray([0.0, .25, .5, .75, 1.0]))
    assert np.isfinite(values).all()
    assert np.all(np.diff(values) > 0)

