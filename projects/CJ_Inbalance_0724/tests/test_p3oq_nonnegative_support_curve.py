import numpy as np

from cj0724.p3oq_nonnegative_support_curve import nonnegative_support_curve, support_curve_tokens


def test_p3oq_curve_is_deterministic_monotone_and_hidden_from_control():
    impulse = np.asarray([[4+0j, 3+1j, 2+0j, -5+0j, 1+0j]], dtype=np.complex128)
    curve = nonnegative_support_curve(impulse)
    assert curve.shape == (1, 8, 4)
    assert np.all(np.diff(curve[0, :, 0]) <= 1e-7)
    assert np.all(np.diff(curve[0, :, 1]) >= -1e-7)
    channels = np.zeros((1, 4, 32), dtype=np.float32); channels[:, 1, [3, 9]] = [1.0, .5]; channels[:, 3] = np.linspace(0, 1, 32)
    control, candidate = support_curve_tokens(channels, curve, False), support_curve_tokens(channels, curve, True)
    assert control.shape == candidate.shape == (1, 16, 8)
    assert np.array_equal(control[:, :, :4], candidate[:, :, :4])
    assert not np.any(control[:, :, 4:])
    assert np.allclose(candidate[:, :8, 4:], curve)
