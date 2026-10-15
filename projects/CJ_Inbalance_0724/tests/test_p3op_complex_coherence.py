import numpy as np

from cj0724.p3op_complex_coherence import complex_coherence_tokens


def test_p3op_keeps_event_locations_and_hides_coherence_from_control():
    channels = np.zeros((1, 8, 32), dtype=np.float32)
    channels[:, 0] = 1.0; channels[:, 1, [4, 15]] = [1.0, 0.5]; channels[:, 2] = .25; channels[:, 3] = np.linspace(0.0, 1.0, 32)
    channels[:, 4] = .7; channels[:, 5] = .2; channels[:, 6] = .9; channels[:, 7] = .8
    control, candidate = complex_coherence_tokens(channels, False), complex_coherence_tokens(channels, True)
    assert control.shape == candidate.shape == (1, 16, 8)
    assert np.array_equal(control[:, :, :4], candidate[:, :, :4])
    assert not np.any(control[:, :, 4:])
    assert np.allclose(candidate[0, :2, 4:], [.7, .2, .9, .8])
    assert not np.any(candidate[0, 2:, 4:])


def test_p3op_recovers_peak_indices_after_fit_position_normalization():
    channels = np.zeros((1, 8, 32), dtype=np.float32)
    channels[:, 0] = 1.0; channels[:, 1, [4, 15]] = [1.0, .5]; channels[:, 2] = .25; channels[:, 3] = np.linspace(0.0, 1.0, 32)
    channels[:, 4, 4], channels[:, 4, 15] = .25, .75
    normalized = channels.copy(); normalized[:, 3] = (normalized[:, 3] - .5) / .2
    candidate = complex_coherence_tokens(normalized, True, position_mean=.5, position_scale=.2)
    assert np.allclose(candidate[0, :2, 4], [.25, .75])
