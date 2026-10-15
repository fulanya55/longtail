import numpy as np

from cj0724.p2y_isrj_phase_reset_isolation import waveform_cosine


def test_waveform_cosine_is_phase_invariant() -> None:
    rng = np.random.default_rng(20260801)
    iq = rng.normal(size=(4, 2, 3000)).astype(np.float32)
    signal = iq[:, 0] + 1j * iq[:, 1]
    rotated = signal * np.exp(1j * 0.7)
    rotated_iq = np.stack((rotated.real, rotated.imag), axis=1).astype(np.float32)
    assert np.allclose(waveform_cosine(rotated_iq, iq), 1.0, atol=1.0e-6)
