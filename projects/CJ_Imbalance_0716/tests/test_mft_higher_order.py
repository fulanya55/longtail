from __future__ import annotations

import numpy as np

from cjlt.models.mft_higher_order import circular_autocorrelation, dft_phase_surrogate, segmented_bicoherence


def test_phase_surrogate_preserves_dft_magnitude_and_circular_autocorrelation() -> None:
    rng = np.random.default_rng(7); value = rng.standard_normal(256) + 1j * rng.standard_normal(256); control = dft_phase_surrogate(value, seed=8)
    assert np.allclose(np.abs(np.fft.fft(value)), np.abs(np.fft.fft(control)), rtol=1e-10, atol=1e-10)
    assert np.allclose(circular_autocorrelation(value), circular_autocorrelation(control), rtol=1e-9, atol=1e-9)


def test_segmented_bicoherence_is_finite() -> None:
    rng = np.random.default_rng(9); value = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    assert np.isfinite(segmented_bicoherence(value))
