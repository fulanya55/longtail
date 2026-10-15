import numpy as np
import torch

from cj0724.vft_fft import fft_matched_filter_peaks
from cj0724.vft_physics_screen import matched_filter_peaks


def test_fft_matches_direct_matched_filter():
    generator = torch.Generator().manual_seed(7)
    iq = torch.randn(3, 2, 3000, generator=generator)
    frequencies = np.asarray([-8000.0, -1500.0, 0.0, 2500.0, 8000.0], dtype=np.float32)
    expected_peaks, expected_positions = matched_filter_peaks(iq, frequencies)
    actual_peaks, actual_positions = fft_matched_filter_peaks(iq, frequencies)
    assert torch.allclose(actual_peaks, expected_peaks, atol=2e-5, rtol=2e-5)
    assert torch.equal(actual_positions, expected_positions)
