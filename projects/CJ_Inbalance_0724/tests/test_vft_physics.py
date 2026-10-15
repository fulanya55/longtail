import numpy as np
import torch

from cj0724.vft_physics_screen import (
    CHIRP_RATE_HZ_PER_S,
    FS_HZ,
    matched_filter_peaks,
)


def test_vft_filter_bank_recovers_synthetic_doppler():
    frequencies = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)
    time = torch.arange(1000, dtype=torch.float32) / FS_HZ
    phase = torch.pi * CHIRP_RATE_HZ_PER_S * time.square() + 2 * torch.pi * 5000 * time
    signal = torch.zeros(1, 2, 3000)
    signal[0, 0, 250:1250] = torch.cos(phase)
    signal[0, 1, 250:1250] = torch.sin(phase)
    peaks, _ = matched_filter_peaks(signal, frequencies)
    recovered = frequencies[int(peaks[0].argmax())]
    assert abs(float(recovered) - 5000.0) <= 500.0
