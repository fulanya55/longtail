import numpy as np
import torch

from cj0724.data import SIGNAL_LENGTH
from cj0724.p2v_vft_phase_increment import estimate_phase_increment_doppler
from cj0724.vft_doppler_aligned_expert import FS_HZ, PULSE_SAMPLES, lfm_reference


def _example(doppler_hz: float, delay: int = 240) -> torch.Tensor:
    reference = lfm_reference(torch.device("cpu"))[:PULSE_SAMPLES]
    time = torch.arange(PULSE_SAMPLES, dtype=torch.float32) / FS_HZ
    pulse = reference * torch.exp(2j * torch.pi * float(doppler_hz) * time)
    signal = torch.zeros(SIGNAL_LENGTH, dtype=torch.complex64)
    signal[delay:delay + PULSE_SAMPLES] = pulse
    return torch.stack((signal.real, signal.imag), dim=0)


def test_phase_increment_recovers_positive_and_negative_doppler() -> None:
    iq = torch.stack((_example(4500.0), _example(-22000.0)), dim=0)
    result = estimate_phase_increment_doppler(iq)
    np.testing.assert_allclose(result["prediction_hz"].numpy(), [4500.0, -22000.0], atol=2.0)
    assert torch.all(result["coherence"] > 0.99)


def test_phase_increment_validates_shape() -> None:
    with np.testing.assert_raises(ValueError):
        estimate_phase_increment_doppler(torch.zeros(2, 2, 100))
