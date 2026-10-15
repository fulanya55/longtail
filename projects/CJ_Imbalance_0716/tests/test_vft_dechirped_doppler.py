import torch

from cjlt.models.vft_dechirped_doppler import dechirped_doppler_hz, fixed_phase_permutation, lfm_reference_phase


def test_dechirped_increment_recovers_known_doppler_and_control_is_matched():
    fs, pulse, delay, fd = 10e6, 1000, 217, 6400.0
    reference = lfm_reference_phase(pulse)
    signal = torch.zeros(1, 2, 3000)
    time = torch.arange(pulse, dtype=torch.float32) / fs
    z = reference * torch.exp(1j * 2 * torch.pi * fd * time)
    signal[0, 0, delay:delay + pulse] = z.real
    signal[0, 1, delay:delay + pulse] = z.imag
    estimated = dechirped_doppler_hz(signal, reference)
    control = fixed_phase_permutation(reference)
    assert torch.allclose(estimated, torch.tensor([fd]), atol=2.0)
    assert control.shape == reference.shape
    assert torch.allclose(control.abs(), reference.abs())
