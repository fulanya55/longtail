"""Fixed, no-fit physical statistic for a delayed LFM VFT Doppler intervention."""
from __future__ import annotations

import torch


def lfm_reference_phase(sample_count: int, fs_hz: float = 10e6, chirp_rate_hz_per_s: float = 5e10,
                        device=None, dtype=torch.complex64) -> torch.Tensor:
    """Return the known unit-modulus 0716 in-pulse LFM reference."""
    time = torch.arange(int(sample_count), device=device, dtype=torch.float32) / float(fs_hz)
    return torch.exp(1j * torch.pi * float(chirp_rate_hz_per_s) * time.square()).to(dtype)


def fixed_phase_permutation(reference: torch.Tensor) -> torch.Tensor:
    """Energy/dimension-matched reference with deterministic phase order destroyed."""
    length = reference.numel()
    index = (torch.arange(length, device=reference.device) * 353 + 17) % length
    return reference[index]


def _pulse_window(z: torch.Tensor, pulse_samples: int) -> torch.Tensor:
    energy = z.abs().square()
    window = torch.ones(1, 1, int(pulse_samples), device=z.device, dtype=energy.dtype)
    scores = torch.nn.functional.conv1d(energy[:, None, :], window).squeeze(1)
    start = scores.argmax(dim=1)
    offsets = torch.arange(int(pulse_samples), device=z.device)[None, :]
    return z.gather(1, start[:, None] + offsets)


def dechirped_doppler_hz(signal_iq: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Estimate VFT Doppler with no fit, labels, normalizer or decision threshold."""
    if signal_iq.ndim != 3 or signal_iq.shape[1] != 2:
        raise ValueError('signal_iq must have shape [batch, 2, time]')
    if reference.ndim != 1 or reference.numel() < 2:
        raise ValueError('reference must be a one-dimensional pulse phase')
    z = torch.complex(signal_iq[:, 0].float(), signal_iq[:, 1].float())
    segment = _pulse_window(z, reference.numel())
    dechirped = segment * reference.conj()[None, :]
    phase_increment = torch.angle((dechirped[:, 1:] * dechirped[:, :-1].conj()).mean(dim=1))
    return phase_increment * (10e6 / (2.0 * torch.pi))
