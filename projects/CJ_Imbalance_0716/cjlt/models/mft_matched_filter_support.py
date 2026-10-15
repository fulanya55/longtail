"""Fixed MFT matched-filter support statistic for a paired H0 screen.

The statistic has no learnable state: a known LFM is circularly matched to an
I/Q record, the physically allowed delay interval is retained, and the
Renyi-2 effective support of the normalized response energy is reported.  It
is intentionally different from the historical threshold/NMS peak-count
guard: no background quantile, peak threshold, NMS radius, calibration or
classification head is involved.
"""
from __future__ import annotations

import torch


def lfm_reference(sample_count: int = 1000, fs_hz: float = 10e6,
                  chirp_rate_hz_per_s: float = 5e10, *, device=None) -> torch.Tensor:
    time = torch.arange(int(sample_count), device=device, dtype=torch.float32) / float(fs_hz)
    return torch.exp(1j * torch.pi * float(chirp_rate_hz_per_s) * time.square()).to(torch.complex64)


def fixed_phase_permutation(reference: torch.Tensor) -> torch.Tensor:
    """Deterministic unit-modulus reference with the LFM phase order destroyed."""
    length = int(reference.numel())
    index = (torch.arange(length, device=reference.device) * 353 + 17) % length
    return reference[index]


def effective_delay_support(signal_iq: torch.Tensor, reference: torch.Tensor,
                            delay_start: int = 400, delay_stop: int = 1000) -> torch.Tensor:
    """Return exp(Renyi-2 entropy) of matched-filter response energy in a fixed ROI.

    ``delay_start:delay_stop`` is fixed by the 0716 MFT contract: common
    arrival 10--50 us plus relative target delay 30--50 us at 10 MHz.
    """
    if signal_iq.ndim != 3 or signal_iq.shape[1] != 2:
        raise ValueError('signal_iq must have shape [batch, 2, time]')
    if reference.ndim != 1 or reference.numel() < 2:
        raise ValueError('reference must be a nonempty one-dimensional LFM')
    length = int(signal_iq.shape[-1])
    start, stop = int(delay_start), int(delay_stop)
    if not (0 <= start < stop <= length):
        raise ValueError('invalid fixed delay ROI')
    z = torch.complex(signal_iq[:, 0].float(), signal_iq[:, 1].float())
    template = torch.zeros(length, dtype=torch.complex64, device=z.device)
    template[:reference.numel()] = reference.to(device=z.device, dtype=torch.complex64)
    response = torch.fft.ifft(torch.fft.fft(z, dim=-1) * torch.fft.fft(template, dim=-1).conj()[None, :], dim=-1)
    energy = response[:, start:stop].abs().square()
    distribution = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return 1.0 / distribution.square().sum(dim=-1).clamp_min(1.0e-12)
