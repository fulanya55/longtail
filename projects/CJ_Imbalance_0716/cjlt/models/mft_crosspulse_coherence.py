"""Fixed no-fit multi-PRI MFT coherence diagnostic used only by Route-A H0n."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lfm_pulse(*, fs_hz: float = 10e6, pulse_samples: int = 1000, device=None) -> torch.Tensor:
    """Return the unit-energy baseband LFM pulse fixed by the 0716 generator."""
    # Keep the generator's double-precision time grid until the caller casts
    # to its input dtype.  At this chirp rate, constructing time directly in
    # float32 changes the terminal phase enough to make the reference a
    # different discrete waveform.
    t = torch.arange(pulse_samples, device=device, dtype=torch.float64) / fs_hz
    chirp_rate = 5e6 / 100e-6
    reference = torch.polar(torch.ones_like(t), torch.pi * chirp_rate * t.square())
    return reference / torch.linalg.vector_norm(reference)


def _as_complex(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [batch, 2, samples]")
    return torch.complex(iq[:, 0], iq[:, 1])


def crosspulse_mft_scores(
    iq: torch.Tensor,
    reference: torch.Tensor,
    *,
    num_pulses: int = 4,
    samples_per_pri: int = 3000,
    delay_start: int = 400,
    delay_stop: int = 999,
    top_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return coherent and phase-invariant fixed matched-filter scores.

    The first output preserves complex agreement over PRI.  The second uses
    only magnitudes, so a whole-PRI phase rotation cannot change it.  Both
    share all fixed windows and top-k choices; no statistic is fitted.
    """
    if delay_start < 0 or delay_stop < delay_start or top_k <= 0:
        raise ValueError("invalid fixed delay window or top_k")
    if iq.shape[-1] != num_pulses * samples_per_pri:
        raise ValueError("unexpected multi-PRI sample length")
    if delay_stop + reference.numel() >= samples_per_pri:
        raise ValueError("matched-filter window exceeds one PRI")
    if top_k > delay_stop - delay_start + 1:
        raise ValueError("top_k exceeds the fixed delay dictionary")

    z = _as_complex(iq).reshape(-1, num_pulses, samples_per_pri)
    ref = reference.to(device=z.device, dtype=z.dtype)
    # torch.conv1d is cross-correlation, so conj(reference) (rather than a
    # reversed FIR convolution kernel) implements sum x[d+n] conj(r[n]).
    kernels = torch.conj(ref).reshape(1, 1, -1)
    flat = z.reshape(-1, 1, samples_per_pri)
    real = F.conv1d(flat.real, kernels.real) - F.conv1d(flat.imag, kernels.imag)
    imag = F.conv1d(flat.real, kernels.imag) + F.conv1d(flat.imag, kernels.real)
    batch_size = z.shape[0]
    corr = torch.complex(real[:, 0], imag[:, 0]).reshape(batch_size, num_pulses, -1)
    # conv index d uses the data window [d, d + pulse_samples); compensate
    # only for the fixed valid-convolution origin, never learned alignment.
    coeff = corr[:, :, delay_start : delay_stop + 1]
    abs_coeff = coeff.abs()
    noncoherent = abs_coeff.mean(dim=1)
    amplitudes, indices = torch.topk(noncoherent, k=top_k, dim=1, largest=True, sorted=False)
    selected = torch.gather(coeff, 2, indices[:, None, :].expand(-1, num_pulses, -1))
    coherence = selected.sum(dim=1).abs() / selected.abs().sum(dim=1).clamp_min(1e-12)
    weights = amplitudes / amplitudes.sum(dim=1, keepdim=True).clamp_min(1e-12)
    coherent_score = (weights * coherence).sum(dim=1)
    invariant_score = amplitudes.mean(dim=1) / noncoherent.mean(dim=1).clamp_min(1e-12)
    return coherent_score, invariant_score
