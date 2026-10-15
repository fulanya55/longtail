"""Fixed no-fit single-PRI delay-autocorrelation diagnostic for Route-A H0p."""
from __future__ import annotations

import torch


def fixed_phase_permutation(length: int, *, device=None) -> torch.Tensor:
    """Return a deterministic nonidentity phase-index permutation."""
    if length < 4:
        raise ValueError("length must be at least 4")
    # Coprime multiplier gives one fixed full permutation; no random state.
    return (torch.arange(length, device=device) * 1193 + 71) % length


def delay_autocorrelation_effective_support(
    iq: torch.Tensor,
    *,
    lag_start: int = 1,
    lag_stop: int = 250,
) -> torch.Tensor:
    """Compute fixed Renyi-2 effective support of nonzero complex ACF lags."""
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [batch, 2, samples]")
    samples = iq.shape[-1]
    if lag_start < 1 or lag_stop < lag_start or lag_stop >= samples:
        raise ValueError("invalid nonzero autocorrelation lag window")
    z = torch.complex(iq[:, 0], iq[:, 1])
    lags = torch.arange(lag_start, lag_stop + 1, device=z.device)
    # The window is intentionally small and fixed; direct products make the
    # physical lag convention explicit and avoid FFT wrap-around ambiguity.
    values = []
    for lag in lags.tolist():
        correlation = (z[:, lag:] * torch.conj(z[:, :-lag])).mean(dim=1)
        values.append(correlation.abs().square())
    energy = torch.stack(values, dim=1)
    distribution = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return 1.0 / distribution.square().sum(dim=1).clamp_min(1e-12)


def permuted_delay_autocorrelation_effective_support(iq: torch.Tensor) -> torch.Tensor:
    """Control readout using the fixed phase-index permutation."""
    index = fixed_phase_permutation(iq.shape[-1], device=iq.device)
    return delay_autocorrelation_effective_support(iq[:, :, index])
