"""Fixed no-fit staggered-PRI VFT phase-consistency diagnostic for Route-A H0q."""
from __future__ import annotations

import torch
import torch.nn.functional as F


FS_HZ = 10e6
PULSE_SAMPLES = 1000
PRI_SAMPLES = 3000
INTERVAL_SAMPLES = (3000, 3310, 3000, 3310, 3000, 3310, 3000)
DELAY_START = 100
DELAY_STOP = 500
FREQUENCY_HZ = tuple(range(1000, 7001, 250))
TIMESTAMP_PERMUTATION = (0, 5, 2, 7, 4, 1, 6, 3)


def pulse_start_samples(*, device=None) -> torch.Tensor:
    """Return fixed, absolute starts of eight staggered PRI pulses."""
    intervals = torch.tensor(INTERVAL_SAMPLES, dtype=torch.int64, device=device)
    return torch.cat((torch.zeros(1, dtype=torch.int64, device=device), torch.cumsum(intervals, dim=0)))


def record_samples() -> int:
    return int(sum(INTERVAL_SAMPLES) + PRI_SAMPLES)


def lfm_pulse(*, device=None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """The fixed unit-energy 0716 baseband up-chirp."""
    time = torch.arange(PULSE_SAMPLES, dtype=torch.float64, device=device) / FS_HZ
    reference = torch.polar(torch.ones_like(time), torch.pi * (5e6 / 100e-6) * time.square())
    return (reference / torch.linalg.vector_norm(reference)).to(dtype)


def _as_complex(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [batch, 2, samples]")
    if iq.shape[-1] != record_samples():
        raise ValueError("unexpected staggered-PRI record length")
    return torch.complex(iq[:, 0], iq[:, 1])


def _matched_coefficients(iq: torch.Tensor) -> torch.Tensor:
    z = _as_complex(iq)
    starts = pulse_start_samples(device=z.device)
    reference = lfm_pulse(device=z.device, dtype=z.dtype)
    kernel = torch.conj(reference).reshape(1, 1, -1)
    values = []
    for start in starts.tolist():
        segment = z[:, start : start + PRI_SAMPLES].reshape(-1, 1, PRI_SAMPLES)
        real = F.conv1d(segment.real, kernel.real) - F.conv1d(segment.imag, kernel.imag)
        imag = F.conv1d(segment.real, kernel.imag) + F.conv1d(segment.imag, kernel.real)
        values.append(torch.complex(real[:, 0], imag[:, 0])[:, DELAY_START : DELAY_STOP + 1])
    return torch.stack(values, dim=1)


def _contrast(coefficients: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    frequency = torch.tensor(FREQUENCY_HZ, device=coefficients.device, dtype=torch.float64)
    phase = torch.exp(-2j * torch.pi * frequency[:, None] * timestamps[None, :] / FS_HZ).to(coefficients.dtype)
    combined = torch.einsum("bpd,fp->bfd", coefficients, phase).abs()
    normalizer = coefficients.abs().sum(dim=1)[:, None, :].clamp_min(1e-12)
    score = (combined / normalizer).amax(dim=2)
    high = frequency >= 5500.0
    low = frequency <= 2500.0
    return score[:, high].amax(dim=1) - score[:, low].amax(dim=1)


def _maximum_score(coefficients: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    frequency = torch.tensor(FREQUENCY_HZ, device=coefficients.device, dtype=torch.float64)
    phase = torch.exp(-2j * torch.pi * frequency[:, None] * timestamps[None, :] / FS_HZ).to(coefficients.dtype)
    combined = torch.einsum("bpd,fp->bfd", coefficients, phase).abs()
    normalizer = coefficients.abs().sum(dim=1)[:, None, :].clamp_min(1e-12)
    return (combined / normalizer).amax(dim=(1, 2))


def _frequency_features(coefficients: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    """Return fixed per-frequency delay-max coherence features [B, 25]."""
    frequency = torch.tensor(FREQUENCY_HZ, device=coefficients.device, dtype=torch.float64)
    phase = torch.exp(-2j * torch.pi * frequency[:, None] * timestamps[None, :] / FS_HZ).to(coefficients.dtype)
    combined = torch.einsum("bpd,fp->bfd", coefficients, phase).abs()
    normalizer = coefficients.abs().sum(dim=1)[:, None, :].clamp_min(1e-12)
    return (combined / normalizer).amax(dim=2)


def staggered_vft_contrasts(iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return true-timestamp and fixed-permuted-timestamp H0q contrasts."""
    coefficients = _matched_coefficients(iq)
    starts = pulse_start_samples(device=coefficients.device)
    permutation = torch.tensor(TIMESTAMP_PERMUTATION, dtype=torch.int64, device=coefficients.device)
    return _contrast(coefficients, starts), _contrast(coefficients, starts[permutation])


def staggered_vft_evidence(iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return H1a real-timestamp and permuted-timestamp full-grid evidence."""
    coefficients = _matched_coefficients(iq)
    starts = pulse_start_samples(device=coefficients.device)
    permutation = torch.tensor(TIMESTAMP_PERMUTATION, dtype=torch.int64, device=coefficients.device)
    return _maximum_score(coefficients, starts), _maximum_score(coefficients, starts[permutation])


def staggered_vft_features(iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return true/permuted fixed frequency coherence grids for H2 heads."""
    coefficients = _matched_coefficients(iq)
    starts = pulse_start_samples(device=coefficients.device)
    permutation = torch.tensor(TIMESTAMP_PERMUTATION, dtype=torch.int64, device=coefficients.device)
    return _frequency_features(coefficients, starts), _frequency_features(coefficients, starts[permutation])


def _beamspace_map(coefficients: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    """Return the normalized complex delay-frequency coherence map [B,3,F,D].

    This deliberately retains the delay coordinate discarded by
    ``_frequency_features``.  It is a fixed, parameter-free H4 P0 observable;
    callers must supply either the physical timestamps or the frozen matched
    permutation, never a learned or selected time ordering.
    """
    frequency = torch.tensor(FREQUENCY_HZ, device=coefficients.device, dtype=torch.float64)
    phase = torch.exp(-2j * torch.pi * frequency[:, None] * timestamps[None, :] / FS_HZ).to(coefficients.dtype)
    combined = torch.einsum("bpd,fp->bfd", coefficients, phase)
    normalizer = coefficients.abs().sum(dim=1).clamp_min(1e-12)[:, None, :]
    normalized = combined / normalizer
    return torch.stack((normalized.real, normalized.imag, normalized.abs()), dim=1)


def staggered_vft_beamspace_maps(iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return true/permuted fixed H4 delay-frequency maps with identical shape.

    The only control difference is the predeclared timestamp permutation.  No
    score, pooling, fit, calibration, threshold or label is used here.
    """
    coefficients = _matched_coefficients(iq)
    starts = pulse_start_samples(device=coefficients.device)
    permutation = torch.tensor(TIMESTAMP_PERMUTATION, dtype=torch.int64, device=coefficients.device)
    return _beamspace_map(coefficients, starts), _beamspace_map(coefficients, starts[permutation])
