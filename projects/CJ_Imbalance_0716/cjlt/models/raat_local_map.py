"""Deterministic Reference-Aligned Ambiguity Topology (RAAT) maps.

This module deliberately accepts only received I/Q and a fixed transmitted LFM
reference.  It has no access to labels, component metadata, power, queries, or
global classifier outputs.  The three extractors are frozen by RAAT-H0/v1.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


FS_HZ = 10_000_000.0
PULSE_SAMPLES = 1000
PRI_SAMPLES = 3000
BANDWIDTH_HZ = 5_000_000.0
DELAY_START, DELAY_STOP = 400, 1000
DELAY_POOL = 25
DOPPLER_HZ = (-12000, -10500, -9000, -7500, -6000, -4500, -3000, -1500,
              0, 1500, 3000, 4500, 6000, 7500, 9000, 10500, 12000)
EPS = 1e-6


def build_lfm_reference(device: torch.device, *, reverse_chirp: bool = False) -> torch.Tensor:
    """Create the known, unit-amplitude transmit reference for one PRI."""
    slope = BANDWIDTH_HZ / (PULSE_SAMPLES / FS_HZ)
    if reverse_chirp:
        slope = -slope
    time = torch.arange(1, PULSE_SAMPLES + 1, device=device, dtype=torch.float32) / FS_HZ
    pulse = torch.exp(1j * math.pi * slope * time.square()).to(torch.complex64)
    reference = torch.zeros(PRI_SAMPLES, device=device, dtype=torch.complex64)
    reference[:PULSE_SAMPLES] = pulse
    return reference


def _complex_iq(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 3 or iq.shape[1:] != (2, PRI_SAMPLES):
        raise ValueError(f"expected (B,2,{PRI_SAMPLES}) I/Q, got {tuple(iq.shape)}")
    return torch.complex(iq[:, 0].float(), iq[:, 1].float())


def _profile_at_dopplers(signal: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cross-ambiguity delay profiles, returned as (B, D, 601) complex values."""
    batch, length = signal.shape
    if length != PRI_SAMPLES or reference.numel() != PRI_SAMPLES:
        raise ValueError("unexpected PRI length")
    time = torch.arange(length, device=signal.device, dtype=torch.float32) / FS_HZ
    nu = torch.as_tensor(DOPPLER_HZ, device=signal.device, dtype=torch.float32)
    rotator = torch.exp(-2j * math.pi * nu[:, None] * time[None, :]).to(torch.complex64)
    mixed = signal[:, None, :] * rotator[None, :, :]
    corr = torch.fft.ifft(
        torch.fft.fft(mixed, dim=-1) * torch.fft.fft(reference, dim=-1).conj()[None, None, :],
        dim=-1,
    )
    return corr[:, :, DELAY_START : DELAY_STOP + 1]


def _normalised_amp_phase(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = values.abs().flatten(1).median(dim=1).values[:, None, None].clamp_min(EPS)
    amplitude = torch.log1p(values.abs() / scale)
    phase = torch.sin(torch.angle(values))
    return amplitude, phase


def _pool_delay(values: torch.Tensor) -> torch.Tensor:
    """Pool 600 of the frozen 601 delay values into 24 non-overlapping bins."""
    values = values[..., :600]
    return values.reshape(*values.shape[:-1], 24, DELAY_POOL).mean(dim=-1)


def mft_delay_multiplicity_map(iq: torch.Tensor, *, reverse_chirp: bool = False) -> torch.Tensor:
    """Return the frozen MFT map with shape (B, 2, 24, 17)."""
    signal = _complex_iq(iq)
    reference = build_lfm_reference(signal.device, reverse_chirp=reverse_chirp)
    ambiguity = _profile_at_dopplers(signal, reference)
    amplitude, phase = _normalised_amp_phase(ambiguity)
    amplitude = _pool_delay(amplitude).transpose(1, 2)
    phase = _pool_delay(phase).transpose(1, 2)
    return torch.stack((amplitude, phase), dim=1)


def _aligned_dechirped(signal: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    profile = _profile_at_dopplers(signal, reference)
    zero_index = DOPPLER_HZ.index(0)
    tau_hat = profile[:, zero_index].abs().argmax(dim=-1) + DELAY_START
    shifts = torch.arange(PRI_SAMPLES, device=signal.device)[None, :] + tau_hat[:, None]
    aligned = signal.gather(1, shifts.remainder(PRI_SAMPLES))
    return aligned[:, :PULSE_SAMPLES] * reference[:PULSE_SAMPLES].conj()[None, :]


def vft_dechirped_lag_coherence_map(iq: torch.Tensor, *, reverse_chirp: bool = False) -> torch.Tensor:
    """Return the frozen VFT lag-frequency map with shape (B, 2, 24, 17)."""
    signal = _complex_iq(iq)
    reference = build_lfm_reference(signal.device, reverse_chirp=reverse_chirp)
    dechirped = _aligned_dechirped(signal, reference)
    time = torch.arange(PULSE_SAMPLES, device=signal.device, dtype=torch.float32) / FS_HZ
    nu = torch.as_tensor(DOPPLER_HZ, device=signal.device, dtype=torch.float32)
    rotator = torch.exp(-2j * math.pi * nu[:, None] * time[None, :]).to(torch.complex64)
    # Zero-padded FFT cross-correlation is algebraically identical to the
    # registered finite-lag sum, but avoids a 921-iteration Python loop.
    # corr[..., lag] = sum_n z[n+lag] conj(z[n]) exp(-j*2*pi*nu*n/fs).
    fft_length = 2048
    left = torch.fft.fft(dechirped[:, None, :] * rotator[None, :, :], n=fft_length, dim=-1)
    right = torch.fft.fft(dechirped, n=fft_length, dim=-1).conj()[:, None, :]
    raw = torch.fft.ifft(left * right, dim=-1)[..., :921]
    lags = torch.arange(921, device=signal.device, dtype=torch.float32)
    correction = torch.exp(2j * math.pi * nu[:, None] * lags[None, :] / FS_HZ).to(torch.complex64)
    coherence = raw * correction[None, :, :]
    counts = (PULSE_SAMPLES - lags).clamp_min(1)[None, None, :]
    coherence = coherence / counts
    amplitude, phase = _normalised_amp_phase(coherence)
    amplitude = F.pad(amplitude, (0, 39)).reshape(len(signal), len(DOPPLER_HZ), 24, 40).mean(dim=-1)
    phase = F.pad(phase, (0, 39)).reshape(len(signal), len(DOPPLER_HZ), 24, 40).mean(dim=-1)
    return torch.stack((amplitude.transpose(1, 2), phase.transpose(1, 2)), dim=1)


def isrj_segmented_delay_fragment_map(iq: torch.Tensor, *, reverse_chirp: bool = False) -> torch.Tensor:
    """Return the frozen ISRJ segment-delay map with shape (B, 2, 12, 24)."""
    signal = _complex_iq(iq)
    reference = build_lfm_reference(signal.device, reverse_chirp=reverse_chirp)
    profiles = []
    for start in range(0, PRI_SAMPLES, 250):
        masked = torch.zeros_like(signal)
        masked[:, start : start + 250] = signal[:, start : start + 250]
        profile = _profile_at_dopplers(masked, reference)[:, DOPPLER_HZ.index(0)]
        profiles.append(profile)
    profile = torch.stack(profiles, dim=1)
    scale = profile.abs().flatten(1).median(dim=1).values[:, None, None].clamp_min(EPS)
    energy = _pool_delay(torch.log1p(profile.abs() / scale))
    unit = profile / profile.abs().clamp_min(EPS)
    adjacent = (unit[:, 1:] * unit[:, :-1].conj()).real
    coherence = torch.cat((torch.zeros_like(adjacent[:, :1]), adjacent), dim=1)
    coherence = _pool_delay(coherence)
    return torch.stack((energy, coherence), dim=1)


def extract_raat_map(iq: torch.Tensor, target: str, *, reverse_chirp: bool = False) -> torch.Tensor:
    extractors = {
        "MFT": mft_delay_multiplicity_map,
        "VFT": vft_dechirped_lag_coherence_map,
        "ISRJ": isrj_segmented_delay_fragment_map,
    }
    try:
        return extractors[target](iq, reverse_chirp=reverse_chirp)
    except KeyError as exc:
        raise ValueError(f"unknown RAAT target: {target}") from exc


class RAATLocalMapNet(nn.Module):
    """The 7,425-parameter fixed H0 classifier used for every RAAT arm."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.features(maps).squeeze(-1)


def parameter_count() -> int:
    return sum(item.numel() for item in RAATLocalMapNet().parameters())
