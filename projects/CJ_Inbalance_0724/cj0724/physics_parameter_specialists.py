"""Physics-shaped oracle-positive specialists for three weak parameter targets."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _phase_increment(iq: torch.Tensor) -> torch.Tensor:
    real, imag = iq[:, 0], iq[:, 1]
    cross = real[:, :-1] * imag[:, 1:] - imag[:, :-1] * real[:, 1:]
    dot = real[:, :-1] * real[:, 1:] + imag[:, :-1] * imag[:, 1:]
    return torch.atan2(cross, dot) / torch.pi


def _normalized_autocorrelation(sequence: torch.Tensor, maximum_lag: int) -> torch.Tensor:
    values = sequence.float() - sequence.float().mean(dim=-1, keepdim=True)
    length = int(values.shape[-1])
    fft_length = 1 << (2 * length - 1).bit_length()
    spectrum = torch.fft.rfft(values, n=fft_length, dim=-1)
    correlation = torch.fft.irfft(spectrum * spectrum.conj(), n=fft_length, dim=-1)
    correlation = correlation[:, 1 : min(length, int(maximum_lag) + 1)]
    zero_lag = values.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return correlation / zero_lag


class SequenceSummary(nn.Module):
    def __init__(self, input_channels: int, width: int = 16):
        super().__init__()
        width = int(width)
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, width, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.GELU(),
            nn.Conv1d(width * 2, width * 4, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.GELU(),
        )
        self.feature_dim = width * 8

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        feature = self.network(sequence)
        return torch.cat(
            [
                F.adaptive_avg_pool1d(feature, 1).squeeze(-1),
                F.adaptive_max_pool1d(feature, 1).squeeze(-1),
            ],
            dim=1,
        )


class SawFMPeriodSequenceSpecialist(nn.Module):
    """Estimate sweep period from the autocorrelation of instantaneous frequency."""

    def __init__(self, width: int = 16, maximum_lag: int = 1000):
        super().__init__()
        self.maximum_lag = int(maximum_lag)
        self.summary = SequenceSummary(2, width)
        self.head = nn.Sequential(nn.LayerNorm(self.summary.feature_dim), nn.Linear(self.summary.feature_dim, 1))

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        frequency = _phase_increment(iq)
        correlation = _normalized_autocorrelation(frequency, self.maximum_lag)
        smoothed = F.avg_pool1d(correlation[:, None], 9, stride=1, padding=4).squeeze(1)
        feature = self.summary(torch.stack([correlation, smoothed], dim=1))
        return {"prediction": self.head(feature).squeeze(-1), "sequence": correlation}


class MFTDelayPeakOrdinalSpecialist(nn.Module):
    """Estimate MFT count from complex autocorrelation delay peaks with an ordinal head."""

    def __init__(self, width: int = 16, maximum_lag: int = 1500):
        super().__init__()
        self.maximum_lag = int(maximum_lag)
        self.summary = SequenceSummary(2, width)
        self.score = nn.Linear(self.summary.feature_dim, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-1.0))
        self.threshold_gaps = nn.Parameter(torch.zeros(4))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.threshold_gaps) + 1.0e-3
        return torch.cat([self.threshold_start[None], self.threshold_start[None] + gaps.cumsum(0)])

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        length = int(values.shape[-1])
        spectrum = torch.fft.fft(values, n=2 * length, dim=-1)
        correlation = torch.fft.ifft(spectrum * spectrum.conj(), dim=-1).abs()
        correlation = correlation[:, 1 : min(length, self.maximum_lag + 1)]
        correlation = correlation / correlation.amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        smoothed = F.avg_pool1d(correlation[:, None], 7, stride=1, padding=3).squeeze(1)
        feature = self.summary(torch.stack([correlation, smoothed], dim=1))
        logits = self.score(feature) - self.thresholds()[None, :]
        prediction = 3.0 + torch.sigmoid(logits).sum(dim=-1)
        return {"prediction": prediction, "ordinal_logits": logits, "sequence": correlation}


def ordinal_count_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_class = (target.round().long() - 3).clamp(0, 5)
    levels = torch.arange(5, device=logits.device)[None, :]
    cumulative = (target_class[:, None] > levels).float()
    return F.binary_cross_entropy_with_logits(logits.float(), cumulative)


class VFTInstantaneousFrequencySpecialist(nn.Module):
    """Estimate Doppler from the time-resolved instantaneous-frequency sequence."""

    def __init__(self, width: int = 16):
        super().__init__()
        self.summary = SequenceSummary(3, width)
        self.head = nn.Sequential(nn.LayerNorm(self.summary.feature_dim), nn.Linear(self.summary.feature_dim, 1))

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        frequency = _phase_increment(iq)
        magnitude = torch.sqrt(iq[:, 0].square() + iq[:, 1].square()).clamp_min(1.0e-8)
        local_magnitude = 0.5 * (magnitude[:, 1:] + magnitude[:, :-1])
        change = magnitude[:, 1:] - magnitude[:, :-1]
        sequence = torch.stack([frequency, local_magnitude, change], dim=1)
        return {"prediction": self.head(self.summary(sequence)).squeeze(-1), "sequence": frequency}


def build_physics_specialist(name: str, width: int = 16) -> nn.Module:
    builders = {
        "sawfm_period_sequence": SawFMPeriodSequenceSpecialist,
        "mft_delay_peak_ordinal": MFTDelayPeakOrdinalSpecialist,
        "vft_instantaneous_frequency": VFTInstantaneousFrequencySpecialist,
    }
    try:
        return builders[str(name)](width=width)
    except KeyError as error:
        raise ValueError(f"Unknown physics specialist: {name!r}") from error
