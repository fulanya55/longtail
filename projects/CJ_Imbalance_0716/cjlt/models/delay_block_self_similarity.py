"""Fixed distributed complex delay-block evidence for ISRJ replica diagnostics."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DelayBlockSelfSimilarityMap(nn.Module):
    """Map complex correlations over fixed lags and local overlap blocks.

    The module has no trainable parameters.  Unlike an ACF-argmax summary, it
    retains the full fixed delay band and the local complex phase relation.
    """

    def __init__(self, lag_start: int = 80, lag_stop: int = 480, lag_step: int = 16, blocks: int = 12):
        super().__init__()
        lags = torch.arange(int(lag_start), int(lag_stop) + 1, int(lag_step), dtype=torch.long)
        if lags.numel() < 2 or int(blocks) < 2:
            raise ValueError("requires at least two lags and blocks")
        self.register_buffer("lags", lags, persistent=True)
        self.blocks = int(blocks)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, int(self.lags.numel()), self.blocks)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 2:
            raise ValueError("signal must have shape [B,2,T]")
        z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
        z = z / z.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
        length = int(z.shape[1])
        if int(self.lags[-1]) >= length - self.blocks:
            raise ValueError("signal is too short for configured lags and blocks")
        rows = []
        for lag in self.lags.tolist():
            overlap = length - int(lag)
            starts = torch.div(torch.arange(self.blocks, device=z.device) * overlap, self.blocks, rounding_mode="floor")
            ends = torch.div(torch.arange(1, self.blocks + 1, device=z.device) * overlap, self.blocks, rounding_mode="floor")
            columns = []
            for start, end in zip(starts.tolist(), ends.tolist()):
                left = z[:, start:end]
                right = z[:, start + int(lag):end + int(lag)]
                numerator = (left.conj() * right).sum(dim=1)
                denominator = (left.abs().square().sum(dim=1) * right.abs().square().sum(dim=1)).sqrt().clamp_min(1.0e-8)
                columns.append(numerator / denominator)
            rows.append(torch.stack(columns, dim=1))
        correlation = torch.stack(rows, dim=1)
        return torch.stack([correlation.real, correlation.imag], dim=1).float()


class DelayBlockReplicaProbe(nn.Module):
    """Small replica-count regressor operating only on the fixed map."""

    def __init__(self, lag_start: int = 80, lag_stop: int = 480, lag_step: int = 16, blocks: int = 12):
        super().__init__()
        self.map = DelayBlockSelfSimilarityMap(lag_start, lag_stop, lag_step, blocks)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.map(signal))
        feature = torch.cat([F.adaptive_avg_pool2d(encoded, 1).flatten(1), F.adaptive_max_pool2d(encoded, 1).flatten(1)], dim=1)
        return self.head(feature).squeeze(1)


class WidthNormalizedDelayBlockMap(nn.Module):
    """Complex block correlations at fixed integer multiples of a supplied width.

    ``width_samples`` is an oracle diagnostic argument in H0b. It is not an
    input feature and this module contains no trainable parameters.
    """

    def __init__(self, multipliers: int = 8, blocks: int = 12):
        super().__init__()
        if int(multipliers) < 2 or int(blocks) < 2:
            raise ValueError("requires at least two multiplier rows and blocks")
        self.multipliers = int(multipliers)
        self.blocks = int(blocks)

    def forward(self, signal: torch.Tensor, width_samples: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 2:
            raise ValueError("signal must have shape [B,2,T]")
        if width_samples.ndim != 1 or width_samples.shape[0] != signal.shape[0]:
            raise ValueError("width_samples must have shape [B]")
        z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
        z = z / z.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
        batch, length = z.shape
        width = width_samples.to(device=z.device, dtype=torch.long).clamp_min(1)
        time = torch.arange(length, device=z.device)[None, :].expand(batch, -1)
        rows = []
        for multiplier in range(1, self.multipliers + 1):
            lag = width * multiplier
            overlap = (length - lag).clamp_min(0)
            columns = []
            for block in range(self.blocks):
                start = torch.div(block * overlap, self.blocks, rounding_mode="floor")
                end = torch.div((block + 1) * overlap, self.blocks, rounding_mode="floor")
                valid = (time >= start[:, None]) & (time < end[:, None]) & (lag[:, None] < length)
                shifted = z.gather(1, (time + lag[:, None]).clamp_max(length - 1))
                numerator = (z.conj() * shifted * valid).sum(dim=1)
                energy_left = (z.abs().square() * valid).sum(dim=1)
                energy_right = (shifted.abs().square() * valid).sum(dim=1)
                columns.append(numerator / (energy_left * energy_right).sqrt().clamp_min(1.0e-8))
            rows.append(torch.stack(columns, dim=1))
        correlation = torch.stack(rows, dim=1)
        return torch.stack([correlation.real, correlation.imag], dim=1).float()


class WidthNormalizedDelayBlockReplicaProbe(nn.Module):
    """H0b replica regressor with an oracle-normalized fixed delay map."""

    def __init__(self, multipliers: int = 8, blocks: int = 12):
        super().__init__()
        self.map = WidthNormalizedDelayBlockMap(multipliers, blocks)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False), nn.GroupNorm(4, 8), nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, signal: torch.Tensor, width_samples: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.map(signal, width_samples))
        feature = torch.cat([F.adaptive_avg_pool2d(encoded, 1).flatten(1), F.adaptive_max_pool2d(encoded, 1).flatten(1)], dim=1)
        return self.head(feature).squeeze(1)


class ScaleHoughDelayBlockMap(nn.Module):
    """Inference-ready correlation bank over fixed width hypotheses.

    This keeps all width hypotheses instead of using an oracle (or a point
    prediction) to choose one discrete delay coordinate.  Rows are ordered by
    ``width_samples`` and then multiplier, so the small downstream CNN can
    aggregate a locally consistent delay-scale ridge without learning a width
    regressor or receiving any physical parameter as input.
    """

    def __init__(self, width_start: int = 100, width_stop: int = 300, width_step: int = 20, multipliers: int = 8, blocks: int = 12):
        super().__init__()
        widths = torch.arange(int(width_start), int(width_stop) + 1, int(width_step), dtype=torch.long)
        if widths.numel() < 2 or int(multipliers) < 2 or int(blocks) < 2:
            raise ValueError("requires at least two widths, multiplier rows, and blocks")
        self.register_buffer("widths", widths, persistent=True)
        self.register_buffer("multipliers", torch.arange(1, int(multipliers) + 1, dtype=torch.long), persistent=True)
        self.blocks = int(blocks)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, int(self.widths.numel() * self.multipliers.numel()), self.blocks)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 2:
            raise ValueError("signal must have shape [B,2,T]")
        z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
        z = z / z.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
        batch, length = z.shape
        if int(self.widths[-1] * self.multipliers[-1]) >= length - self.blocks:
            raise ValueError("signal is too short for configured width bank and blocks")
        time = torch.arange(length, device=z.device)[None, :]
        rows = []
        for width in self.widths.tolist():
            lag = (int(width) * self.multipliers).to(device=z.device)
            overlap = length - lag
            shifted_index = (time[:, None, :] + lag[None, :, None]).clamp_max(length - 1).expand(batch, -1, -1)
            shifted = z[:, None, :].expand(-1, int(lag.numel()), -1).gather(2, shifted_index)
            local_rows = []
            for block in range(self.blocks):
                start = torch.div(block * overlap, self.blocks, rounding_mode="floor")
                end = torch.div((block + 1) * overlap, self.blocks, rounding_mode="floor")
                valid = (time[:, None, :] >= start[None, :, None]) & (time[:, None, :] < end[None, :, None])
                left = z[:, None, :]
                numerator = (left.conj() * shifted * valid).sum(dim=2)
                energy_left = (left.abs().square() * valid).sum(dim=2)
                energy_right = (shifted.abs().square() * valid).sum(dim=2)
                local_rows.append(numerator / (energy_left * energy_right).sqrt().clamp_min(1.0e-8))
            rows.append(torch.stack(local_rows, dim=2))
        correlation = torch.cat(rows, dim=1)
        return torch.stack([correlation.real, correlation.imag], dim=1).float()


class ScaleHoughReplicaProbe(nn.Module):
    """Compact replica regressor for a fixed, no-oracle scale-Hough map."""

    def __init__(self, width_start: int = 100, width_stop: int = 300, width_step: int = 20, multipliers: int = 8, blocks: int = 12):
        super().__init__()
        self.map = ScaleHoughDelayBlockMap(width_start, width_stop, width_step, multipliers, blocks)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False), nn.GroupNorm(4, 8), nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))

    def forward_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1] != 2:
            raise ValueError("feature_map must have shape [B,2,rows,blocks]")
        encoded = self.encoder(feature_map)
        feature = torch.cat([F.adaptive_avg_pool2d(encoded, 1).flatten(1), F.adaptive_max_pool2d(encoded, 1).flatten(1)], dim=1)
        return self.head(feature).squeeze(1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.forward_map(self.map(signal))
