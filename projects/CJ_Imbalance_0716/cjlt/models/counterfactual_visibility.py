"""Single-view I/Q probe for counterfactual visibility-margin diagnostics."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CounterfactualVisibilityProbe(nn.Module):
    """Compact six-logit encoder; pair identity is deliberately not an input."""

    def __init__(self, width: int = 16, labels: int = 6) -> None:
        super().__init__()
        base = int(width)
        if base < 4 or int(labels) != 6:
            raise ValueError("width must be >=4 and the fixed schema has six labels")
        self.encoder = nn.Sequential(
            nn.Conv1d(2, base, 15, stride=4, padding=7, bias=False), nn.GroupNorm(4, base), nn.GELU(),
            nn.Conv1d(base, 2 * base, 9, stride=4, padding=4, bias=False), nn.GroupNorm(4, 2 * base), nn.GELU(),
            nn.Conv1d(2 * base, 3 * base, 7, stride=4, padding=3, bias=False), nn.GroupNorm(4, 3 * base), nn.GELU(),
        )
        dimension = 6 * base
        self.head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 2 * base), nn.GELU(), nn.Linear(2 * base, labels))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError("iq must have shape [B,2,T]")
        feature = self.encoder(iq.float())
        pooled = torch.cat([feature.mean(dim=-1), F.adaptive_max_pool1d(feature, 1).flatten(1)], dim=1)
        return self.head(pooled)
