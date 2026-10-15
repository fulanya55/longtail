"""P3o-U fixed local phase-dynamics features and bounded ISRJ-only residual.

This module deliberately contains no inverse-delay, global FFT/ACF, oracle
component, parameter, or frozen MLCNN global-feature input.  It is separate
from the frozen P3o-T implementation because it tests a different observable
mechanism after the P3o-T support hypothesis failed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


SIGNAL_LENGTH = 3000
ISRJ_CLASS_INDEX = 5
MFT_CLASS_INDEX = 3
PHASE_WINDOWS = (9, 33, 129)
STFT_NFFT, STFT_HOP, TIME_BINS = 128, 32, 16
FEATURE_DIM = len(PHASE_WINDOWS) * 2 * TIME_BINS * 2 + 5 * TIME_BINS * 2


def _binned_mean_and_max(values: torch.Tensor, bins: int = TIME_BINS) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected [B,T], got {tuple(values.shape)}")
    average = F.adaptive_avg_pool1d(values[:, None], bins).squeeze(1)
    maximum = F.adaptive_max_pool1d(values[:, None], bins).squeeze(1)
    return torch.cat((average, maximum), dim=1)


class LocalPhaseDynamics(nn.Module):
    """Fixed single-PRI local phase and STFT dynamics, summarized by time bin."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("hann", torch.hann_window(STFT_NFFT, periodic=True))
        # A fixed nontrivial permutation avoids data-dependent shuffled controls.
        self.register_buffer("time_permutation", torch.tensor([7, 2, 13, 4, 15, 0, 10, 5, 12, 1, 9, 14, 3, 11, 6, 8]))

    @staticmethod
    def _normalized_signal(iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
            raise ValueError(f"Expected [B,2,{SIGNAL_LENGTH}], got {tuple(iq.shape)}")
        signal = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        return signal / signal.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)

    def _phase_features(self, signal: torch.Tensor) -> torch.Tensor:
        increment = torch.angle(signal[:, 1:] * signal[:, :-1].conj()) / math.pi
        parts: list[torch.Tensor] = []
        for width in PHASE_WINDOWS:
            smooth = F.avg_pool1d(increment[:, None], width, stride=1, padding=width // 2).squeeze(1)
            derivative = F.pad(smooth[:, 1:] - smooth[:, :-1], (1, 0)).abs()
            parts.extend((_binned_mean_and_max(smooth), _binned_mean_and_max(derivative)))
        return torch.cat(parts, dim=1)

    def _stft_features(self, signal: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(signal, n_fft=STFT_NFFT, hop_length=STFT_HOP, win_length=STFT_NFFT, window=self.hann, center=True, return_complex=True)
        power = spectrum.abs().square().transpose(1, 2)
        normalized = power / power.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        frequency = torch.linspace(-1.0, 1.0, normalized.shape[2], device=normalized.device, dtype=normalized.dtype)
        centroid = (normalized * frequency).sum(dim=2)
        spread = (normalized * (frequency - centroid[:, :, None]).square()).sum(dim=2).sqrt()
        entropy = -(normalized * normalized.clamp_min(1.0e-10).log()).sum(dim=2) / math.log(float(normalized.shape[2]))
        peak = normalized.argmax(dim=2).to(normalized.dtype) / max(normalized.shape[2] - 1, 1) * 2.0 - 1.0
        concentration = normalized.amax(dim=2)
        return torch.cat([_binned_mean_and_max(item) for item in (centroid, spread, entropy, peak, concentration)], dim=1)

    def forward(self, iq: torch.Tensor, mode: str = "phase") -> torch.Tensor:
        signal = self._normalized_signal(iq)
        features = torch.cat((self._phase_features(signal), self._stft_features(signal)), dim=1)
        if features.shape[1] != FEATURE_DIM:
            raise RuntimeError(f"P3o-U feature contract mismatch: {features.shape[1]} != {FEATURE_DIM}")
        if mode == "phase":
            return features
        if mode == "zero":
            return torch.zeros_like(features)
        if mode == "shuffled":
            blocks = features.reshape(len(features), -1, TIME_BINS)
            return blocks[:, :, self.time_permutation].reshape_as(features)
        raise ValueError(f"Unknown P3o-U feature mode {mode!r}")


class BoundedISRJResidual(nn.Module):
    """Equal-capacity, posterior-gated negative ISRJ correction only."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 32, bound: float = 2.0) -> None:
        super().__init__()
        self.bound = float(bound)
        self.network = nn.Sequential(nn.LayerNorm(feature_dim + 1), nn.Linear(feature_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -8.0)

    def forward(self, frozen_logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if frozen_logits.ndim != 2 or frozen_logits.shape[1] != 6 or features.ndim != 2 or len(features) != len(frozen_logits):
            raise ValueError("Incompatible P3o-U logits/features")
        gate = torch.sigmoid(frozen_logits[:, MFT_CLASS_INDEX]).detach().unsqueeze(1)
        raw = self.network(torch.cat((features.detach(), gate), dim=1)).squeeze(1)
        return -self.bound * torch.sigmoid(raw) * gate.squeeze(1)


def corrected_logits(base_logits: torch.Tensor, delta_isrj: torch.Tensor) -> torch.Tensor:
    if base_logits.ndim != 2 or base_logits.shape[1] != 6 or delta_isrj.shape != (len(base_logits),):
        raise ValueError("Incompatible P3o-U logits/residual")
    corrected = base_logits.clone()
    corrected[:, ISRJ_CLASS_INDEX] += delta_isrj
    return corrected


def conditional_residual_loss(corrected: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor, weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(corrected[:, ISRJ_CLASS_INDEX], labels[:, ISRJ_CLASS_INDEX])
    source = (labels[:, MFT_CLASS_INDEX] > 0.5) & (labels[:, ISRJ_CLASS_INDEX] < 0.5)
    conditional = F.softplus(corrected[source, ISRJ_CLASS_INDEX] - base_logits[source, ISRJ_CLASS_INDEX]).mean() if bool(source.any()) else corrected.sum() * 0.0
    return bce + float(weight) * conditional, bce, conditional
