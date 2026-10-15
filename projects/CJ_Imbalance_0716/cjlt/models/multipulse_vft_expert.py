"""Role-isolated eight-pulse six-label H2 candidate family."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cjlt.models.vft_staggered_pri import PRI_SAMPLES, pulse_start_samples, record_samples, staggered_vft_features

LABEL_COUNT, SAWFM_INDEX, VFT_INDEX = 6, 1, 4


@dataclass(frozen=True)
class MultiPulseMode:
    use_saw_expert: bool
    vft_mode: str


MODES = {
    "C0-mp": MultiPulseMode(False, "none"),
    "C1-mp": MultiPulseMode(True, "none"),
    "C2-mp": MultiPulseMode(False, "true"),
    "C3-mp": MultiPulseMode(True, "true"),
    "C4-mp": MultiPulseMode(True, "permuted"),
    # H3 is an identifiability probe without the SawFM branch.  These names
    # make its T0/T1/T2 capacity control explicit rather than overloading the
    # six-label H2 C4 arm, whose SawFM expert is intentionally enabled.
    "T0-h3": MultiPulseMode(False, "none"),
    "T1-h3": MultiPulseMode(False, "true"),
    "T2-h3": MultiPulseMode(False, "permuted"),
}


def _windows(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[-1] != record_samples():
        raise ValueError(f"expected [B,2,{record_samples()}] I/Q input")
    starts = pulse_start_samples(device=iq.device)
    return torch.stack([iq[..., int(start):int(start) + PRI_SAMPLES] for start in starts.tolist()], dim=1)


class PulseBackbone(nn.Module):
    """Shared I/Q encoder applied identically to every PRI window."""
    def __init__(self, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, width, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(width), nn.SiLU(),
            nn.Conv1d(width, width * 2, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(width * 2), nn.SiLU(),
            nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2, groups=2, bias=False), nn.BatchNorm1d(width * 2), nn.SiLU(),
        )
        self.output_dim = width * 4

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        windows = _windows(iq); batch, pulses = windows.shape[:2]
        encoded = self.encoder(windows.reshape(batch * pulses, 2, PRI_SAMPLES)).mean(dim=-1).reshape(batch, pulses, -1)
        return torch.cat((encoded.mean(dim=1), encoded.std(dim=1, unbiased=False)), dim=1)


class SawRawExpert(nn.Module):
    """Raw local SawFM branch restricted to the SawFM logit."""
    def __init__(self, width: int = 16):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(2, width, 21, stride=4, padding=10, bias=False), nn.BatchNorm1d(width), nn.SiLU(),
            nn.Conv1d(width, width, 9, padding=4, groups=width, bias=False), nn.BatchNorm1d(width), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        windows = _windows(iq); batch, pulses = windows.shape[:2]
        encoded = self.extractor(windows.reshape(batch * pulses, 2, PRI_SAMPLES)).mean(dim=-1).reshape(batch, pulses, -1)
        return self.head(torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=1)).squeeze(-1)


class MultiPulseVFTExpertClassifier(nn.Module):
    """C0--C4 H2 arms with auditable target-only expert contributions."""
    def __init__(self, mode: str, width: int = 32, expert_dim: int = 16):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown H2 mode {mode}")
        self.mode, self.spec = mode, MODES[mode]
        self.backbone = PulseBackbone(width)
        self.global_head = nn.Linear(self.backbone.output_dim, LABEL_COUNT)
        self.saw_expert = SawRawExpert(width // 2) if self.spec.use_saw_expert else None
        self.vft_projection = nn.Sequential(nn.Linear(25, expert_dim), nn.SiLU(), nn.Linear(expert_dim, 1)) if self.spec.vft_mode != "none" else None

    def forward(self, iq: torch.Tensor, *, return_components: bool = False):
        global_logits = self.global_head(self.backbone(iq)); logits = global_logits.clone()
        saw = torch.zeros_like(logits[:, SAWFM_INDEX]); vft = torch.zeros_like(logits[:, VFT_INDEX])
        if self.saw_expert is not None:
            saw = self.saw_expert(iq); logits[:, SAWFM_INDEX] = logits[:, SAWFM_INDEX] + saw
        if self.vft_projection is not None:
            true_features, permuted_features = staggered_vft_features(iq)
            vft = self.vft_projection(true_features if self.spec.vft_mode == "true" else permuted_features).squeeze(-1)
            logits[:, VFT_INDEX] = logits[:, VFT_INDEX] + vft
        if return_components:
            return {"logits": logits, "global_logits": global_logits, "sawfm_expert_logit": saw, "vft_expert_logit": vft}
        return logits
