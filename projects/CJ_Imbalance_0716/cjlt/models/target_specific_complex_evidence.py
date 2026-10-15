"""Target-specific raw complex local-evidence maps with a recoverable global path."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cjlt.models.complex_layers import ComplexConv1d, complex_magnitude


TARGET_INDICES = (3, 4, 5)  # MFT, VFT, ISRJ in the fixed six-label schema.


class TargetSpecificComplexEvidenceClassifier(nn.Module):
    """Six-label global classifier plus three pre-classifier local evidence branches.

    ``source_control=True`` retains every learnable tensor but cycles target local
    branches. This is a semantic-source capacity control, not a smaller model.
    """

    def __init__(self, channels: int = 8, topk: int = 16, source_control: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.topk = int(topk)
        self.source_control = bool(source_control)
        if self.channels < 2 or self.topk <= 0:
            raise ValueError("channels must be >=2 and topk positive")
        self.stem = ComplexConv1d(1, self.channels, kernel_size=17, stride=4, padding=8)
        self.stem_norm = nn.BatchNorm1d(self.channels)
        self.global_classifier = nn.Linear(2 * self.channels, 6)
        self.local_filters = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(self.channels, self.channels, kernel, padding=kernel // 2, groups=self.channels, bias=False)
                        for kernel in (9, 33, 129)
                    ]
                )
                for _ in TARGET_INDICES
            ]
        )
        self.local_mixers = nn.ModuleList([nn.Conv1d(3 * self.channels, self.channels, 1) for _ in TARGET_INDICES])
        self.evidence_heads = nn.ModuleList([nn.Conv1d(self.channels, 1, 1) for _ in TARGET_INDICES])
        self.local_classifiers = nn.ModuleList([nn.Linear(2, 1) for _ in TARGET_INDICES])
        self.local_scale = nn.Parameter(torch.zeros(len(TARGET_INDICES)))

    def _global_features(self, carrier: torch.Tensor) -> torch.Tensor:
        return torch.cat([carrier.mean(dim=-1), carrier.std(dim=-1, unbiased=False)], dim=1)

    def _branch(self, carrier: torch.Tensor, branch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filtered = torch.cat([F.silu(layer(carrier)) for layer in self.local_filters[branch_index]], dim=1)
        local = F.silu(self.local_mixers[branch_index](filtered))
        evidence = self.evidence_heads[branch_index](local).squeeze(1)
        k = min(self.topk, int(evidence.shape[-1]))
        pooled = torch.stack([evidence.mean(dim=-1), evidence.topk(k, dim=-1).values.mean(dim=-1)], dim=1)
        return evidence, self.local_classifiers[branch_index](pooled).squeeze(1)

    def forward(self, iq: torch.Tensor, return_evidence: bool = False):
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError("iq must have shape [B,2,T]")
        real, imag = self.stem(iq[:, :1], iq[:, 1:])
        carrier = F.silu(self.stem_norm(complex_magnitude(real, imag)))
        global_logits = self.global_classifier(self._global_features(carrier))
        local_maps, local_logits = [], []
        for branch_index in range(len(TARGET_INDICES)):
            evidence, local_logit = self._branch(carrier, branch_index)
            local_maps.append(evidence); local_logits.append(local_logit)
        output = global_logits.clone()
        route = (1, 2, 0) if self.source_control else (0, 1, 2)
        for target_position, target_index in enumerate(TARGET_INDICES):
            output[:, target_index] = global_logits[:, target_index] + torch.tanh(self.local_scale[target_position]) * local_logits[route[target_position]]
        if return_evidence:
            return {"logits": output, "global_logits": global_logits, "local_logits": torch.stack(local_logits, dim=1), "local_evidence": torch.stack(local_maps, dim=1), "carrier": carrier}
        return output
