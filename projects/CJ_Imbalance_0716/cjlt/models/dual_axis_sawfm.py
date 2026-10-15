"""SawFM-scoped modular routing for Route-A frequency/visibility studies."""

from __future__ import annotations

import torch
from torch import nn


class RoutedSawFMModel(nn.Module):
    """Replace only the SawFM logit with a raw-I/Q expert logit.

    The module intentionally preserves all other core logits exactly.  This
    matches the historical RawSaw evidence scope and prevents a label-specific
    visibility mechanism from silently changing the other five decisions.
    """

    def __init__(self, core: nn.Module, sawfm_expert: nn.Module, sawfm_index: int = 1):
        super().__init__()
        self.core = core
        self.sawfm_expert = sawfm_expert
        self.sawfm_index = int(sawfm_index)
        if self.sawfm_index < 0:
            raise ValueError("sawfm_index must be nonnegative")

    def forward(self, signal: torch.Tensor, return_components: bool = False):
        if signal.ndim != 3 or signal.shape[1] != 2:
            raise ValueError("signal must have shape [B,2,T]")
        core_logits = self.core(signal)
        if core_logits.ndim != 2 or self.sawfm_index >= core_logits.shape[1]:
            raise ValueError("core output does not contain the SawFM column")
        sawfm_logit = self.sawfm_expert(signal[:, 0:1], signal[:, 1:2])
        if sawfm_logit.shape != (signal.shape[0],):
            raise ValueError("SawFM expert must return one logit per sample")
        routed = torch.cat(
            [
                core_logits[:, : self.sawfm_index],
                sawfm_logit[:, None],
                core_logits[:, self.sawfm_index + 1 :],
            ],
            dim=1,
        )
        if return_components:
            return routed, {"core_logits": core_logits, "sawfm_logit": sawfm_logit}
        return routed
