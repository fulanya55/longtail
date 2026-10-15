"""P3o-J: bounded shared delay residual with detached posterior component conditioning."""
from __future__ import annotations

import torch
from torch import nn

from cj0724.p3og_bounded_combo_cvar import BoundedStatePreservingResidual
from cj0724.p3ob_dualview_representation import delay_peak_sequence


class ComponentConditionedResidual(BoundedStatePreservingResidual):
    """Adds a 6x12 posterior-selected correction without changing the frozen classifier."""
    def __init__(self, base, residual_bound: float = 0.25) -> None:
        super().__init__(base, residual_bound)
        self.component = nn.Parameter(torch.zeros(6, 12))

    def forward_with_embedding(self, iq: torch.Tensor):
        with torch.no_grad():
            logits, base_parameter, embedding = self.base.forward_with_embedding(iq)
        delay = self.delay(delay_peak_sequence(iq))
        shared = self.residual(torch.cat((embedding, delay), dim=1))
        selected = torch.sigmoid(logits).detach() @ self.component
        residual = self.residual_bound * torch.tanh(shared + selected).reshape(-1, 6, 2)
        return logits, base_parameter + torch.sigmoid(logits).detach().unsqueeze(-1) * residual, embedding

    def set_component_enabled(self, enabled: bool) -> None:
        """J0 is the exact shared-residual control; J1 alone trains this tensor."""
        self.component.requires_grad_(bool(enabled))
