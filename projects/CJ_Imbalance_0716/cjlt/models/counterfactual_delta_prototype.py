"""Counterfactual temporal-difference prototypes with single-view inference."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

TARGET_INDICES=(3,4,5)


class CounterfactualDeltaPrototypeProbe(nn.Module):
    """Six-label global path plus bounded target-specific prototype evidence."""

    def __init__(self, width: int = 16, projection_dim: int = 16) -> None:
        super().__init__()
        base=int(width); dim=int(projection_dim)
        if base < 4 or dim < 4: raise ValueError("width and projection_dim must be >=4")
        self.encoder=nn.Sequential(
            nn.Conv1d(2,base,15,stride=4,padding=7,bias=False),nn.GroupNorm(4,base),nn.GELU(),
            nn.Conv1d(base,2*base,9,stride=4,padding=4,bias=False),nn.GroupNorm(4,2*base),nn.GELU(),
            nn.Conv1d(2*base,3*base,7,stride=4,padding=3,bias=False),nn.GroupNorm(4,3*base),nn.GELU(),
        )
        self.global_head=nn.Sequential(nn.LayerNorm(6*base),nn.Linear(6*base,2*base),nn.GELU(),nn.Linear(2*base,6))
        self.projector=nn.Sequential(nn.LayerNorm(3*base),nn.Linear(3*base,dim),nn.GELU(),nn.Linear(dim,dim))
        self.prototypes=nn.Parameter(torch.empty(3,dim)); nn.init.orthogonal_(self.prototypes)
        self.prototype_scale=nn.Parameter(torch.zeros(3))

    def features(self, iq: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        if iq.ndim!=3 or iq.shape[1]!=2: raise ValueError("iq must have shape [B,2,T]")
        temporal=self.encoder(iq.float()); pooled=torch.cat([temporal.mean(-1),F.adaptive_max_pool1d(temporal,1).flatten(1)],1)
        return temporal.mean(-1), self.global_head(pooled)

    def embedding(self, iq: torch.Tensor) -> torch.Tensor:
        pooled,_=self.features(iq); return F.normalize(self.projector(pooled),dim=1)

    def forward(self, iq: torch.Tensor, return_features: bool=False):
        pooled,global_logits=self.features(iq); embedding=F.normalize(self.projector(pooled),dim=1); prototypes=F.normalize(self.prototypes,dim=1)
        output=global_logits.clone(); output[:,TARGET_INDICES]=global_logits[:,TARGET_INDICES]+torch.tanh(self.prototype_scale)[None,:]*(embedding@prototypes.T)
        if return_features:return {'logits':output,'global_logits':global_logits,'embedding':embedding,'prototypes':prototypes}
        return output
