"""Lightweight label-conditioned embeddings used only by auxiliary branches."""

import torch
from torch import nn
from torch.nn import functional as F


class LabelConditionedFeatureBank(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_labels,
        relation_dim=64,
        dropout=0.1,
        conditioning_mode="gated",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)
        self.relation_dim = int(relation_dim)
        self.conditioning_mode = str(conditioning_mode).strip().lower()
        if self.conditioning_mode not in ("gated", "independent"):
            raise ValueError("conditioning_mode must be 'gated' or 'independent'")
        self.projection = None
        self.label_embedding = None
        self.label_projections = None
        if self.conditioning_mode == "gated":
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, self.relation_dim),
                nn.LayerNorm(self.relation_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            self.label_embedding = nn.Parameter(
                torch.empty(self.num_labels, self.relation_dim)
            )
            nn.init.normal_(self.label_embedding, mean=0.0, std=0.02)
        else:
            self.label_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.feature_dim, self.relation_dim),
                        nn.LayerNorm(self.relation_dim),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                    )
                    for _ in range(self.num_labels)
                ]
            )

    def gates(self):
        if self.conditioning_mode != "gated":
            raise RuntimeError("gates are only defined for gated conditioning")
        return 1.0 + torch.tanh(self.label_embedding)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "features must have shape [B, {}], got {}".format(
                    self.feature_dim, tuple(features.shape)
                )
            )
        if self.conditioning_mode == "gated":
            projected = self.projection(features)
            embeddings = projected[:, None, :] * self.gates()[None, :, :]
        else:
            embeddings = torch.stack(
                [projection(features) for projection in self.label_projections],
                dim=1,
            )
        return F.normalize(embeddings, p=2, dim=-1)
