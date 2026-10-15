"""Projection wrapper and contrastive objective for paired-power pretraining."""

import torch
from torch import nn
from torch.nn import functional as F


class PowerContrastiveModel(nn.Module):
    def __init__(self, backbone, projection_dim=64, hidden_dim=128):
        super().__init__()
        if not hasattr(backbone, "feature_dim"):
            raise ValueError("backbone must expose feature_dim")
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(int(backbone.feature_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(projection_dim)),
        )
        self.power_head = nn.Sequential(
            nn.Linear(int(backbone.feature_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, signal):
        features = self.backbone.forward_features(signal)
        projection = F.normalize(self.projector(features), dim=-1)
        return features, projection

    def predict_power_gap(self, first_features, second_features):
        first = torch.as_tensor(first_features)
        second = torch.as_tensor(second_features, device=first.device)
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("power-gap features must match [batch, dimension]")
        return self.power_head(second - first).squeeze(-1)


def symmetric_contrastive_loss(first, second, temperature=0.10):
    first_value = F.normalize(torch.as_tensor(first), dim=-1)
    second_value = F.normalize(
        torch.as_tensor(second, device=first_value.device), dim=-1
    )
    if first_value.shape != second_value.shape or first_value.ndim != 2:
        raise ValueError("contrastive views must match [batch, dimension]")
    logits = first_value @ second_value.T / float(temperature)
    target = torch.arange(len(first_value), device=first_value.device)
    return 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.T, target)
    )


def multi_positive_contrastive_loss(
    first, second, identities, temperature=0.10
):
    first_value = F.normalize(torch.as_tensor(first), dim=-1)
    second_value = F.normalize(
        torch.as_tensor(second, device=first_value.device), dim=-1
    )
    identity = torch.as_tensor(
        identities, device=first_value.device, dtype=torch.long
    ).reshape(-1)
    if first_value.shape != second_value.shape or first_value.ndim != 2:
        raise ValueError("contrastive views must match [batch, dimension]")
    if identity.shape != (len(first_value),):
        raise ValueError("identities must contain one value per contrastive pair")
    positive_mask = identity[:, None].eq(identity[None, :])

    logits = first_value @ second_value.T / float(temperature)

    def direction(values, mask):
        log_probability = values - torch.logsumexp(values, dim=1, keepdim=True)
        weights = mask.to(log_probability.dtype)
        return -(
            (log_probability * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        ).mean()

    return 0.5 * (
        direction(logits, positive_mask)
        + direction(logits.T, positive_mask.T)
    )


@torch.no_grad()
def contrastive_similarity_diagnostics(first, second):
    first_value = F.normalize(torch.as_tensor(first), dim=-1)
    second_value = F.normalize(
        torch.as_tensor(second, device=first_value.device), dim=-1
    )
    similarity = first_value @ second_value.T
    positive = similarity.diagonal()
    if len(similarity) <= 1:
        negative = similarity.new_zeros(0)
    else:
        mask = ~torch.eye(len(similarity), dtype=torch.bool, device=similarity.device)
        negative = similarity[mask]
    return {
        "positive_cosine_mean": float(positive.mean().cpu().item()),
        "negative_cosine_mean": (
            float(negative.mean().cpu().item()) if len(negative) else float("nan")
        ),
        "cosine_gap": (
            float((positive.mean() - negative.mean()).cpu().item())
            if len(negative) else float("nan")
        ),
    }
