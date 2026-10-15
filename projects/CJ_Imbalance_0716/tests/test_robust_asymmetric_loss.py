from __future__ import annotations

import torch

from cjlt.losses.classification import RobustAsymmetricLoss


def _official_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(x); positive, negative = probability, 1.0 - probability
    negative = (negative + 0.05).clamp(max=1.0)
    pos = y * (torch.log(positive.clamp(min=1e-8)) + (1.0 - positive.clamp(min=1e-8)) - 1.25 * (1.0 - positive.clamp(min=1e-8)).square())
    neg = (1.0 - y) * torch.log(negative.clamp(min=1e-8)) * (1.5 - probability) * probability.square() * (1.5 - negative)
    pt = positive * y + negative * (1.0 - y); gamma = 4.0 * (1.0 - y)
    return (-(pos + neg) * (1.0 - pt).pow(gamma)).sum()


def test_ral_sum_matches_pinned_official_formula() -> None:
    torch.manual_seed(11); logits = torch.randn(4, 6); targets = torch.randint(0, 2, (4, 6)).float()
    assert torch.allclose(RobustAsymmetricLoss(reduction="sum")(logits, targets), _official_sum(logits, targets))


def test_ral_mean_is_only_constant_normalization() -> None:
    torch.manual_seed(12); logits = torch.randn(3, 6); targets = torch.randint(0, 2, (3, 6)).float()
    assert torch.allclose(RobustAsymmetricLoss(reduction="mean")(logits, targets) * targets.numel(), _official_sum(logits, targets))
