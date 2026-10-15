"""Classification losses used by the dual-axis experiments."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionBalancedLoss(nn.Module):
    """Sigmoid Distribution-Balanced Loss for multi-label long tails."""

    def __init__(
        self,
        positive_counts: Sequence[int],
        sample_count: int,
        map_alpha: float = 0.1,
        map_beta: float = 10.0,
        map_gamma: float = 0.2,
        negative_scale: float = 2.0,
        initial_bias: float = 0.05,
        focal_gamma: float = 2.0,
        balance: float = 2.0,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(positive_counts, dtype=torch.float32)
        total = int(sample_count)
        if counts.ndim != 1 or counts.numel() == 0 or bool((counts <= 0).any()):
            raise ValueError("positive_counts must contain one positive count per label")
        if total <= 0 or bool((counts >= float(total)).any()):
            raise ValueError("sample_count must exceed every positive count")
        self.register_buffer("class_frequency", counts)
        self.sample_count = total
        self.map_alpha = float(map_alpha)
        self.map_beta = float(map_beta)
        self.map_gamma = float(map_gamma)
        self.negative_scale = float(negative_scale)
        self.focal_gamma = float(focal_gamma)
        self.balance = float(balance)
        prior = counts / float(total)
        bias = -torch.log((1.0 - prior) / prior) * float(initial_bias)
        self.register_buffer("initial_logit_bias", bias / self.negative_scale)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target = targets.float()
        inverse = 1.0 / self.class_frequency
        repeat_rate = (target * inverse[None, :]).sum(dim=1, keepdim=True).clamp_min(1e-12)
        positive_weight = inverse[None, :] / repeat_rate
        weight = torch.sigmoid(
            self.map_beta * (positive_weight - self.map_gamma)
        ) + self.map_alpha
        adjusted = logits + self.initial_logit_bias[None, :]
        adjusted = adjusted * (target + (1.0 - target) * self.negative_scale)
        weight = weight * (target + (1.0 - target) / self.negative_scale)
        base = F.binary_cross_entropy_with_logits(adjusted, target, reduction="none")
        pt = torch.exp(-base)
        return (
            self.balance * torch.pow(1.0 - pt, self.focal_gamma) * base * weight
        ).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric focal loss for multi-label classification.

    Easy negatives are down-weighted more aggressively than positives. This is
    useful here because VFT-D11 contains both a rare positive label and many
    visually ambiguous negative combinations.
    """

    def __init__(
        self,
        gamma_negative: float = 4.0,
        gamma_positive: float = 0.0,
        negative_clip: float = 0.05,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if gamma_negative < 0.0 or gamma_positive < 0.0:
            raise ValueError("ASL focusing parameters must be non-negative")
        if not 0.0 <= negative_clip < 1.0:
            raise ValueError("negative_clip must be in [0, 1)")
        self.gamma_negative = float(gamma_negative)
        self.gamma_positive = float(gamma_positive)
        self.negative_clip = float(negative_clip)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target = targets.float()
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1.0 - positive_probability
        if self.negative_clip > 0.0:
            negative_probability = (negative_probability + self.negative_clip).clamp(max=1.0)

        log_likelihood = target * torch.log(positive_probability.clamp_min(self.epsilon))
        log_likelihood += (1.0 - target) * torch.log(
            negative_probability.clamp_min(self.epsilon)
        )

        positive_term = positive_probability * target
        negative_term = negative_probability * (1.0 - target)
        asymmetric_gamma = (
            self.gamma_positive * target
            + self.gamma_negative * (1.0 - target)
        )
        asymmetric_weight = torch.pow(
            (1.0 - positive_term - negative_term).clamp_min(0.0),
            asymmetric_gamma,
        )
        return -(log_likelihood * asymmetric_weight).mean()


class TargetHardNegativeLoss(nn.Module):
    """Six-label BCE with target-label OHEM and optional pairwise ranking.

    The base BCE keeps the original task unchanged. For the selected label,
    high-logit negatives receive an additional full-strength BCE penalty. When
    ranking is enabled, every positive in the batch is also required to score
    above the selected hard negatives by a fixed margin.
    """

    def __init__(
        self,
        target_index: int,
        ohem_weight: float = 0.5,
        negative_to_positive_ratio: float = 2.0,
        ranking_weight: float = 0.0,
        ranking_margin: float = 0.5,
    ) -> None:
        super().__init__()
        if int(target_index) < 0:
            raise ValueError("target_index must be non-negative")
        if ohem_weight < 0.0 or ranking_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if negative_to_positive_ratio <= 0.0:
            raise ValueError("negative_to_positive_ratio must be positive")
        if ranking_margin < 0.0:
            raise ValueError("ranking_margin must be non-negative")
        self.target_index = int(target_index)
        self.ohem_weight = float(ohem_weight)
        self.negative_to_positive_ratio = float(negative_to_positive_ratio)
        self.ranking_weight = float(ranking_weight)
        self.ranking_margin = float(ranking_margin)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target = targets.float()
        if logits.ndim != 2 or target.shape != logits.shape:
            raise ValueError("logits and targets must have the same [B, C] shape")
        if self.target_index >= logits.shape[1]:
            raise ValueError("target_index is outside the class dimension")

        base = F.binary_cross_entropy_with_logits(logits, target)
        target_logits = logits[:, self.target_index]
        target_labels = target[:, self.target_index] > 0.5
        positive_logits = target_logits[target_labels]
        negative_logits = target_logits[~target_labels]
        if negative_logits.numel() == 0:
            return base

        reference_positive_count = max(int(positive_logits.numel()), 1)
        hard_count = min(
            int(negative_logits.numel()),
            max(1, int(math.ceil(self.negative_to_positive_ratio * reference_positive_count))),
        )
        hard_negative_logits = torch.topk(negative_logits, k=hard_count).values
        hard_negative_loss = F.softplus(hard_negative_logits).mean()
        total = base + self.ohem_weight * hard_negative_loss

        if self.ranking_weight > 0.0 and positive_logits.numel() > 0:
            pairwise_margin = (
                self.ranking_margin
                - positive_logits[:, None]
                + hard_negative_logits[None, :]
            )
            ranking_loss = F.relu(pairwise_margin).mean()
            total = total + self.ranking_weight * ranking_loss
        return total


class SpecializedVFTResidualLoss(nn.Module):
    """Train bounded VFT residual experts with explicit weak-positive/negative roles."""

    def __init__(
        self,
        target_index: int = 4,
        general_aux_weight: float = 0.3,
        specialist_weight: float = 0.2,
        negative_to_positive_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.target_index = int(target_index)
        self.general_aux_weight = float(general_aux_weight)
        self.specialist_weight = float(specialist_weight)
        self.negative_to_positive_ratio = float(negative_to_positive_ratio)

    def forward(self, components: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        target = targets.float()
        fused_logits = components["logits"]
        general_logits = components["general_logits"]
        weak_positive_score = components["weak_positive_score"]
        hard_negative_score = components["hard_negative_score"]
        base = F.binary_cross_entropy_with_logits(fused_logits, target)
        general_aux = F.binary_cross_entropy_with_logits(general_logits, target)

        target_labels = target[:, self.target_index] > 0.5
        positive_indices = torch.nonzero(target_labels, as_tuple=False).squeeze(1)
        negative_indices = torch.nonzero(~target_labels, as_tuple=False).squeeze(1)
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            return base + self.general_aux_weight * general_aux

        general_target_logits = general_logits[:, self.target_index]
        hard_count = min(
            int(negative_indices.numel()),
            max(
                1,
                int(math.ceil(self.negative_to_positive_ratio * int(positive_indices.numel()))),
            ),
        )
        hard_local = torch.topk(general_target_logits[negative_indices], k=hard_count).indices
        hard_indices = negative_indices[hard_local]

        weak_positive_loss = 0.5 * (
            F.softplus(-weak_positive_score[positive_indices]).mean()
            + F.softplus(weak_positive_score[hard_indices]).mean()
        )
        hard_negative_loss = 0.5 * (
            F.softplus(-hard_negative_score[hard_indices]).mean()
            + F.softplus(hard_negative_score[positive_indices]).mean()
        )
        specialist = weak_positive_loss + hard_negative_loss
        return (
            base
            + self.general_aux_weight * general_aux
            + self.specialist_weight * specialist
        )


class PairedWeakVisibilityResidualLoss(nn.Module):
    """Weak-positive recovery with reference ranking and explicit hard negatives."""

    def __init__(
        self,
        weak_weight: float = 1.5,
        ranking_weight: float = 0.5,
        hard_negative_weight: float = 1.0,
        residual_weight: float = 0.02,
        reference_margin: float = 0.2,
        max_reference_delta: float = 2.0,
        negative_to_positive_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if min(weak_weight, ranking_weight, hard_negative_weight, residual_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if max_reference_delta <= 0.0 or negative_to_positive_ratio <= 0.0:
            raise ValueError("delta and negative ratio must be positive")
        self.weak_weight = float(weak_weight)
        self.ranking_weight = float(ranking_weight)
        self.hard_negative_weight = float(hard_negative_weight)
        self.residual_weight = float(residual_weight)
        self.reference_margin = float(reference_margin)
        self.max_reference_delta = float(max_reference_delta)
        self.negative_to_positive_ratio = float(negative_to_positive_ratio)

    def forward(
        self,
        components: dict[str, torch.Tensor],
        targets: torch.Tensor,
        weak_mask: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = components["logits"]
        base_logits = components["base_logits"]
        residual = components["residual"]
        target = targets.float()
        weak_positive = weak_mask.bool() & (target > 0.5)
        negative = target <= 0.5
        base = F.binary_cross_entropy_with_logits(logits, target)

        zero = logits.sum() * 0.0
        weak_loss = F.softplus(-logits[weak_positive]).mean() if weak_positive.any() else zero
        if weak_positive.any():
            capped_reference = torch.minimum(
                reference_logits.detach(), base_logits + self.max_reference_delta
            )
            ranking_target = capped_reference + self.reference_margin
            ranking = F.relu(ranking_target[weak_positive] - logits[weak_positive]).mean()
        else:
            ranking = zero

        hard_negative_logits = []
        for class_index in range(logits.shape[1]):
            candidates = torch.nonzero(negative[:, class_index], as_tuple=False).squeeze(1)
            if candidates.numel() == 0:
                continue
            positive_count = max(int(weak_positive[:, class_index].sum()), 1)
            hard_count = min(
                int(candidates.numel()),
                max(1, int(math.ceil(self.negative_to_positive_ratio * positive_count))),
            )
            local = torch.topk(base_logits[candidates, class_index], k=hard_count).indices
            hard_negative_logits.append(logits[candidates[local], class_index])
        hard_negative = (
            F.softplus(torch.cat(hard_negative_logits)).mean() if hard_negative_logits else zero
        )
        residual_penalty = residual.square().mean()
        total = (
            base
            + self.weak_weight * weak_loss
            + self.ranking_weight * ranking
            + self.hard_negative_weight * hard_negative
            + self.residual_weight * residual_penalty
        )
        return total, {
            "base": base.detach(),
            "weak": weak_loss.detach(),
            "ranking": ranking.detach(),
            "hard_negative": hard_negative.detach(),
            "residual": residual_penalty.detach(),
        }


class PairedSideRoutedResidualLoss(PairedWeakVisibilityResidualLoss):
    """Add train-only low/neutral/high side supervision to the residual objective."""

    def __init__(self, route_weight: float = 0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.route_weight = float(route_weight)

    def forward(
        self,
        components: dict[str, torch.Tensor],
        targets: torch.Tensor,
        weak_mask: torch.Tensor,
        reference_logits: torch.Tensor,
        levels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, parts = super().forward(components, targets, weak_mask, reference_logits)
        level = levels.to(components["route_logits"].device)
        route_target = torch.where(
            level > 30,
            torch.zeros_like(level),
            torch.where(level == 30, torch.ones_like(level), torch.full_like(level, 2)),
        ).long()
        route_loss = F.cross_entropy(components["route_logits"], route_target)
        parts["route"] = route_loss.detach()
        return total + self.route_weight * route_loss, parts


class PositiveReferenceDistillationLoss(nn.Module):
    """Weak-view BCE plus bounded reference chasing for true target positives.

    The reference logits are detached. Only target-positive examples for which
    the 30 dB reference scores above the weak view receive the extra loss. The
    weak target is capped relative to its current detached logit so a noisy
    teacher cannot create an arbitrarily large positive-response shift.
    """

    def __init__(
        self,
        target_index: int = 4,
        distillation_weight: float = 0.2,
        max_logit_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if distillation_weight < 0.0:
            raise ValueError("distillation_weight must be non-negative")
        if max_logit_delta <= 0.0:
            raise ValueError("max_logit_delta must be positive")
        self.target_index = int(target_index)
        self.distillation_weight = float(distillation_weight)
        self.max_logit_delta = float(max_logit_delta)

    def forward(
        self,
        weak_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        target = targets.float()
        if weak_logits.ndim != 2 or weak_logits.shape != target.shape:
            raise ValueError("weak_logits and targets must have the same [B, C] shape")
        if reference_logits.shape != weak_logits.shape:
            raise ValueError("reference_logits must match weak_logits")
        if self.target_index >= weak_logits.shape[1]:
            raise ValueError("target_index is outside the class dimension")

        base = F.binary_cross_entropy_with_logits(weak_logits, target)
        weak_target = weak_logits[:, self.target_index]
        reference_target = reference_logits[:, self.target_index].detach()
        positive = target[:, self.target_index] > 0.5
        eligible = positive & (reference_target > weak_target.detach())
        if self.distillation_weight == 0.0 or not eligible.any():
            return base

        bounded_target = torch.minimum(
            reference_target[eligible],
            weak_target.detach()[eligible] + self.max_logit_delta,
        )
        distillation = F.smooth_l1_loss(weak_target[eligible], bounded_target)
        return base + self.distillation_weight * distillation
