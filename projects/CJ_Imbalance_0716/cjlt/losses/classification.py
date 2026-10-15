"""BCE, focal, ASL, and class-balanced multilabel losses."""

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def effective_number_weights(
    positive_counts: Sequence[int],
    beta: float = 0.99,
    minimum: float = 0.5,
    maximum: float = 8.0,
) -> np.ndarray:
    counts = np.asarray(positive_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        raise ValueError("positive_counts must be a nonempty sequence of positive values")
    beta_value = float(beta)
    if beta_value <= 0.0 or beta_value >= 1.0:
        raise ValueError("beta must be in (0, 1)")
    lower = float(minimum)
    upper = float(maximum)
    if lower <= 0.0 or upper < lower or not (lower <= 1.0 <= upper):
        raise ValueError("weight bounds must satisfy 0 < minimum <= 1 <= maximum")
    raw = (1.0 - beta_value) / (1.0 - np.power(beta_value, counts))

    low_scale = 0.0
    high_scale = max(1.0, 1.0 / raw.min())
    while np.clip(raw * high_scale, lower, upper).mean() < 1.0:
        high_scale *= 2.0
    for _ in range(80):
        middle = 0.5 * (low_scale + high_scale)
        mean = np.clip(raw * middle, lower, upper).mean()
        if mean < 1.0:
            low_scale = middle
        else:
            high_scale = middle
    weights = np.clip(raw * high_scale, lower, upper)
    return weights.astype(np.float32)


class MultiLabelClassificationLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        label_weights: Optional[Sequence[float]] = None,
        gamma_pos: float = 0.0,
        gamma_neg: float = 3.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.mode = str(mode)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        if label_weights is None:
            self.register_buffer("label_weights", torch.empty(0))
        else:
            values = torch.as_tensor(label_weights, dtype=torch.float32)
            self.register_buffer("label_weights", values)

    def elementwise(self, logits, targets):
        target = targets.float()
        if logits.shape != target.shape:
            raise ValueError("logits and targets must have matching shapes")
        if self.mode in ("bce", "cb_bce"):
            loss = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
        else:
            probability = torch.sigmoid(logits)
            if self.mode == "focal":
                base = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                focal_weight = (
                    target * torch.pow(1.0 - probability, self.gamma_pos)
                    + (1.0 - target) * torch.pow(probability, self.gamma_neg)
                )
                loss = base * focal_weight
            elif self.mode in ("asl", "cb_asl"):
                positive_probability = probability
                negative_probability = 1.0 - probability
                if self.clip > 0.0:
                    negative_probability = torch.clamp(
                        negative_probability + self.clip, max=1.0
                    )
                log_likelihood = (
                    target * torch.log(torch.clamp(positive_probability, min=1e-8))
                    + (1.0 - target)
                    * torch.log(torch.clamp(negative_probability, min=1e-8))
                )
                pt = (
                    target * positive_probability
                    + (1.0 - target) * negative_probability
                )
                gamma = (
                    target * self.gamma_pos
                    + (1.0 - target) * self.gamma_neg
                )
                loss = -log_likelihood * torch.pow(1.0 - pt, gamma)
            else:
                raise RuntimeError("Unsupported loss mode: {}".format(self.mode))

        if self.label_weights.numel():
            if self.label_weights.shape != (logits.shape[1],):
                raise ValueError("label weight count must match logits columns")
            loss = loss * self.label_weights.view(1, -1)
        return loss

    def forward(self, logits, targets):
        return self.elementwise(logits, targets).mean()


class DistributionBalancedLoss(nn.Module):
    """Compact sigmoid Distribution-Balanced Loss for multi-label tails."""

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
    ):
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

    def elementwise(self, logits, targets):
        target = targets.float()
        if logits.shape != target.shape:
            raise ValueError("logits and targets must have matching shapes")
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
        loss = self.balance * torch.pow(1.0 - pt, self.focal_gamma) * base * weight
        return loss

    def forward(self, logits, targets):
        return self.elementwise(logits, targets).mean()


class ASLDBMixedLoss(nn.Module):
    """Convex ASL/DB interpolation for preserving tails under extreme IF."""

    def __init__(
        self,
        positive_counts: Sequence[int],
        sample_count: int,
        db_weight: float,
        gamma_pos: float = 0.0,
        gamma_neg: float = 3.0,
        clip: float = 0.05,
    ):
        super().__init__()
        weight = float(db_weight)
        if weight < 0.0 or weight > 1.0:
            raise ValueError("db_weight must be in [0, 1]")
        self.db_weight = weight
        self.asl = MultiLabelClassificationLoss(
            mode="asl",
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
        self.db = DistributionBalancedLoss(
            positive_counts=positive_counts,
            sample_count=sample_count,
        )

    def forward(self, logits, targets):
        asl_loss = self.asl(logits, targets)
        db_loss = self.db(logits, targets)
        return (1.0 - self.db_weight) * asl_loss + self.db_weight * db_loss


class RobustAsymmetricLoss(nn.Module):
    """Formula-equivalent multi-label Ralloss (ICCVW 2023 official code).

    ``mean`` only divides the official summed objective by a positive,
    batch-constant element count so it shares the project-wide loss scale.
    No published RAL coefficient is selected or changed here.
    """

    def __init__(
        self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05,
        eps: float = 1e-8, lamb: float = 1.5, epsilon_neg: float = 0.0,
        epsilon_pos: float = 1.0, epsilon_pos_pow: float = -2.5,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = map(float, (gamma_neg, gamma_pos, clip, eps))
        self.lamb, self.epsilon_neg, self.epsilon_pos = map(float, (lamb, epsilon_neg, epsilon_pos))
        self.epsilon_pos_pow, self.reduction = float(epsilon_pos_pow), reduction

    def forward(self, logits, targets):
        target = targets.float()
        if logits.shape != target.shape:
            raise ValueError("logits and targets must have matching shapes")
        probability = torch.sigmoid(logits); positive, negative = probability, 1.0 - probability
        if self.clip > 0.0:
            negative = (negative + self.clip).clamp(max=1.0)
        positive_term = target * (
            torch.log(positive.clamp(min=self.eps)) + self.epsilon_pos * (1.0 - positive.clamp(min=self.eps))
            + self.epsilon_pos_pow * 0.5 * torch.pow(1.0 - positive.clamp(min=self.eps), 2)
        )
        negative_term = (1.0 - target) * (
            torch.log(negative.clamp(min=self.eps)) + self.epsilon_neg * negative.clamp(min=self.eps)
        ) * (self.lamb - probability) * probability.square() * (self.lamb - negative)
        pt = positive * target + negative * (1.0 - target)
        gamma = self.gamma_pos * target + self.gamma_neg * (1.0 - target)
        value = -(positive_term + negative_term) * torch.pow(1.0 - pt, gamma)
        return value.sum() if self.reduction == "sum" else value.mean()


class TailConditionalASLDBLoss(nn.Module):
    """Use DB on one registered tail output and ASL on all other outputs."""

    def __init__(
        self,
        positive_counts: Sequence[int],
        sample_count: int,
        tail_label_index: int,
        tail_weight: float = 1.0,
        gamma_pos: float = 0.0,
        gamma_neg: float = 3.0,
        clip: float = 0.05,
    ):
        super().__init__()
        counts = list(positive_counts)
        index = int(tail_label_index)
        if index < 0 or index >= len(counts):
            raise ValueError("tail_label_index must identify one label column")
        weight = float(tail_weight)
        if weight <= 0.0:
            raise ValueError("tail_weight must be positive")
        self.tail_label_index = index
        self.tail_weight = weight
        self.asl = MultiLabelClassificationLoss(
            mode="asl",
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
        self.db = DistributionBalancedLoss(
            positive_counts=counts,
            sample_count=sample_count,
        )

    def forward(self, logits, targets):
        asl = self.asl.elementwise(logits, targets)
        db = self.db.elementwise(logits, targets)
        loss = asl.clone()
        loss[:, self.tail_label_index] = (
            self.tail_weight * db[:, self.tail_label_index]
        )
        return loss.mean()


def build_classification_loss(
    name: str,
    positive_counts: Optional[Sequence[int]] = None,
    gamma_pos: float = 0.0,
    gamma_neg: float = 3.0,
    clip: float = 0.05,
    beta: float = 0.99,
    sample_count: Optional[int] = None,
    db_mix_weight: float = 0.5,
    tail_label_index: Optional[int] = None,
    tail_loss_weight: float = 1.0,
) -> nn.Module:
    normalized = str(name).strip().lower()
    valid = (
        "bce",
        "focal",
        "asl",
        "cb_bce",
        "cb_asl",
        "db",
        "asl_db_mix",
        "tail_asl_db",
    )
    if normalized not in valid:
        raise ValueError("Unknown classification loss: {!r}".format(name))
    if normalized == "db":
        if positive_counts is None or sample_count is None:
            raise ValueError("positive_counts and sample_count are required for DB loss")
        return DistributionBalancedLoss(positive_counts, sample_count)
    if normalized == "asl_db_mix":
        if positive_counts is None or sample_count is None:
            raise ValueError("positive_counts and sample_count are required for ASL-DB mix")
        return ASLDBMixedLoss(
            positive_counts=positive_counts,
            sample_count=sample_count,
            db_weight=db_mix_weight,
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
    if normalized == "tail_asl_db":
        if positive_counts is None or sample_count is None:
            raise ValueError(
                "positive_counts and sample_count are required for tail ASL-DB"
            )
        if tail_label_index is None:
            raise ValueError("tail_label_index is required for tail ASL-DB")
        return TailConditionalASLDBLoss(
            positive_counts=positive_counts,
            sample_count=sample_count,
            tail_label_index=tail_label_index,
            tail_weight=tail_loss_weight,
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
    weights = None
    if normalized.startswith("cb_"):
        if positive_counts is None:
            raise ValueError("positive_counts are required for class-balanced loss")
        weights = effective_number_weights(positive_counts, beta=beta)
    return MultiLabelClassificationLoss(
        mode=normalized,
        label_weights=weights,
        gamma_pos=gamma_pos,
        gamma_neg=gamma_neg,
        clip=clip,
    )
