"""Strength-directed consistency for label-conditioned paired power views."""

from typing import Optional, Sequence

import torch
from torch.nn import functional as F


def _level_column(value, name, batch_size, device):
    level = torch.as_tensor(value, dtype=torch.int16, device=device)
    if level.ndim == 0:
        return level
    if level.shape != (batch_size,):
        raise ValueError("{} must be scalar or have shape [B]".format(name))
    return level[:, None]


def _direction_masks(
    labels, stronger_level_by_label, reference_level, paired_level, device
):
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    stronger = torch.as_tensor(
        stronger_level_by_label, dtype=torch.int16, device=device
    )
    if target.shape != stronger.shape or target.ndim != 2:
        raise ValueError("labels and stronger_level_by_label must match [B, C]")
    reference = _level_column(reference_level, "reference_level", len(target), device)
    paired = _level_column(paired_level, "paired_level", len(target), device)
    positive = target > 0.5
    return (
        torch.logical_and(positive, stronger == reference),
        torch.logical_and(positive, stronger == paired),
        torch.logical_and(positive, stronger == 0),
    )


def _weighted_mean(values, mask, label_weights=None):
    weights = mask.to(dtype=values.dtype)
    if label_weights is not None:
        label_weight = torch.as_tensor(
            label_weights, dtype=values.dtype, device=values.device
        )
        if label_weight.shape != (values.shape[1],):
            raise ValueError("label_weights must have one value per label")
        weights = weights * label_weight[None, :]
    denominator = weights.sum()
    if float(denominator.detach().cpu().item()) == 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / denominator


def power_curriculum_max_gap(progress: float) -> float:
    value = float(progress)
    if value < 0.0 or value > 1.0:
        raise ValueError("progress must be in [0, 1]")
    if value < 0.20:
        return 5.0
    if value < 0.45:
        return 10.0
    if value < 0.75:
        return 15.0
    return float("inf")


def label_paired_power_consistency_loss(
    reference_embeddings,
    paired_embeddings,
    labels,
    stronger_level_by_label,
    reference_level,
    paired_level,
    label_weights: Optional[Sequence[float]] = None,
):
    if reference_embeddings.shape != paired_embeddings.shape:
        raise ValueError("reference and paired embeddings must have the same shape")
    if reference_embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [B, C, D]")
    target = labels.float()
    if target.shape != reference_embeddings.shape[:2]:
        raise ValueError("labels must match embedding batch and label dimensions")
    stronger = torch.as_tensor(
        stronger_level_by_label,
        dtype=torch.int16,
        device=reference_embeddings.device,
    )
    if stronger.shape != target.shape:
        raise ValueError("stronger_level_by_label must match labels")

    reference = F.normalize(reference_embeddings, p=2, dim=-1)
    paired = F.normalize(paired_embeddings, p=2, dim=-1)
    reference_is_strong, paired_is_strong, _ = _direction_masks(
        target,
        stronger,
        reference_level,
        paired_level,
        reference_embeddings.device,
    )

    loss_reference_strong = 1.0 - (
        paired * reference.detach()
    ).sum(dim=-1)
    loss_paired_strong = 1.0 - (
        reference * paired.detach()
    ).sum(dim=-1)
    mask_reference = reference_is_strong.float()
    mask_paired = paired_is_strong.float()
    if label_weights is not None:
        weights = torch.as_tensor(
            label_weights,
            dtype=reference_embeddings.dtype,
            device=reference_embeddings.device,
        )
        if weights.shape != (reference_embeddings.shape[1],):
            raise ValueError("label_weights must have one value per label")
        mask_reference = mask_reference * weights[None, :]
        mask_paired = mask_paired * weights[None, :]
    denominator = (mask_reference + mask_paired).sum()
    if float(denominator.detach().cpu().item()) == 0.0:
        return (reference_embeddings.sum() + paired_embeddings.sum()) * 0.0
    numerator = (
        loss_reference_strong * mask_reference
        + loss_paired_strong * mask_paired
    ).sum()
    return numerator / denominator


def directional_power_ordering_loss(
    reference_logits,
    paired_logits,
    labels,
    stronger_level_by_label,
    reference_level,
    paired_level,
    margin=0.20,
    label_weights: Optional[Sequence[float]] = None,
    mode="bidirectional",
):
    """Enforce the known component-strength direction in logit space.

    A positive label whose component is stronger in one view must have a logit
    at least ``margin`` larger than in the weaker view.
    """

    reference = torch.as_tensor(reference_logits)
    paired = torch.as_tensor(paired_logits, device=reference.device)
    if reference.shape != paired.shape or reference.ndim != 2:
        raise ValueError("reference_logits and paired_logits must match [B, C]")
    reference_strong, paired_strong, _ = _direction_masks(
        labels,
        stronger_level_by_label,
        reference_level,
        paired_level,
        reference.device,
    )
    mode_value = str(mode)
    if mode_value not in ("bidirectional", "strong_only"):
        raise ValueError("mode must be bidirectional or strong_only")
    if mode_value == "strong_only":
        loss_reference = F.relu(float(margin) - (reference - paired.detach()))
        loss_paired = F.relu(float(margin) - (paired - reference.detach()))
    else:
        loss_reference = F.relu(float(margin) - (reference - paired))
        loss_paired = F.relu(float(margin) - (paired - reference))
    values = torch.where(reference_strong, loss_reference, loss_paired)
    return _weighted_mean(
        values,
        torch.logical_or(reference_strong, paired_strong),
        label_weights=label_weights,
    )


def unchanged_component_stability_loss(
    reference_logits,
    paired_logits,
    labels,
    stronger_level_by_label,
    reference_level,
    paired_level,
    tolerance=0.25,
    label_weights: Optional[Sequence[float]] = None,
):
    """Penalize excessive confidence drift for unchanged positive components."""

    reference = torch.as_tensor(reference_logits)
    paired = torch.as_tensor(paired_logits, device=reference.device)
    if reference.shape != paired.shape or reference.ndim != 2:
        raise ValueError("reference_logits and paired_logits must match [B, C]")
    _, _, unchanged = _direction_masks(
        labels,
        stronger_level_by_label,
        reference_level,
        paired_level,
        reference.device,
    )
    probability_drift = torch.abs(torch.sigmoid(reference) - torch.sigmoid(paired))
    values = F.relu(probability_drift - float(tolerance))
    return _weighted_mean(values, unchanged, label_weights=label_weights)


@torch.no_grad()
def paired_power_diagnostic_values(
    reference_logits,
    paired_logits,
    labels,
    stronger_level_by_label,
    reference_level,
    paired_level,
):
    """Return per-position signed margins and unchanged probability drifts.

    Positive signed margins mean the physically stronger view has the larger
    logit.  The returned one-dimensional tensors are intended for logging only.
    """

    reference = torch.as_tensor(reference_logits)
    paired = torch.as_tensor(paired_logits, device=reference.device)
    if reference.shape != paired.shape or reference.ndim != 2:
        raise ValueError("reference_logits and paired_logits must match [B, C]")
    reference_strong, paired_strong, unchanged = _direction_masks(
        labels,
        stronger_level_by_label,
        reference_level,
        paired_level,
        reference.device,
    )
    signed = torch.where(
        reference_strong,
        reference - paired,
        paired - reference,
    )
    directional = torch.logical_or(reference_strong, paired_strong)
    drift = torch.abs(torch.sigmoid(reference) - torch.sigmoid(paired))
    return signed[directional].detach(), drift[unchanged].detach()
