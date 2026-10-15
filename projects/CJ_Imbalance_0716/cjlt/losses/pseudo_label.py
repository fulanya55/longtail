"""Per-label pseudo-label filtering and reliability scoring."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class PseudoLabelBatch:
    targets: torch.Tensor
    accepted_mask: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    reliability: torch.Tensor


def prototype_reliability(
    embeddings,
    positive_prototypes,
    negative_prototypes,
    temperature=1.0,
):
    features = F.normalize(embeddings, p=2, dim=-1)
    positive = F.normalize(
        torch.as_tensor(positive_prototypes, device=features.device, dtype=features.dtype),
        p=2,
        dim=-1,
    )
    negative = F.normalize(
        torch.as_tensor(negative_prototypes, device=features.device, dtype=features.dtype),
        p=2,
        dim=-1,
    )
    margin = (features * positive[None]).sum(dim=-1) - (
        features * negative[None]
    ).sum(dim=-1)
    return torch.sigmoid(margin / float(temperature))


def build_pseudo_labels(
    probability,
    tau_pos: Sequence[float],
    tau_neg: Sequence[float],
    reliability: Optional[torch.Tensor] = None,
    reliability_threshold=0.0,
    stability: Optional[torch.Tensor] = None,
    stability_ceiling=float("inf"),
    require_reliability_for_negative=False,
    max_positive_fraction=1.0,
    max_negative_fraction=1.0,
):
    values = torch.as_tensor(probability).float()
    positive_threshold = torch.as_tensor(
        tau_pos, dtype=values.dtype, device=values.device
    )
    negative_threshold = torch.as_tensor(
        tau_neg, dtype=values.dtype, device=values.device
    )
    if positive_threshold.shape != (values.shape[1],) or negative_threshold.shape != (
        values.shape[1],
    ):
        raise ValueError("thresholds must provide one value per label")
    if reliability is None:
        reliability_value = torch.ones_like(values)
    else:
        reliability_value = torch.as_tensor(
            reliability, dtype=values.dtype, device=values.device
        )
    if reliability_value.shape != values.shape:
        raise ValueError("reliability must match probability")
    if stability is None:
        stable = torch.ones_like(values, dtype=torch.bool)
    else:
        stability_value = torch.as_tensor(
            stability, dtype=values.dtype, device=values.device
        )
        if stability_value.shape != values.shape:
            raise ValueError("stability must match probability")
        stable = stability_value <= float(stability_ceiling)
    positive = torch.logical_and(
        values >= positive_threshold[None, :],
        reliability_value >= float(reliability_threshold),
    )
    positive = torch.logical_and(positive, stable)
    negative = torch.logical_and(values <= negative_threshold[None, :], stable)
    if bool(require_reliability_for_negative):
        negative = torch.logical_and(
            negative,
            reliability_value >= float(reliability_threshold),
        )
    positive = _cap_mask_per_label(
        positive, values, max_positive_fraction, largest=True
    )
    negative = _cap_mask_per_label(
        negative, values, max_negative_fraction, largest=False
    )
    accepted = torch.logical_or(positive, negative)
    targets = positive.float()
    return PseudoLabelBatch(
        targets=targets,
        accepted_mask=accepted,
        positive_mask=positive,
        negative_mask=negative,
        reliability=reliability_value,
    )


def _cap_mask_per_label(mask, score, maximum_fraction, largest):
    fraction = float(maximum_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("maximum pseudo-label fraction must be in (0, 1]")
    if fraction >= 1.0:
        return mask
    sample_count = int(mask.shape[0])
    limit = max(1, int(np.floor(sample_count * fraction)))
    capped = torch.zeros_like(mask)
    for label_index in range(mask.shape[1]):
        candidates = torch.nonzero(
            mask[:, label_index], as_tuple=False
        ).reshape(-1)
        if len(candidates) <= limit:
            capped[candidates, label_index] = True
            continue
        candidate_scores = score[candidates, label_index]
        selected = torch.topk(
            candidate_scores, k=limit, largest=bool(largest), sorted=False
        ).indices
        capped[candidates[selected], label_index] = True
    return capped


def masked_pseudo_label_bce(logits, targets, accepted_mask, balance_sides=False):
    target = targets.float()
    mask = accepted_mask.float()
    if logits.shape != target.shape or mask.shape != target.shape:
        raise ValueError("logits, targets, and accepted_mask must match")
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if not bool(balance_sides):
        denominator = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denominator
    positive = mask * target
    negative = mask * (1.0 - target)
    positive_count = positive.sum()
    negative_count = negative.sum()
    pieces = []
    if float(positive_count.detach().cpu().item()) > 0.0:
        pieces.append((loss * positive).sum() / positive_count)
    if float(negative_count.detach().cpu().item()) > 0.0:
        pieces.append((loss * negative).sum() / negative_count)
    if not pieces:
        return loss.sum() * 0.0
    return torch.stack(pieces).mean()


class DynamicPseudoLabelThresholds:
    def __init__(
        self,
        num_labels,
        tau_pos=0.95,
        tau_neg=0.05,
        tau_pos_floor=0.70,
        tau_neg_ceiling=0.20,
    ):
        self.tau_pos = np.broadcast_to(
            np.asarray(tau_pos, dtype=np.float32), (int(num_labels),)
        ).copy()
        self.tau_neg = np.broadcast_to(
            np.asarray(tau_neg, dtype=np.float32), (int(num_labels),)
        ).copy()
        self.tau_pos_floor = float(tau_pos_floor)
        self.tau_neg_ceiling = float(tau_neg_ceiling)

    def update(
        self,
        positive_precision,
        positive_coverage,
        target_precision=0.95,
        minimum_coverage=0.10,
    ):
        precision = np.asarray(positive_precision, dtype=np.float64)
        coverage = np.asarray(positive_coverage, dtype=np.float64)
        if precision.shape != self.tau_pos.shape or coverage.shape != self.tau_pos.shape:
            raise ValueError("precision and coverage must have one value per label")
        for index in range(len(self.tau_pos)):
            if precision[index] < float(target_precision):
                self.tau_pos[index] += 0.02
            elif coverage[index] < float(minimum_coverage):
                self.tau_pos[index] -= 0.01
        self.tau_pos = np.clip(self.tau_pos, self.tau_pos_floor, 0.999)
        self.tau_neg = np.clip(self.tau_neg, 0.001, self.tau_neg_ceiling)
