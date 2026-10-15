"""Frozen-feature classifiers for label/combination frequency reconstruction."""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrozenLinearHead(nn.Module):
    """A six-label linear head initialized from a trained B0 classifier."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("linear head weight/bias shapes are inconsistent")
        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def classifier_state(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the final linear classifier without depending on dropout state."""

    linear = None
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            linear = module
    if linear is None or linear.bias is None:
        raise ValueError("B0 classifier does not expose a biased linear layer")
    return linear.weight.detach().clone(), linear.bias.detach().clone()


def _combo_code(row: np.ndarray) -> int:
    return int(sum(int(value) << index for index, value in enumerate(row)))


def build_pair_map(
    labels: np.ndarray,
    seed: int,
    mode: Literal["hard", "random"],
) -> np.ndarray:
    """Map every positive label occurrence to one legal negative identity.

    Hard pairs maximize Jaccard similarity over the other five labels.  Ties
    are broken only with the frozen seed.  Random pairs use the same number of
    legal pairs and sample uniformly from all negatives.
    """

    matrix = np.asarray(labels, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("labels must be [N,K] with K>=2")
    if mode not in ("hard", "random"):
        raise ValueError("pair mode must be hard or random")
    rng = np.random.RandomState(int(seed))
    result = np.full(matrix.shape, -1, dtype=np.int64)
    combo_groups: dict[int, np.ndarray] = {}
    for code in sorted({_combo_code(row) for row in matrix}):
        combo_groups[code] = np.flatnonzero(
            np.asarray([_combo_code(row) == code for row in matrix], dtype=bool)
        )
    for label_index in range(matrix.shape[1]):
        positives = np.flatnonzero(matrix[:, label_index] == 1)
        negatives = np.flatnonzero(matrix[:, label_index] == 0)
        if not len(positives) or not len(negatives):
            raise ValueError(f"label {label_index} lacks positive or negative samples")
        if mode == "random":
            shuffled = negatives.copy()
            rng.shuffle(shuffled)
            result[positives, label_index] = np.resize(shuffled, len(positives))
            continue
        negative_codes = [code for code, values in combo_groups.items() if matrix[values[0], label_index] == 0]
        pools: dict[int, np.ndarray] = {}
        for positive_code, positive_values in combo_groups.items():
            if matrix[positive_values[0], label_index] == 0:
                continue
            positive_row = matrix[positive_values[0]].copy()
            positive_row[label_index] = 0
            scored: list[tuple[float, int]] = []
            for negative_code in negative_codes:
                negative_row = matrix[combo_groups[negative_code][0]].copy()
                negative_row[label_index] = 0
                union = np.logical_or(positive_row, negative_row).sum()
                score = 1.0 if union == 0 else float(np.logical_and(positive_row, negative_row).sum() / union)
                scored.append((score, negative_code))
            best_score = max(score for score, _ in scored)
            tied = [code for score, code in scored if abs(score - best_score) <= 1e-12]
            chosen_code = tied[int(rng.randint(len(tied)))]
            pool = combo_groups[chosen_code].copy()
            rng.shuffle(pool)
            pools[positive_code] = pool
            ordered_positive = positive_values.copy()
            rng.shuffle(ordered_positive)
            result[ordered_positive, label_index] = np.resize(pool, len(ordered_positive))
    if np.any(result[matrix == 1] < 0):
        raise AssertionError("not every positive occurrence received a pair")
    rows, labels_positive = np.nonzero(matrix == 1)
    paired = result[rows, labels_positive]
    if np.any(matrix[paired, labels_positive] != 0):
        raise AssertionError("pair map contains an illegal positive target")
    return result


def label_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_counts: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """BCE whose expected epoch mass is 0.5 positive/0.5 negative per label."""

    positive = positive_counts.to(device=logits.device, dtype=logits.dtype)
    negative = float(sample_count) - positive
    if torch.any(positive <= 0) or torch.any(negative <= 0):
        raise ValueError("label-balanced BCE requires positive and negative samples")
    positive_weight = 0.5 * float(sample_count) / positive
    negative_weight = 0.5 * float(sample_count) / negative
    weights = targets * positive_weight + (1.0 - targets) * negative_weight
    return (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).mean()


def pair_ranking_loss(
    head: FrozenLinearHead,
    features: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,
    pair_map: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    current_logits = head(features[positions])
    terms = []
    for label_index in range(targets.shape[1]):
        selected = targets[:, label_index] == 1
        if not torch.any(selected):
            continue
        positive_positions = positions[selected]
        negative_positions = pair_map[positive_positions, label_index]
        positive_logits = current_logits[selected, label_index]
        negative_logits = head(features[negative_positions])[:, label_index]
        terms.append(F.softplus(float(margin) - positive_logits + negative_logits))
    if not terms:
        return current_logits.sum() * 0.0
    return torch.cat(terms).mean()


def normalized_anchor_loss(
    head: FrozenLinearHead,
    anchor_weight: torch.Tensor,
    anchor_bias: torch.Tensor,
) -> torch.Tensor:
    weight_scale = anchor_weight.pow(2).mean().clamp_min(1e-12)
    bias_scale = anchor_bias.pow(2).mean().clamp_min(1e-12)
    return (
        (head.linear.weight - anchor_weight).pow(2).mean() / weight_scale
        + (head.linear.bias - anchor_bias).pow(2).mean() / bias_scale
    )


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return copy.deepcopy({name: value.detach().cpu() for name, value in module.state_dict().items()})
