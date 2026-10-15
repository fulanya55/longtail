"""Label-wise positive/negative prototype contrastive objective."""

from typing import Optional, Sequence

import torch
from torch.nn import functional as F

from cjlt.models.prototype_memory import TailPrototypeMemory


def label_prototype_contrastive_loss(
    embeddings,
    labels,
    memory: TailPrototypeMemory,
    label_weights: Optional[Sequence[float]] = None,
    temperature=0.1,
):
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [B, C, D]")
    target = labels.float()
    if target.shape != embeddings.shape[:2]:
        raise ValueError("labels must match embedding batch and label dimensions")
    if embeddings.shape[1:] != (memory.num_labels, memory.embedding_dim):
        raise ValueError("embedding dimensions do not match prototype memory")
    scale = float(temperature)
    if scale <= 0.0:
        raise ValueError("temperature must be positive")

    valid_labels = torch.logical_and(
        memory.positive_initialized, memory.negative_initialized
    )
    if not bool(valid_labels.any()):
        return embeddings.sum() * 0.0
    positive = memory.positive_prototype.to(embeddings.device)
    negative = memory.negative_prototype.to(embeddings.device)
    similarity_positive = (embeddings * positive[None, :, :]).sum(dim=-1)
    similarity_negative = (embeddings * negative[None, :, :]).sum(dim=-1)
    logits = torch.stack(
        [similarity_negative, similarity_positive], dim=-1
    ) / scale
    loss = F.cross_entropy(
        logits.reshape(-1, 2),
        target.long().reshape(-1),
        reduction="none",
    ).reshape_as(target)
    valid = valid_labels.to(embeddings.device).float()[None, :]
    if label_weights is not None:
        weights = torch.as_tensor(
            label_weights, dtype=embeddings.dtype, device=embeddings.device
        )
        if weights.shape != (memory.num_labels,):
            raise ValueError("label_weights must have one value per label")
        valid = valid * weights[None, :]
    denominator = valid.expand_as(loss).sum().clamp_min(1.0)
    return (loss * valid).sum() / denominator

