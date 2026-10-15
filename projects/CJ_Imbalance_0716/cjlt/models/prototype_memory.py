"""EMA positive/negative prototypes and collapse diagnostics."""

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class TailPrototypeMemory(nn.Module):
    def __init__(self, num_labels, embedding_dim, momentum=0.99):
        super().__init__()
        self.num_labels = int(num_labels)
        self.embedding_dim = int(embedding_dim)
        self.momentum = float(momentum)
        if self.momentum < 0.0 or self.momentum >= 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.register_buffer(
            "positive_prototype",
            torch.zeros(self.num_labels, self.embedding_dim),
        )
        self.register_buffer(
            "negative_prototype",
            torch.zeros(self.num_labels, self.embedding_dim),
        )
        self.register_buffer(
            "positive_initialized",
            torch.zeros(self.num_labels, dtype=torch.bool),
        )
        self.register_buffer(
            "negative_initialized",
            torch.zeros(self.num_labels, dtype=torch.bool),
        )
        self.register_buffer(
            "positive_count",
            torch.zeros(self.num_labels, dtype=torch.long),
        )
        self.register_buffer(
            "negative_count",
            torch.zeros(self.num_labels, dtype=torch.long),
        )

    @torch.no_grad()
    def update(self, embeddings, labels):
        if embeddings.ndim != 3 or embeddings.shape[1:] != (
            self.num_labels,
            self.embedding_dim,
        ):
            raise ValueError(
                "embeddings must have shape [B, {}, {}]".format(
                    self.num_labels, self.embedding_dim
                )
            )
        target = labels.float()
        if target.shape != embeddings.shape[:2]:
            raise ValueError("labels must match embedding batch and label dimensions")
        detached = F.normalize(embeddings.detach(), p=2, dim=-1)
        for label_index in range(self.num_labels):
            positive_mask = target[:, label_index] > 0.5
            negative_mask = ~positive_mask
            if bool(positive_mask.any()):
                mean = F.normalize(
                    detached[positive_mask, label_index].mean(dim=0), p=2, dim=0
                )
                if bool(self.positive_initialized[label_index]):
                    mean = F.normalize(
                        self.momentum * self.positive_prototype[label_index]
                        + (1.0 - self.momentum) * mean,
                        p=2,
                        dim=0,
                    )
                self.positive_prototype[label_index].copy_(mean)
                self.positive_initialized[label_index] = True
                self.positive_count[label_index] += int(positive_mask.sum().item())
            if bool(negative_mask.any()):
                mean = F.normalize(
                    detached[negative_mask, label_index].mean(dim=0), p=2, dim=0
                )
                if bool(self.negative_initialized[label_index]):
                    mean = F.normalize(
                        self.momentum * self.negative_prototype[label_index]
                        + (1.0 - self.momentum) * mean,
                        p=2,
                        dim=0,
                    )
                self.negative_prototype[label_index].copy_(mean)
                self.negative_initialized[label_index] = True
                self.negative_count[label_index] += int(negative_mask.sum().item())


def _max_offdiagonal_cosine(prototypes, initialized):
    selected = prototypes[initialized]
    if len(selected) < 2:
        return float("nan")
    normalized = F.normalize(selected, p=2, dim=-1)
    cosine = normalized @ normalized.t()
    mask = ~torch.eye(len(selected), dtype=torch.bool, device=cosine.device)
    return float(cosine[mask].max().detach().cpu().item())


def prototype_diagnostics(memory: TailPrototypeMemory) -> Dict[str, object]:
    return {
        "positive_initialized": int(memory.positive_initialized.sum().item()),
        "negative_initialized": int(memory.negative_initialized.sum().item()),
        "positive_cosine_max_offdiag": _max_offdiagonal_cosine(
            memory.positive_prototype, memory.positive_initialized
        ),
        "negative_cosine_max_offdiag": _max_offdiagonal_cosine(
            memory.negative_prototype, memory.negative_initialized
        ),
        "positive_norm_mean": float(
            memory.positive_prototype.norm(dim=-1).mean().detach().cpu().item()
        ),
        "negative_norm_mean": float(
            memory.negative_prototype.norm(dim=-1).mean().detach().cpu().item()
        ),
        "positive_update_count": [
            int(value) for value in memory.positive_count.detach().cpu().tolist()
        ],
        "negative_update_count": [
            int(value) for value in memory.negative_count.detach().cpu().tolist()
        ],
    }

