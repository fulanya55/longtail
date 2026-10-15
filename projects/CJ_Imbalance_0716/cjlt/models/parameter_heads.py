"""Independent label-gated parameter regressors."""

import torch
from torch import nn


class LabelGatedParameterHeads(nn.Module):
    def __init__(self, num_labels, embedding_dim, parameters_per_label=2, hidden_dim=32):
        super().__init__()
        self.num_labels = int(num_labels)
        self.embedding_dim = int(embedding_dim)
        self.parameters_per_label = int(parameters_per_label)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embedding_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), self.parameters_per_label),
                )
                for _ in range(self.num_labels)
            ]
        )

    def forward(self, label_embeddings):
        if label_embeddings.ndim != 3 or label_embeddings.shape[1:] != (
            self.num_labels,
            self.embedding_dim,
        ):
            raise ValueError("label_embeddings have incompatible shape")
        outputs = [
            head(label_embeddings[:, index, :])
            for index, head in enumerate(self.heads)
        ]
        return torch.stack(outputs, dim=1)

