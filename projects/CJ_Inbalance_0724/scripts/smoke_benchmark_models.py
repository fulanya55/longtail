"""Verify the unified classifier/loss path for every registered backbone."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cj0724 import DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset
from cj0724.models import MODEL_NAMES, build_benchmark_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = V73IQDataset(DEFAULT_TRAIN_MAT, range(4))
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    for name in MODEL_NAMES:
        model = build_benchmark_model(name, width=8, head_dim=64).to(device)
        logits = model(batch["iq"].to(device))
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"].to(device))
        loss.backward()
        print(f"{name}: logits={tuple(logits.shape)}, loss={loss.item():.6f}")
        del logits, loss, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dataset.close()


if __name__ == "__main__":
    main()
