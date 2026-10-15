"""Read actual 0716 train samples and execute forward/backward for both interfaces."""

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
from cj0724.models import MODEL_NAMES, build_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = V73IQDataset(DEFAULT_TRAIN_MAT, range(8))
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    print(f"device={device}; iq={tuple(batch['iq'].shape)}; label={tuple(batch['label'].shape)}")
    for model_name in MODEL_NAMES:
        model = build_model(model_name, width=8).to(device)
        logits = model(batch["iq"].to(device))
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"].to(device))
        loss.backward()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        optimizer.step()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{model_name}: logits={tuple(logits.shape)}, loss={loss.item():.6f}, parameters={parameters}")
        del optimizer, logits, loss, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dataset.close()


if __name__ == "__main__":
    main()
