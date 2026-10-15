from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cjlt.engine.trainer import predict_loader, train_model
from cjlt.losses.classification import build_classification_loss
from cjlt.models.backbone_adapter import LightweightComplexIQClassifier


class SyntheticIQDataset(Dataset):
    def __init__(self, samples=32, length=64):
        generator = torch.Generator().manual_seed(2026)
        self.signal = torch.randn(samples, 2, length, generator=generator)
        self.label = torch.stack(
            [
                (self.signal[:, 0].mean(dim=1) > 0).float(),
                (self.signal[:, 1].mean(dim=1) > 0).float(),
            ],
            dim=1,
        )

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        return {
            "signal": self.signal[index],
            "label": self.label[index],
            "sample_index": int(index),
            "raw_rms": 1.0,
        }


def test_trainer_runs_two_epochs_and_saves_best_checkpoint(tmp_path):
    torch.manual_seed(2026)
    dataset = SyntheticIQDataset()
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = LightweightComplexIQClassifier(
        num_classes=2,
        width=4,
        dropout=0.0,
    )
    checkpoint = tmp_path / "best_model.pth"

    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=build_classification_loss("bce"),
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        checkpoint_path=checkpoint,
        threshold=0.5,
        use_amp=False,
        gradient_clip=5.0,
    )

    assert checkpoint.is_file()
    assert 1 <= result.best_epoch <= 2
    assert result.epochs_ran == 2
    assert len(result.history) == 2
    assert np.isfinite(result.best_subset_acc)
    assert np.isfinite(result.best_hamming_loss)


def test_predict_loader_returns_logits_targets_and_sample_ids():
    dataset = SyntheticIQDataset(samples=10, length=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = LightweightComplexIQClassifier(num_classes=2, width=4, dropout=0.0)

    prediction = predict_loader(model, loader, torch.device("cpu"), use_amp=False)

    assert prediction.logits.shape == (10, 2)
    assert prediction.targets.shape == (10, 2)
    assert prediction.sample_ids.tolist() == list(range(10))

