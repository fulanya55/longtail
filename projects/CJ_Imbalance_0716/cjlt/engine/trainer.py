"""Shared supervised trainer with auditable checkpoint selection."""

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from cjlt.engine.evaluator import compute_multilabel_metrics


@dataclass
class PredictionOutput:
    logits: np.ndarray
    targets: np.ndarray
    sample_ids: np.ndarray


@dataclass
class TrainingResult:
    best_epoch: int
    epochs_ran: int
    best_subset_acc: float
    best_hamming_loss: float
    train_seconds: float
    history: List[Dict[str, float]]


def _unpack_batch(batch, generated_offset=0):
    if isinstance(batch, dict):
        inputs = batch["signal"]
        targets = batch["label"]
        sample_ids = batch.get("sample_index")
    else:
        inputs, targets = batch[:2]
        sample_ids = None
    if sample_ids is None:
        sample_ids = torch.arange(
            generated_offset, generated_offset + len(inputs), dtype=torch.int64
        )
    elif not torch.is_tensor(sample_ids):
        sample_ids = torch.as_tensor(sample_ids, dtype=torch.int64)
    return inputs, targets, sample_ids


def predict_loader(model, loader, device, use_amp=False) -> PredictionOutput:
    model.eval()
    logits_parts = []
    target_parts = []
    id_parts = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    generated_offset = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, sample_ids = _unpack_batch(batch, generated_offset)
            inputs = inputs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(inputs)
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32))
            target_parts.append(targets.detach().cpu().numpy().astype(np.uint8))
            id_parts.append(sample_ids.detach().cpu().numpy().astype(np.int64))
            generated_offset += len(inputs)
    if not logits_parts:
        return PredictionOutput(
            logits=np.zeros((0, 0), dtype=np.float32),
            targets=np.zeros((0, 0), dtype=np.uint8),
            sample_ids=np.zeros(0, dtype=np.int64),
        )
    return PredictionOutput(
        logits=np.concatenate(logits_parts, axis=0),
        targets=np.concatenate(target_parts, axis=0),
        sample_ids=np.concatenate(id_parts, axis=0),
    )


def _validation_metrics(model, loader, device, threshold, use_amp):
    output = predict_loader(model, loader, device, use_amp=use_amp)
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(output.logits.astype(np.float64), -40.0, 40.0))
    )
    predictions = (probabilities >= float(threshold)).astype(np.uint8)
    return compute_multilabel_metrics(
        output.targets, predictions, probabilities=probabilities
    )


def _is_better(metrics, best_subset, best_hamming):
    subset = float(metrics["subset_acc"])
    hamming = float(metrics["hamming_loss"])
    if subset > best_subset:
        return True
    return bool(np.isclose(subset, best_subset) and hamming < best_hamming)


def train_model(
    model,
    train_loader,
    validation_loader,
    loss_fn,
    device,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    checkpoint_path: Optional[Path] = None,
    threshold=0.5,
    use_amp=False,
    gradient_clip=5.0,
) -> TrainingResult:
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1")
    model.to(device)
    loss_fn.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state = None
    best_epoch = 0
    best_subset = -math.inf
    best_hamming = math.inf
    bad_epochs = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            inputs, targets, _ = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(inputs)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            if float(gradient_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip)
                )
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu().item()) * len(inputs)
            sample_count += len(inputs)

        validation = _validation_metrics(
            model, validation_loader, device, threshold, use_amp
        )
        record = {
            "epoch": float(epoch),
            "train_loss": float(epoch_loss / max(sample_count, 1)),
            "subset_acc": float(validation["subset_acc"]),
            "macro_f1": float(validation["macro_f1"]),
            "hamming_loss": float(validation["hamming_loss"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(record)
        if _is_better(validation, best_subset, best_hamming):
            best_subset = float(validation["subset_acc"])
            best_hamming = float(validation["hamming_loss"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    result = TrainingResult(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_subset_acc=best_subset,
        best_hamming_loss=best_hamming,
        train_seconds=float(time.perf_counter() - started),
        history=history,
    )
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": result.best_epoch,
                "epochs_ran": result.epochs_ran,
                "best_subset_acc": result.best_subset_acc,
                "best_hamming_loss": result.best_hamming_loss,
                "train_seconds": result.train_seconds,
                "history": result.history,
            },
            str(path),
        )
    return result

