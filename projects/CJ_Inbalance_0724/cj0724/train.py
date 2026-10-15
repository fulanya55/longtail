"""Reproducible supervised training for the two controlled I/Q interface models."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset, make_train_val_split, read_labels
from cj0724.models import DEFAULT_UNIFIED_HEAD_DIM, MODEL_NAMES, build_benchmark_model, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def multilabel_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold) -> dict[str, float]:
    threshold_tensor = torch.as_tensor(threshold, dtype=logits.dtype, device=logits.device)
    if threshold_tensor.ndim > 1 or (threshold_tensor.ndim == 1 and threshold_tensor.numel() != logits.shape[1]):
        raise ValueError("threshold must be scalar or have one value per class")
    predicted = torch.sigmoid(logits) >= threshold_tensor
    target = labels >= 0.5
    exact_subset = predicted.eq(target).all(dim=1).float().mean()
    tp = (predicted & target).sum().float()
    fp = (predicted & ~target).sum().float()
    fn = (~predicted & target).sum().float()
    micro_f1 = (2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)).item()
    per_label_tp = (predicted & target).sum(dim=0).float()
    per_label_fp = (predicted & ~target).sum(dim=0).float()
    per_label_fn = (~predicted & target).sum(dim=0).float()
    macro_f1 = (2.0 * per_label_tp / (2.0 * per_label_tp + per_label_fp + per_label_fn).clamp_min(1.0)).mean().item()
    return {"subset_acc": exact_subset.item(), "micro_f1": micro_f1, "macro_f1": macro_f1}


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, threshold: float, optimizer=None, scaler=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses, all_logits, all_labels = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        autocast = torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(is_train):
            with autocast:
                logits = model(iq)
                loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = multilabel_metrics(logits, labels, threshold)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-mode", choices=("unified", "native"), default="unified")
    parser.add_argument("--head-dim", type=int, default=DEFAULT_UNIFIED_HEAD_DIM)
    parser.add_argument(
        "--legacy-attention",
        default="se",
        choices=("se", "eca", "cbam", "coordatt", "sk", "sk_eca", "pcaf", "gc", "none"),
        help="Attention used only by legacy_effnet_v3; other backbones ignore it.",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    set_seed(args.seed)
    device = get_device(args.device)
    labels = read_labels(args.data_file)
    train_indices, val_indices = make_train_val_split(
        len(labels), args.validation_fraction, args.seed, args.max_train_samples, args.max_val_samples
    )
    train_dataset = V73IQDataset(args.data_file, train_indices)
    val_dataset = V73IQDataset(args.data_file, val_indices)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    builder = build_benchmark_model if args.head_mode == "unified" else build_model
    builder_arguments = {
        "name": args.model,
        "num_classes": len(CLASS_NAMES),
        "width": args.width,
        "dropout": args.dropout,
        "legacy_attention": args.legacy_attention,
    }
    if args.head_mode == "unified":
        builder_arguments["head_dim"] = args.head_dim
    model = builder(**builder_arguments).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.model}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"device_resolved": str(device), "class_names": CLASS_NAMES, "train_samples": int(len(train_indices)), "val_samples": int(len(val_indices))}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "split_indices.npz", train_indices=train_indices, val_indices=val_indices)

    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, args.threshold, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, device, args.threshold)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({"model": args.model, "model_state": model.state_dict(), "config": config, "epoch": epoch, "val": val_metrics}, run_dir / "best.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint to {run_dir / 'best.pt'}; best_val_macro_f1={best_macro_f1:.6f}")
    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    main()
