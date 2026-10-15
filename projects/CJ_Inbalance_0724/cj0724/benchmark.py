"""Matched backbone comparison with one split, one head, one loss, and one training recipe."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset, make_train_val_split, read_labels
from cj0724.models import DEFAULT_UNIFIED_HEAD_DIM, MODEL_NAMES, build_benchmark_model
from cj0724.train import get_device, run_epoch, set_seed


def _selection_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    # Compound recognition prioritizes the exact six-label set, then macro-F1.
    return metrics["subset_acc"], metrics["macro_f1"], -metrics["loss"]


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "model",
        "parameters",
        "backbone_parameters",
        "best_epoch",
        "val_subset_acc",
        "val_macro_f1",
        "val_micro_f1",
        "val_loss",
        "checkpoint",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-root", default="outputs/backbone_benchmark")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_UNIFIED_HEAD_DIM)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--legacy-attention", default="se")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.head_dim < 1:
        raise ValueError("epochs, batch-size, and head-dim must be positive")
    device = get_device(args.device)
    labels = read_labels(args.data_file)
    train_indices, val_indices = make_train_val_split(
        len(labels),
        args.validation_fraction,
        args.seed,
        args.max_train_samples,
        args.max_val_samples,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"seed{args.seed}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_root / "split_indices.npz", train_indices=train_indices, val_indices=val_indices)
    protocol = vars(args) | {
        "device_resolved": str(device),
        "class_names": CLASS_NAMES,
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "classifier": f"Linear(raw_dim,{args.head_dim}) + LayerNorm + GELU + Dropout({args.dropout}) + Linear({args.head_dim},6)",
        "loss": "BCEWithLogitsLoss",
        "checkpoint_selection": "val subset_acc, then macro_f1, then loss",
        "locked_test_used": False,
    }
    (run_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    train_dataset = V73IQDataset(args.data_file, train_indices)
    val_dataset = V73IQDataset(args.data_file, val_indices)
    summary_rows: list[dict[str, object]] = []

    for model_name in args.models:
        print(f"START model={model_name}")
        set_seed(args.seed)
        shuffle_generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=shuffle_generator,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        model = build_benchmark_model(
            model_name,
            num_classes=len(CLASS_NAMES),
            width=args.width,
            dropout=args.dropout,
            head_dim=args.head_dim,
            legacy_attention=args.legacy_attention,
        ).to(device)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None
        model_dir = run_root / model_name
        model_dir.mkdir()
        history = []
        best_metrics = None
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, args.threshold, optimizer, scaler)
            val_metrics = run_epoch(model, val_loader, criterion, device, args.threshold)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(row)
            print(json.dumps({"model": model_name, **row}, ensure_ascii=False))
            if best_metrics is None or _selection_key(val_metrics) > _selection_key(best_metrics):
                best_metrics = dict(val_metrics)
                best_epoch = epoch
                torch.save(
                    {
                        "model": model_name,
                        "model_state": model.state_dict(),
                        "protocol": protocol,
                        "epoch": epoch,
                        "val": val_metrics,
                    },
                    model_dir / "best.pt",
                )
        (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "model": model_name,
                "parameters": total_parameters,
                "backbone_parameters": backbone_parameters,
                "best_epoch": best_epoch,
                "val_subset_acc": best_metrics["subset_acc"],
                "val_macro_f1": best_metrics["macro_f1"],
                "val_micro_f1": best_metrics["micro_f1"],
                "val_loss": best_metrics["loss"],
                "checkpoint": str(model_dir / "best.pt"),
            }
        )
        _write_summary_csv(run_root / "summary.csv", summary_rows)
        (run_root / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        del optimizer, scaler, criterion, model, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_dataset.close()
    val_dataset.close()
    from cj0724.evaluation import evaluate_benchmark_run

    evaluate_benchmark_run(run_root, device_name=str(device), workers=args.workers)
    print(f"Benchmark complete: {run_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
