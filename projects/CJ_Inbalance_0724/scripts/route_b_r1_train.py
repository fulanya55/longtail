"""Preregistered Route-B R1 development trainer.

The command requires ``--authorize-training`` so a smoke/preflight invocation
cannot accidentally consume the full GPU budget.  It creates only a
development split; no test-v2 or locked-test file is discovered or opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset, read_labels
from cj0724.models import build_benchmark_model
from cj0724.multitask import PARAMETER_COLUMNS, build_multitask_model
from cj0724.multitask_benchmark import (
    equal_combo_limit,
    fit_normalizer,
    make_combination_stratified_split,
    run_epoch,
)
from cj0724.train import get_device, multilabel_metrics, set_seed


VARIANTS = {
    "B0": ("classification", "legacy_mlcnn", 0),
    "B1": ("joint", "legacy_mlcnn_global", 0),
    "B2": ("joint", "legacy_mlcnn_class_conditioned", 0),
    "B3": ("joint", "legacy_mlcnn_class_conditioned_detached", 0),
    "B4": ("joint", "legacy_mlcnn_class_conditioned", 2),
}


def classification_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    criterion = nn.BCEWithLogitsLoss()
    logits_parts, labels_parts, losses = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(iq)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        logits_parts.append(logits.detach().float().cpu())
        labels_parts.append(labels.detach().float().cpu())
        losses.append(float(loss.detach().cpu()))
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    probabilities = torch.sigmoid(logits).numpy()
    metrics = multilabel_metrics(logits, labels, 0.5)
    metrics.update({
        "macro_aupr": float(average_precision_score(labels.numpy(), probabilities, average="macro")),
        "total_loss": float(np.mean(losses)),
        "classification_loss": float(np.mean(losses)),
        "parameter_loss": float("nan"),
        "parameter_macro_nmae": float("nan"),
        "parameter_micro_nmae": float("nan"),
    })
    return metrics


def compact(metrics):
    parameter = metrics.get("parameter")
    return {
        "total_loss": float(metrics["total_loss"]),
        "classification_loss": float(metrics["classification_loss"]),
        "parameter_loss": float(metrics["parameter_loss"]),
        "macro_aupr": float(metrics["macro_aupr"]),
        "macro_f1": float(metrics["macro_f1"]),
        "subset_acc": float(metrics["subset_acc"]),
        "parameter_macro_nmae": float(metrics.get("parameter_macro_nmae", parameter["oracle_macro_target_nmae"] if parameter else np.nan)),
        "parameter_micro_nmae": float(metrics.get("parameter_micro_nmae", parameter["oracle_micro_nmae"] if parameter else np.nan)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043, 2044, 2045])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--parameter-hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--parameter-weight", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260826)
    parser.add_argument("--train-per-combination", type=int, default=None)
    parser.add_argument("--val-per-combination", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/route_b_r1_full_development")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.authorize_training:
        raise SystemExit("Refusing full R1 training without --authorize-training")
    if len(args.seeds) < 5:
        raise ValueError("R1 requires at least five development seeds")

    device = get_device(args.device)
    labels = read_labels(args.data_file)
    full_train, full_val = make_combination_stratified_split(labels, 0.1, args.split_seed)
    train_indices = equal_combo_limit(full_train, labels, args.train_per_combination, args.split_seed + 1)
    val_indices = equal_combo_limit(full_val, labels, args.val_per_combination, args.split_seed + 2)
    normalizer = fit_normalizer(args.data_file, train_indices, labels)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"r1_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_root / "split_indices.npz", train_indices=train_indices, val_indices=val_indices)
    protocol = vars(args) | {
        "device_resolved": str(device),
        "class_names": list(CLASS_NAMES),
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "development_only": True,
        "locked_test_used": False,
        "test_generated": False,
        "test_opened": False,
        "parameter_valid_rule": "label=1 AND parameter_mask=1 AND target finite",
        "variants": VARIANTS,
    }
    (run_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    train_joint = V73IQDataset(args.data_file, train_indices, PARAMETER_COLUMNS)
    val_joint = V73IQDataset(args.data_file, val_indices, PARAMETER_COLUMNS)
    train_cls = V73IQDataset(args.data_file, train_indices)
    val_cls = V73IQDataset(args.data_file, val_indices)
    rows = []
    for variant in args.variants:
        kind, model_name, warmup_epochs = VARIANTS[variant]
        for seed in args.seeds:
            print(f"START variant={variant} model={model_name} seed={seed}", flush=True)
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            if kind == "classification":
                model = build_benchmark_model(model_name, num_classes=6, width=args.width, head_dim=args.head_dim).to(device)
                train_loader = DataLoader(train_cls, batch_size=args.batch_size, shuffle=True, generator=generator)
                val_loader = DataLoader(val_cls, batch_size=args.batch_size, shuffle=False)
            else:
                model = build_multitask_model(model_name, width=args.width, head_dim=args.head_dim, parameter_hidden_dim=args.parameter_hidden_dim).to(device)
                train_loader = DataLoader(train_joint, batch_size=args.batch_size, shuffle=True, generator=generator)
                val_loader = DataLoader(val_joint, batch_size=args.batch_size, shuffle=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            best = float("inf")
            stale = 0
            history = []
            model_dir = run_root / f"{variant}_seed{seed}"
            model_dir.mkdir()
            for epoch in range(1, args.epochs + 1):
                if kind == "classification":
                    train_metrics = classification_epoch(model, train_loader, device, optimizer)
                    val_metrics = classification_epoch(model, val_loader, device)
                else:
                    weight = 0.0 if epoch <= warmup_epochs else args.parameter_weight
                    train_metrics = run_epoch(model, train_loader, device, normalizer, weight, 0.5, optimizer)
                    val_metrics = run_epoch(model, val_loader, device, normalizer, weight, 0.5)
                effective_weight = 0.0 if kind == "classification" or epoch <= warmup_epochs else args.parameter_weight
                entry = {"epoch": epoch, "parameter_weight": effective_weight, "train": compact(train_metrics), "val": compact(val_metrics)}
                history.append(entry)
                print(json.dumps({"variant": variant, "seed": seed, **entry}), flush=True)
                objective = float(val_metrics["total_loss"])
                if objective < best:
                    best, stale = objective, 0
                    torch.save({"variant": variant, "seed": seed, "epoch": epoch, "model_state": model.state_dict(), "protocol": protocol, "val": val_metrics}, model_dir / "best.pt")
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            if kind == "classification":
                final = classification_epoch(model, val_loader, device)
            else:
                final = run_epoch(model, val_loader, device, normalizer, args.parameter_weight, 0.5)
            row = {"variant": variant, "model": model_name, "seed": seed, "best_epoch": checkpoint["epoch"], "parameters": sum(p.numel() for p in model.parameters()), **compact(final), "evidence_status": "verified_development", "locked_test_used": False}
            rows.append(row)
            with (run_root / "per_seed_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    train_joint.close(); val_joint.close(); train_cls.close(); val_cls.close()
    (run_root / "result_receipt.json").write_text(json.dumps({"stage": "R1", "status": "verified_development", "rows": rows, "locked_test_used": False, "test_generated": False, "test_opened": False}, indent=2), encoding="utf-8")
    print(f"COMPLETE {run_root}", flush=True)


if __name__ == "__main__":
    main()
