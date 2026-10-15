"""Balanced-data supervised multi-task benchmark for canonical 0724 backbones."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset, read_labels
from cj0724.multitask import (
    PARAMETER_COLUMNS,
    PARAMETER_SPECS,
    ParameterNormalizer,
    build_multitask_model,
    masked_parameter_smooth_l1,
    parameter_metrics,
    parameter_schema,
    reshape_selected_parameters,
)
from cj0724.train import get_device, multilabel_metrics, set_seed


CANONICAL_BACKBONES = (
    "complex1d",
    "iq_grouped",
    "efficientnet_b0_1d",
    "efficientnet_b1_1d",
    "convnext_tiny_1d",
)
R1_PARAMETER_VARIANTS = (
    "legacy_mlcnn_global",
    "legacy_mlcnn_class_conditioned",
    "legacy_mlcnn_class_conditioned_detached",
)


def _combination_ids(labels: np.ndarray) -> np.ndarray:
    weights = (1 << np.arange(labels.shape[1], dtype=np.int64))[None, :]
    return (labels.astype(np.int64) * weights).sum(axis=1)


def make_combination_stratified_split(
    labels: np.ndarray,
    validation_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    combo_ids = _combination_ids(labels)
    rng = np.random.default_rng(int(split_seed))
    train_parts, val_parts = [], []
    for combo_id in np.unique(combo_ids):
        group = np.flatnonzero(combo_ids == combo_id)
        group = rng.permutation(group)
        val_count = max(1, int(round(len(group) * validation_fraction)))
        val_parts.append(group[:val_count])
        train_parts.append(group[val_count:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def equal_combo_limit(
    indices: np.ndarray,
    labels: np.ndarray,
    per_combination: int | None,
    seed: int,
) -> np.ndarray:
    if per_combination is None:
        return np.asarray(indices, dtype=np.int64)
    if per_combination < 1:
        raise ValueError("per-combination limits must be positive")
    combo_ids = _combination_ids(labels)
    rng = np.random.default_rng(int(seed))
    parts = []
    for combo_id in np.unique(combo_ids[indices]):
        group = indices[combo_ids[indices] == combo_id]
        if len(group) < per_combination:
            raise ValueError(f"combination {combo_id} has only {len(group)} samples")
        parts.append(rng.permutation(group)[:per_combination])
    return np.sort(np.concatenate(parts)).astype(np.int64)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_selected_parameters(path: str | Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["parameter_values"][list(PARAMETER_COLUMNS), :], dtype=np.float32).T
        mask = np.asarray(handle["parameter_mask"][list(PARAMETER_COLUMNS), :], dtype=np.uint8).T.astype(bool)
    values = values[indices].reshape(len(indices), len(CLASS_NAMES), 2)
    mask = mask[indices].reshape(len(indices), len(CLASS_NAMES), 2)
    return values, mask


def audit_alignment(
    path: str | Path,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> dict[str, object]:
    all_indices = np.concatenate([train_indices, val_indices])
    values, mask = read_selected_parameters(path, all_indices)
    selected_labels = labels[all_indices]
    rows = []
    for flat_index, spec in enumerate(PARAMETER_SPECS):
        class_index, parameter_index = divmod(flat_index, 2)
        positive = selected_labels[:, class_index].astype(bool)
        selected_mask = mask[:, class_index, parameter_index]
        rows.append({
            "class_name": spec.class_name,
            "parameter": spec.name,
            "column_index": spec.column_index,
            "positive_samples": int(positive.sum()),
            "valid_positive": int((positive & selected_mask).sum()),
            "mask_on_negative": int((~positive & selected_mask).sum()),
            "nonfinite_valid_positive": int((positive & selected_mask & ~np.isfinite(values[:, class_index, parameter_index])).sum()),
        })
    return {
        "rows": rows,
        "all_valid_positive_finite": all(row["nonfinite_valid_positive"] == 0 for row in rows),
        "all_masks_label_aligned": all(row["mask_on_negative"] == 0 for row in rows),
    }


def fit_normalizer(path: str | Path, indices: np.ndarray, labels: np.ndarray) -> ParameterNormalizer:
    values, mask = read_selected_parameters(path, indices)
    valid = mask & (labels[indices, :, None] > 0)
    return ParameterNormalizer().fit(torch.from_numpy(values), torch.from_numpy(valid))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: ParameterNormalizer,
    parameter_weight: float,
    threshold: float,
    optimizer=None,
    scaler=None,
) -> dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, class_loss, parameter_loss = [], [], []
    logits_parts, label_parts, prediction_parts, target_parts, mask_parts = [], [], [], [], []
    criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        raw_target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
        raw_mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
        normalized_target = normalizer.transform(raw_target)
        autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(is_train):
            with autocast:
                logits, normalized_prediction = model(iq)
                cls = criterion(logits, labels)
                param = masked_parameter_smooth_l1(normalized_prediction, normalized_target, labels, raw_mask)
                loss = cls + float(parameter_weight) * param
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
        total_loss.append(float(loss.detach().cpu()))
        class_loss.append(float(cls.detach().cpu()))
        parameter_loss.append(float(param.detach().cpu()))
        logits_parts.append(logits.detach().float().cpu())
        label_parts.append(labels.detach().float().cpu())
        prediction_parts.append(normalizer.inverse_transform(normalized_prediction.detach().float().cpu()))
        target_parts.append(raw_target.detach().float().cpu())
        mask_parts.append(raw_mask.detach().cpu())
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    prediction = torch.cat(prediction_parts).numpy()
    target = torch.cat(target_parts).numpy()
    raw_mask = torch.cat(mask_parts).numpy().astype(bool)
    valid = raw_mask & (labels.numpy()[:, :, None] > 0.5)
    metrics = multilabel_metrics(logits, labels, threshold)
    probabilities = torch.sigmoid(logits).numpy()
    metrics.update({
        "macro_aupr": float(average_precision_score(labels.numpy(), probabilities, average="macro")),
        "total_loss": float(np.mean(total_loss)),
        "classification_loss": float(np.mean(class_loss)),
        "parameter_loss": float(np.mean(parameter_loss)),
        "parameter": parameter_metrics(prediction, target, valid, normalizer),
    })
    return metrics


def _compact(metrics: dict[str, object]) -> dict[str, float]:
    parameter = metrics["parameter"]
    return {
        "total_loss": float(metrics["total_loss"]),
        "classification_loss": float(metrics["classification_loss"]),
        "parameter_loss": float(metrics["parameter_loss"]),
        "macro_aupr": float(metrics["macro_aupr"]),
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "subset_acc": float(metrics["subset_acc"]),
        "parameter_macro_nmae": float(parameter["oracle_macro_target_nmae"]),
        "parameter_micro_nmae": float(parameter["oracle_micro_nmae"]),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _backbone_parameter_count(model: nn.Module) -> int:
    """Count feature-extractor parameters across unified and legacy wrappers."""
    if hasattr(model, "backbone"):
        return sum(parameter.numel() for parameter in model.backbone.parameters())
    if hasattr(model, "mlcnn"):
        return sum(parameter.numel() for parameter in model.mlcnn.parameters())
    if hasattr(model, "classification_backbone"):
        return sum(parameter.numel() for parameter in model.classification_backbone.parameters())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--models", nargs="+", choices=CANONICAL_BACKBONES + R1_PARAMETER_VARIANTS, default=list(CANONICAL_BACKBONES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--output-root", default="outputs/multitask_backbone_benchmark")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--parameter-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--parameter-weight", type=float, default=0.1)
    parser.add_argument("--parameter-warmup-epochs", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260724)
    parser.add_argument("--train-per-combination", type=int, default=None)
    parser.add_argument("--val-per-combination", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive")

    device = get_device(args.device)
    labels = read_labels(args.data_file)
    full_train, full_val = make_combination_stratified_split(labels, args.validation_fraction, args.split_seed)
    train_indices = equal_combo_limit(full_train, labels, args.train_per_combination, args.split_seed + 1)
    val_indices = equal_combo_limit(full_val, labels, args.val_per_combination, args.split_seed + 2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"balanced_multitask_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_root / "split_indices.npz", train_indices=train_indices, val_indices=val_indices)
    audit = audit_alignment(args.data_file, labels, train_indices, val_indices)
    if not audit["all_valid_positive_finite"] or not audit["all_masks_label_aligned"]:
        raise RuntimeError("parameter/label alignment audit failed; inspect audit_alignment.json")
    (run_root / "audit_alignment.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    normalizer = fit_normalizer(args.data_file, train_indices, labels)
    protocol = vars(args) | {
        "device_resolved": str(device),
        "class_names": list(CLASS_NAMES),
        "parameter_schema": parameter_schema(),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "train_combinations": int(len(np.unique(_combination_ids(labels[train_indices])))),
        "val_combinations": int(len(np.unique(_combination_ids(labels[val_indices])))),
        "train_indices_sha256": _sha256(train_indices),
        "val_indices_sha256": _sha256(val_indices),
        "normalizer": normalizer.state_dict(),
        "architecture": "raw backbone -> common 256-D adapter -> common 6-label classifier + six common 2-parameter experts",
        "loss": f"BCEWithLogits + scheduled({args.parameter_weight}) * positive-and-mask-gated SmoothL1(zscore target)",
        "checkpoint_selection": "minimum validation joint total loss",
        "long_tail_used": False,
        "class_weighting_used": False,
        "locked_test_used": False,
    }
    (run_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    train_dataset = V73IQDataset(args.data_file, train_indices, PARAMETER_COLUMNS)
    val_dataset = V73IQDataset(args.data_file, val_indices, PARAMETER_COLUMNS)
    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"START model={model_name} seed={seed}", flush=True)
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.workers, pin_memory=device.type == "cuda")
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
            model = build_multitask_model(model_name, args.width, args.dropout, args.head_dim, args.parameter_hidden_dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None
            model_dir = run_root / f"{model_name}_seed{seed}"
            model_dir.mkdir()
            history, best_total, best_epoch, stale = [], float("inf"), 0, 0
            for epoch in range(1, args.epochs + 1):
                effective_parameter_weight = 0.0 if epoch <= int(args.parameter_warmup_epochs) else float(args.parameter_weight)
                train_metrics = run_epoch(model, train_loader, device, normalizer, effective_parameter_weight, args.threshold, optimizer, scaler)
                val_metrics = run_epoch(model, val_loader, device, normalizer, effective_parameter_weight, args.threshold)
                entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                history.append(entry)
                print(json.dumps({"model": model_name, "seed": seed, "epoch": epoch, "parameter_weight": effective_parameter_weight, "train": _compact(train_metrics), "val": _compact(val_metrics)}), flush=True)
                if float(val_metrics["total_loss"]) < best_total:
                    best_total, best_epoch, stale = float(val_metrics["total_loss"]), epoch, 0
                    torch.save({"model": model_name, "seed": seed, "epoch": epoch, "model_state": model.state_dict(), "protocol": protocol, "val": val_metrics}, model_dir / "best.pt")
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            final_metrics = run_epoch(model, val_loader, device, normalizer, args.parameter_weight, args.threshold)
            (model_dir / "best_validation_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
            compact = _compact(final_metrics)
            rows.append({
                "model": model_name,
                "seed": seed,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "backbone_parameters": _backbone_parameter_count(model),
                "best_epoch": best_epoch,
                **compact,
                "checkpoint": str(model_dir / "best.pt"),
            })
            _write_csv(run_root / "per_seed_summary.csv", rows)
            (run_root / "per_seed_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            del model, optimizer, scaler, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate = []
    for model_name in args.models:
        selected = [row for row in rows if row["model"] == model_name]
        aggregate.append({
            "model": model_name,
            "seeds": len(selected),
            "parameters": selected[0]["parameters"],
            **{
                f"{metric}_{stat}": float(function([row[metric] for row in selected]))
                for metric in ("macro_aupr", "macro_f1", "subset_acc", "parameter_macro_nmae", "parameter_micro_nmae", "total_loss")
                for stat, function in (("mean", np.mean), ("std", np.std))
            },
        })
    aggregate.sort(key=lambda row: (row["parameter_macro_nmae_mean"], -row["macro_aupr_mean"]))
    _write_csv(run_root / "aggregate_summary.csv", aggregate)
    (run_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    train_dataset.close()
    val_dataset.close()
    print(f"COMPLETE {run_root}", flush=True)


if __name__ == "__main__":
    main()
