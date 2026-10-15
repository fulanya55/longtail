"""Small supervised multi-task tuning sweep with leakage-safe threshold calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from cj0724.models import LEGACY_MODEL_NAMES
from cj0724.multitask import (
    MLCNN_PARAMETER_INPUT_MODELS,
    TASK_ROUTED_MULTITASK_MODELS,
    PARAMETER_COLUMNS,
    ParameterNormalizer,
    build_multitask_model,
    masked_parameter_smooth_l1,
    parameter_metrics,
    reshape_selected_parameters,
)
from cj0724.multitask_benchmark import (
    CANONICAL_BACKBONES,
    _combination_ids,
    audit_alignment,
    equal_combo_limit,
    fit_normalizer,
)
from cj0724.train import get_device, multilabel_metrics, set_seed


MULTITASK_BACKBONES = (
    CANONICAL_BACKBONES + tuple(LEGACY_MODEL_NAMES)
    + MLCNN_PARAMETER_INPUT_MODELS + TASK_ROUTED_MULTITASK_MODELS
)
PROFILE_CONFIGS = {
    "cos_lr1e4_wd1e5": {"scheduler": "cosine", "learning_rate": 1.0e-4, "weight_decay": 1.0e-5},
    "cos_lr1e4_wd1e4": {"scheduler": "cosine", "learning_rate": 1.0e-4, "weight_decay": 1.0e-4},
    "cos_lr3e4_wd1e5": {"scheduler": "cosine", "learning_rate": 3.0e-4, "weight_decay": 1.0e-5},
    "cos_lr3e4_wd1e4": {"scheduler": "cosine", "learning_rate": 3.0e-4, "weight_decay": 1.0e-4},
    "plateau_lr1e4_wd1e4": {"scheduler": "plateau", "learning_rate": 1.0e-4, "weight_decay": 1.0e-4},
}


def parameter_loss_target_weights(model_name: str) -> torch.Tensor:
    """Return frozen per-target training weights without changing inference."""
    weights = torch.ones((len(CLASS_NAMES), 2), dtype=torch.float32)
    if model_name == "mlcnn_iq_grouped_task_routed_saw_loss2":
        weights[1, 1] = 2.0
    return weights


def make_five_way_combination_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Split every combination into 300/40/20/20/20 train/early/cal/tune/confirm."""
    combo_ids = _combination_ids(labels)
    rng = np.random.default_rng(int(seed))
    parts = {role: [] for role in ("train", "early", "calibration", "tune", "confirm")}
    for combo_id in np.unique(combo_ids):
        group = rng.permutation(np.flatnonzero(combo_ids == combo_id))
        if len(group) < 20:
            raise ValueError("each combination needs at least 20 samples")
        early_count = max(1, int(round(len(group) * 0.10)))
        role_count = max(1, int(round(len(group) * 0.05)))
        cursor = 0
        parts["early"].append(group[cursor : cursor + early_count])
        cursor += early_count
        for role in ("calibration", "tune", "confirm"):
            parts[role].append(group[cursor : cursor + role_count])
            cursor += role_count
        parts["train"].append(group[cursor:])
    return {role: np.sort(np.concatenate(values)).astype(np.int64) for role, values in parts.items()}


def limit_roles(
    roles: dict[str, np.ndarray],
    labels: np.ndarray,
    per_combination: dict[str, int | None],
    seed: int,
) -> dict[str, np.ndarray]:
    return {
        role: equal_combo_limit(indices, labels, per_combination[role], seed + offset)
        for offset, (role, indices) in enumerate(roles.items(), start=1)
    }


def optimize_class_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Choose one F1-maximizing threshold per class; ties prefer values nearest 0.5."""
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if probability.shape != target.shape or probability.ndim != 2:
        raise ValueError("probabilities and labels must match [N, C]")
    candidates = np.asarray(grid if grid is not None else np.linspace(0.05, 0.95, 91), dtype=np.float64)
    thresholds = np.full(probability.shape[1], 0.5, dtype=np.float64)
    for class_index in range(probability.shape[1]):
        scores = []
        for threshold in candidates:
            predicted = probability[:, class_index] >= threshold
            expected = target[:, class_index]
            tp = np.logical_and(predicted, expected).sum()
            fp = np.logical_and(predicted, ~expected).sum()
            fn = np.logical_and(~predicted, expected).sum()
            scores.append(2.0 * tp / max(2.0 * tp + fp + fn, 1))
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        nearest = best[np.argmin(np.abs(candidates[best] - 0.5))]
        thresholds[class_index] = candidates[nearest]
    return thresholds.astype(np.float32)


def _set_warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    min_ratio: float = 0.05,
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        ratio = epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs - 1) / max(epochs - warmup_epochs - 1, 1)
        ratio = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    learning_rate = float(base_lr) * ratio
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: ParameterNormalizer,
    parameter_weight: float,
    parameter_target_weights: torch.Tensor | None = None,
) -> dict[str, object]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    logits_parts, labels_parts, prediction_parts, target_parts, mask_parts = [], [], [], [], []
    total_losses, class_losses, parameter_losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            raw_target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
            raw_mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
            normalized_target = normalizer.transform(raw_target)
            autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
            with autocast:
                logits, normalized_prediction = model(iq)
                classification_loss = criterion(logits, labels)
                parameter_loss = masked_parameter_smooth_l1(
                    normalized_prediction, normalized_target, labels, raw_mask,
                    parameter_target_weights,
                )
                total_loss = classification_loss + float(parameter_weight) * parameter_loss
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels.float().cpu())
            prediction_parts.append(normalizer.inverse_transform(normalized_prediction.float().cpu()))
            target_parts.append(raw_target.float().cpu())
            mask_parts.append(raw_mask.cpu())
            total_losses.append(float(total_loss.cpu()))
            class_losses.append(float(classification_loss.cpu()))
            parameter_losses.append(float(parameter_loss.cpu()))
    return {
        "logits": torch.cat(logits_parts),
        "labels": torch.cat(labels_parts),
        "prediction": torch.cat(prediction_parts).numpy(),
        "target": torch.cat(target_parts).numpy(),
        "mask": torch.cat(mask_parts).numpy().astype(bool),
        "total_loss": float(np.mean(total_losses)),
        "classification_loss": float(np.mean(class_losses)),
        "parameter_loss": float(np.mean(parameter_losses)),
    }


def _metrics(outputs: dict[str, object], thresholds, normalizer: ParameterNormalizer) -> dict[str, object]:
    logits = outputs["logits"]
    labels = outputs["labels"]
    result = multilabel_metrics(logits, labels, thresholds)
    probabilities = torch.sigmoid(logits).numpy()
    valid = outputs["mask"] & (labels.numpy()[:, :, None] > 0.5)
    result.update({
        "macro_aupr": float(average_precision_score(labels.numpy(), probabilities, average="macro")),
        "total_loss": outputs["total_loss"],
        "classification_loss": outputs["classification_loss"],
        "parameter_loss": outputs["parameter_loss"],
        "parameter": parameter_metrics(outputs["prediction"], outputs["target"], valid, normalizer),
    })
    return result


def _compact(metrics: dict[str, object]) -> dict[str, float]:
    return {
        "total_loss": float(metrics["total_loss"]),
        "macro_aupr": float(metrics["macro_aupr"]),
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "subset_acc": float(metrics["subset_acc"]),
        "parameter_macro_nmae": float(metrics["parameter"]["oracle_macro_target_nmae"]),
        "parameter_micro_nmae": float(metrics["parameter"]["oracle_micro_nmae"]),
        "parameter_macro_pearson": float(metrics["parameter"]["oracle_macro_pearson_r"]),
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: ParameterNormalizer,
    parameter_weight: float,
    optimizer: torch.optim.Optimizer,
    scaler,
    parameter_target_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
        mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
        normalized_target = normalizer.transform(target)
        autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with autocast:
            logits, prediction = model(iq)
            loss = criterion(logits, labels) + float(parameter_weight) * masked_parameter_smooth_l1(
                prediction, normalized_target, labels, mask, parameter_target_weights
            )
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
    return float(np.mean(losses))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--models", nargs="+", choices=MULTITASK_BACKBONES, required=True)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILE_CONFIGS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041])
    parser.add_argument("--output-root", default="outputs/tuned_multitask_sweep")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--parameter-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--parameter-weight", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260724)
    parser.add_argument("--train-per-combination", type=int, default=None)
    parser.add_argument("--early-per-combination", type=int, default=None)
    parser.add_argument("--calibration-per-combination", type=int, default=None)
    parser.add_argument("--tune-per-combination", type=int, default=None)
    parser.add_argument("--confirm-per-combination", type=int, default=None)
    parser.add_argument("--legacy-attention", default="se")
    parser.add_argument(
        "--checkpoint-objectives",
        nargs="+",
        choices=("joint", "classification", "parameter"),
        default=["joint"],
        help="Evaluate checkpoints selected by early joint loss, macro-AUPR, or parameter NMAE.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs must be smaller than epochs")

    device = get_device(args.device)
    labels = read_labels(args.data_file)
    roles = make_five_way_combination_split(labels, args.split_seed)
    roles = limit_roles(roles, labels, {
        "train": args.train_per_combination,
        "early": args.early_per_combination,
        "calibration": args.calibration_per_combination,
        "tune": args.tune_per_combination,
        "confirm": args.confirm_per_combination,
    }, args.split_seed + 100)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"tuned_multitask_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_root / "role_indices.npz", **roles)
    audit = audit_alignment(args.data_file, labels, roles["train"], np.concatenate([roles[x] for x in ("early", "calibration", "tune", "confirm")]))
    if not audit["all_valid_positive_finite"] or not audit["all_masks_label_aligned"]:
        raise RuntimeError("alignment audit failed")
    (run_root / "audit_alignment.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    normalizer = fit_normalizer(args.data_file, roles["train"], labels)
    role_hashes = {f"{role}_sha256": hashlib.sha256(indices.tobytes()).hexdigest() for role, indices in roles.items()}
    protocol = vars(args) | {
        "device_resolved": str(device),
        "profile_configs": {name: PROFILE_CONFIGS[name] for name in args.profiles},
        "role_samples": {role: len(indices) for role, indices in roles.items()},
        "role_combinations": {role: int(len(np.unique(_combination_ids(labels[indices])))) for role, indices in roles.items()},
        "role_hashes": role_hashes,
        "threshold_calibration": "per-class F1 grid 0.05:0.01:0.95 on calibration role only",
        "hyperparameter_selection_role": "tune",
        "unbiased_within_root_confirmation_role": "confirm",
        "normalizer": normalizer.state_dict(),
        "parameter_loss_target_weights": {
            model_name: parameter_loss_target_weights(model_name).tolist()
            for model_name in args.models
        },
        "long_tail_used": False,
        "locked_test_used": False,
    }
    (run_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    datasets = {role: V73IQDataset(args.data_file, indices, PARAMETER_COLUMNS) for role, indices in roles.items()}
    rows = []
    for model_name in args.models:
        for profile_name in args.profiles:
            profile = PROFILE_CONFIGS[profile_name]
            for seed in args.seeds:
                print(f"START model={model_name} profile={profile_name} seed={seed}", flush=True)
                set_seed(seed)
                generator = torch.Generator().manual_seed(seed)
                loaders = {
                    role: DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=role == "train",
                        generator=generator if role == "train" else None,
                        num_workers=args.workers,
                        pin_memory=device.type == "cuda",
                    )
                    for role, dataset in datasets.items()
                }
                model = build_multitask_model(
                    model_name, args.width, args.dropout, args.head_dim,
                    args.parameter_hidden_dim, args.legacy_attention,
                ).to(device)
                target_weights = parameter_loss_target_weights(model_name).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=profile["learning_rate"], weight_decay=profile["weight_decay"]
                )
                plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=args.plateau_patience, min_lr=profile["learning_rate"] * 0.05
                ) if profile["scheduler"] == "plateau" else None
                scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None
                run_name = f"{model_name}__{profile_name}__seed{seed}"
                model_dir = run_root / run_name
                model_dir.mkdir()
                best_values = {"joint": float("inf"), "classification": -float("inf"), "parameter": float("inf")}
                best_epochs = {objective: 0 for objective in best_values}
                stale, history = 0, []
                for epoch in range(1, args.epochs + 1):
                    if profile["scheduler"] == "cosine":
                        lr = _set_warmup_cosine_lr(optimizer, profile["learning_rate"], epoch, args.epochs, args.warmup_epochs)
                    elif epoch <= args.warmup_epochs:
                        lr = profile["learning_rate"] * epoch / args.warmup_epochs
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    else:
                        lr = optimizer.param_groups[0]["lr"]
                    train_loss = _train_epoch(
                        model, loaders["train"], device, normalizer, args.parameter_weight,
                        optimizer, scaler, target_weights,
                    )
                    early_outputs = _collect_outputs(
                        model, loaders["early"], device, normalizer,
                        args.parameter_weight, target_weights,
                    )
                    early_metrics = _metrics(early_outputs, 0.5, normalizer)
                    if plateau is not None and epoch >= args.warmup_epochs:
                        plateau.step(early_metrics["total_loss"])
                    history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "early": _compact(early_metrics)})
                    print(json.dumps({"model": model_name, "profile": profile_name, "seed": seed, **history[-1]}), flush=True)
                    objective_values = {
                        "joint": float(early_metrics["total_loss"]),
                        "classification": float(early_metrics["macro_aupr"]),
                        "parameter": float(early_metrics["parameter"]["oracle_macro_target_nmae"]),
                    }
                    improved = {
                        "joint": objective_values["joint"] < best_values["joint"],
                        "classification": objective_values["classification"] > best_values["classification"],
                        "parameter": objective_values["parameter"] < best_values["parameter"],
                    }
                    for objective, is_improved in improved.items():
                        if is_improved:
                            best_values[objective] = objective_values[objective]
                            best_epochs[objective] = epoch
                            checkpoint_payload = {
                                "model_state": model.state_dict(), "epoch": epoch,
                                "early": early_metrics, "profile": profile,
                                "checkpoint_objective": objective,
                            }
                            torch.save(checkpoint_payload, model_dir / f"best_{objective}.pt")
                            if objective == "joint":
                                torch.save(checkpoint_payload, model_dir / "best.pt")
                    if improved["joint"]:
                        stale = 0
                    else:
                        stale += 1
                        if stale >= args.patience:
                            break
                (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                for checkpoint_objective in args.checkpoint_objectives:
                    checkpoint_path = model_dir / f"best_{checkpoint_objective}.pt"
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint["model_state"])
                    calibration_outputs = _collect_outputs(
                        model, loaders["calibration"], device, normalizer,
                        args.parameter_weight, target_weights,
                    )
                    thresholds = optimize_class_thresholds(
                        torch.sigmoid(calibration_outputs["logits"]).numpy(), calibration_outputs["labels"].numpy()
                    )
                    tune_outputs = _collect_outputs(
                        model, loaders["tune"], device, normalizer,
                        args.parameter_weight, target_weights,
                    )
                    confirm_outputs = _collect_outputs(
                        model, loaders["confirm"], device, normalizer,
                        args.parameter_weight, target_weights,
                    )
                    tune_fixed, tune_adaptive = _metrics(tune_outputs, 0.5, normalizer), _metrics(tune_outputs, thresholds, normalizer)
                    confirm_fixed, confirm_adaptive = _metrics(confirm_outputs, 0.5, normalizer), _metrics(confirm_outputs, thresholds, normalizer)
                    receipt = {
                        "model": model_name, "profile": profile_name, "seed": seed,
                        "profile_config": profile, "best_epoch": best_epochs[checkpoint_objective],
                        "checkpoint_objective": checkpoint_objective,
                        "early_selection_value": best_values[checkpoint_objective],
                        "thresholds": dict(zip(CLASS_NAMES, thresholds.tolist())),
                        "tune_fixed": tune_fixed, "tune_adaptive": tune_adaptive,
                        "confirm_fixed": confirm_fixed, "confirm_adaptive": confirm_adaptive,
                    }
                    receipt_path = model_dir / f"receipt_{checkpoint_objective}.json"
                    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    if checkpoint_objective == "joint":
                        (model_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    row = {
                        "model": model_name, "profile": profile_name, "seed": seed,
                        "checkpoint_objective": checkpoint_objective,
                        "scheduler": profile["scheduler"], "learning_rate": profile["learning_rate"],
                        "weight_decay": profile["weight_decay"], "best_epoch": best_epochs[checkpoint_objective],
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "thresholds": json.dumps(thresholds.tolist()),
                        **{f"tune_fixed_{key}": value for key, value in _compact(tune_fixed).items()},
                        **{f"tune_adaptive_{key}": value for key, value in _compact(tune_adaptive).items()},
                        **{f"confirm_fixed_{key}": value for key, value in _compact(confirm_fixed).items()},
                        **{f"confirm_adaptive_{key}": value for key, value in _compact(confirm_adaptive).items()},
                        "checkpoint": str(checkpoint_path),
                    }
                    rows.append(row)
                    _write_csv(run_root / "summary.csv", rows)
                    (run_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                del model, optimizer, plateau, scaler, loaders
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    for dataset in datasets.values():
        dataset.close()
    print(f"COMPLETE {run_root}", flush=True)


if __name__ == "__main__":
    main()
