"""Train EfficientNet-B1 on one frequency--visibility experimental cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader

from cj0724.dual_axis import (
    CLASS_NAMES,
    POWER_LEVELS,
    PairedPowerViewDataset,
    PowerViewDataset,
    audit_power_alignment,
    build_balanced_manifest,
    build_identity_roles,
    build_target_label_lt_manifest,
    combo_bits,
    discover_power_train_files,
    read_identity_metadata,
    roles_receipt,
)
from cj0724.losses import (
    AsymmetricLoss,
    DistributionBalancedLoss,
    PositiveReferenceDistillationLoss,
    SpecializedVFTResidualLoss,
    TargetHardNegativeLoss,
)
from cj0724.models import build_benchmark_model


CELLS = ("D00", "D10", "D01", "D11")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_index(name: str) -> int:
    normalized = str(name).lower()
    for index, class_name in enumerate(CLASS_NAMES):
        if class_name.lower() == normalized:
            return index
    raise ValueError(f"Unknown target label: {name}")


def weak_levels(label_index: int) -> tuple[int, int, int]:
    return (40, 45, 50) if int(label_index) < 3 else (10, 15, 20)


def metrics_from_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    weak_mask: np.ndarray | None = None,
) -> dict:
    true = np.asarray(labels, dtype=np.uint8)
    probability = np.asarray(probabilities, dtype=np.float64)
    predicted = probability >= np.asarray(thresholds, dtype=np.float64)[None, :]
    per_aupr = []
    for index in range(true.shape[1]):
        per_aupr.append(
            float(sk_metrics.average_precision_score(true[:, index], probability[:, index]))
        )
    result = {
        "macro_aupr": float(np.mean(per_aupr)),
        "micro_aupr": float(sk_metrics.average_precision_score(true.reshape(-1), probability.reshape(-1))),
        "macro_f1": float(sk_metrics.f1_score(true, predicted, average="macro", zero_division=0)),
        "micro_f1": float(sk_metrics.f1_score(true, predicted, average="micro", zero_division=0)),
        "hamming_loss": float(sk_metrics.hamming_loss(true, predicted)),
        "subset_accuracy": float(sk_metrics.accuracy_score(true, predicted)),
        "label_accuracy": float((true == predicted).mean()),
        "cardinality_mae": float(
            np.abs(
                true.sum(axis=1).astype(np.int64)
                - predicted.sum(axis=1).astype(np.int64)
            ).mean()
        ),
        "per_label_aupr": per_aupr,
    }
    negative = np.logical_not(true.astype(bool))
    false_positive = np.logical_and(predicted, negative)
    negative_count = negative.sum(axis=0)
    result["per_label_fpr"] = np.divide(
        false_positive.sum(axis=0),
        negative_count,
        out=np.full(true.shape[1], np.nan, dtype=np.float64),
        where=negative_count > 0,
    ).tolist()
    if weak_mask is None:
        result["weak_component_count"] = 0
        result["weak_component_recall"] = float("nan")
    else:
        weak_positive = np.logical_and(np.asarray(weak_mask, dtype=bool), true.astype(bool))
        result["weak_component_count"] = int(weak_positive.sum())
        result["weak_component_recall"] = (
            float(predicted[weak_positive].mean()) if weak_positive.any() else float("nan")
        )
    result["predicted"] = predicted.astype(np.uint8)
    return result


def select_f1_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 91)
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float64)
    for label_index in range(labels.shape[1]):
        best = (-1.0, 0.5)
        for threshold in grid:
            score = sk_metrics.f1_score(
                labels[:, label_index],
                probabilities[:, label_index] >= threshold,
                zero_division=0,
            )
            candidate = (float(score), -abs(float(threshold) - 0.5))
            if candidate > (best[0], -abs(best[1] - 0.5)):
                best = (float(score), float(threshold))
        thresholds[label_index] = best[1]
    return thresholds


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_parts = []
    label_parts = []
    weak_parts = []
    index_parts = []
    level_parts = []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        logits_parts.append(model(iq).float().cpu().numpy())
        label_parts.append(batch["label"].numpy())
        weak_parts.append(batch["weak_mask"].numpy())
        index_parts.append(np.asarray(batch["sample_index"], dtype=np.int64))
        level_parts.append(np.asarray(batch["level"], dtype=np.int16))
    logits = np.concatenate(logits_parts)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return (
        np.concatenate(label_parts).astype(np.uint8),
        probabilities,
        np.concatenate(weak_parts).astype(bool),
        np.concatenate(index_parts),
        np.concatenate(level_parts),
    )


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    view_mode: str,
    expert_aux_weight: float,
    reference_model: nn.Module | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        reference_logits = None
        if isinstance(criterion, PositiveReferenceDistillationLoss):
            teacher = reference_model if reference_model is not None else model
            teacher.eval()
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                reference_logits = teacher(
                    batch["iq_reference"].to(device, non_blocking=True)
                )
            if teacher is model:
                model.train()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            def supervised_loss(iq: torch.Tensor) -> torch.Tensor:
                if hasattr(model, "forward_components") and isinstance(
                    criterion, SpecializedVFTResidualLoss
                ):
                    return criterion(model.forward_components(iq), labels)
                if hasattr(model, "forward_experts"):
                    expert_logits = model.forward_experts(iq)
                    ensemble_loss = criterion(expert_logits.mean(dim=1), labels)
                    if expert_aux_weight <= 0.0:
                        return ensemble_loss
                    auxiliary = torch.stack(
                        [criterion(expert_logits[:, index], labels) for index in range(expert_logits.shape[1])]
                    ).mean()
                    return ensemble_loss + float(expert_aux_weight) * auxiliary
                return criterion(model(iq), labels)

            if view_mode == "pair":
                if isinstance(criterion, PositiveReferenceDistillationLoss):
                    weak_logits = model(batch["iq_weak"].to(device, non_blocking=True))
                    loss = criterion(weak_logits, reference_logits, labels)
                else:
                    loss = 0.5 * supervised_loss(
                        batch["iq_reference"].to(device, non_blocking=True)
                    ) + 0.5 * supervised_loss(batch["iq_weak"].to(device, non_blocking=True))
            else:
                loss = supervised_loss(batch["iq"].to(device, non_blocking=True))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def make_loader(
    dataset: PowerViewDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def evaluate_level(
    model: nn.Module,
    files: dict[int, Path],
    labels: np.ndarray,
    indices: np.ndarray,
    level: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = PowerViewDataset(files, labels, indices, (int(level),), seed)
    loader = make_loader(dataset, batch_size, False, seed)
    true, probability, weak, sample_indices, _ = predict(model, loader, device)
    dataset.close()
    return true, probability, weak, sample_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--power-root",
        type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1"),
    )
    parser.add_argument("--target-label", choices=CLASS_NAMES, required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument(
        "--model",
        choices=(
            "efficientnet_b1_1d",
            "efficientnet_b1_moe3_1d",
            "efficientnet_b1_vft_residual_moe_1d",
        ),
        default="efficientnet_b1_1d",
    )
    parser.add_argument(
        "--loss",
        choices=(
            "bce",
            "db",
            "asl",
            "ohem",
            "ohem_rank",
            "vft_residual_moe",
            "positive_reference_distill",
        ),
        default="bce",
    )
    parser.add_argument("--view-mode", choices=("single", "pair"), default="single")
    parser.add_argument("--method-name")
    parser.add_argument("--expert-aux-weight", type=float, default=0.3)
    parser.add_argument("--ohem-weight", type=float, default=0.5)
    parser.add_argument("--ohem-negative-ratio", type=float, default=2.0)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--distillation-weight", type=float, default=0.2)
    parser.add_argument("--distillation-max-logit-delta", type=float, default=2.0)
    parser.add_argument("--reference-teacher-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dual_axis_b1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    label_index = target_index(args.target_label)
    default_method = args.loss if args.view_mode == "single" else f"{args.loss}_pair"
    method = args.method_name or default_method
    run_dir = (
        args.output_root
        / args.target_label
        / args.cell
        / method
        / f"seed_{args.seed}"
    )
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    files = discover_power_train_files(args.power_root)
    audit = audit_power_alignment(files)
    _, labels = read_identity_metadata(files[30])
    roles = build_identity_roles(labels, args.seed)
    split_receipt = roles_receipt(roles, labels, args.seed)
    frequency_lt = args.cell in ("D10", "D11")
    visibility_weak = args.cell in ("D01", "D11")
    if frequency_lt:
        manifest = build_target_label_lt_manifest(
            labels, roles["optimization"], label_index, args.seed
        )
    else:
        manifest = build_balanced_manifest(labels, roles["optimization"], args.seed)
    train_indices = np.asarray(manifest["selected_indices"], dtype=np.int64)
    train_levels = weak_levels(label_index) if visibility_weak else (30,)
    primary_level = (50 if label_index < 3 else 10) if visibility_weak else 30

    np.savez_compressed(
        run_dir / "split_indices.npz",
        **{role: indices for role, indices in roles.items()},
    )
    write_json(run_dir / "data_audit.json", audit)
    write_json(run_dir / "split_receipt.json", split_receipt)
    write_json(run_dir / "longtail_manifest.json", manifest)
    protocol = {
        "model": args.model,
        "target_label": args.target_label,
        "target_label_index": label_index,
        "cell": args.cell,
        "frequency_lt": frequency_lt,
        "visibility_weak": visibility_weak,
        "train_levels": list(train_levels),
        "primary_level": int(primary_level),
        "loss": args.loss,
        "view_mode": args.view_mode,
        "method": method,
        "expert_aux_weight": float(args.expert_aux_weight),
        "ohem_weight": float(args.ohem_weight),
        "ohem_negative_ratio": float(args.ohem_negative_ratio),
        "ranking_weight": float(args.ranking_weight),
        "ranking_margin": float(args.ranking_margin),
        "distillation_weight": float(args.distillation_weight),
        "distillation_max_logit_delta": float(args.distillation_max_logit_delta),
        "reference_teacher_checkpoint": (
            str(args.reference_teacher_checkpoint.resolve())
            if args.reference_teacher_checkpoint is not None
            else None
        ),
        "reference_teacher_sha256": (
            file_sha256(args.reference_teacher_checkpoint)
            if args.reference_teacher_checkpoint is not None
            else None
        ),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "train_samples": int(len(train_indices)),
        "manifest_sha256": manifest["manifest_sha256"],
        "split_sha256": split_receipt["split_sha256"],
        "locked_test_used": False,
    }
    write_json(run_dir / "protocol.json", protocol)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = build_benchmark_model(
        name=args.model, num_classes=6, dropout=0.2, head_dim=256
    ).to(device)
    reference_model: nn.Module | None = None
    if args.reference_teacher_checkpoint is not None:
        if args.loss != "positive_reference_distill" or args.view_mode != "pair":
            raise ValueError(
                "--reference-teacher-checkpoint is only valid for paired positive reference distillation"
            )
        reference_model = build_benchmark_model(
            name=args.model, num_classes=6, dropout=0.2, head_dim=256
        ).to(device)
        teacher_checkpoint = torch.load(
            args.reference_teacher_checkpoint, map_location=device, weights_only=False
        )
        reference_model.load_state_dict(teacher_checkpoint["model_state"])
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    positive_counts = labels[train_indices].sum(axis=0).astype(int)
    criterion: nn.Module
    if args.loss == "db":
        criterion = DistributionBalancedLoss(positive_counts, len(train_indices)).to(device)
    elif args.loss == "asl":
        criterion = AsymmetricLoss().to(device)
    elif args.loss in ("ohem", "ohem_rank"):
        criterion = TargetHardNegativeLoss(
            target_index=label_index,
            ohem_weight=args.ohem_weight,
            negative_to_positive_ratio=args.ohem_negative_ratio,
            ranking_weight=args.ranking_weight if args.loss == "ohem_rank" else 0.0,
            ranking_margin=args.ranking_margin,
        ).to(device)
    elif args.loss == "vft_residual_moe":
        criterion = SpecializedVFTResidualLoss(target_index=label_index).to(device)
    elif args.loss == "positive_reference_distill":
        if args.view_mode != "pair":
            raise ValueError("positive_reference_distill requires --view-mode pair")
        criterion = PositiveReferenceDistillationLoss(
            target_index=label_index,
            distillation_weight=args.distillation_weight,
            max_logit_delta=args.distillation_max_logit_delta,
        ).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    dataset_class = PairedPowerViewDataset if args.view_mode == "pair" else PowerViewDataset
    train_dataset = dataset_class(files, labels, train_indices, train_levels, args.seed)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    early_dataset = PowerViewDataset(
        files, labels, roles["early_stop"], (primary_level,), args.seed
    )
    early_loader = make_loader(early_dataset, args.batch_size, False, args.seed)

    best_key = (-float("inf"), -float("inf"), -float("inf"))
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.view_mode,
            args.expert_aux_weight,
            reference_model,
        )
        early_true, early_probability, early_weak, _, _ = predict(
            model, early_loader, device
        )
        early_metrics = metrics_from_probabilities(
            early_true, early_probability, np.full(6, 0.5), early_weak
        )
        tail_aupr = float(early_metrics["per_label_aupr"][label_index])
        key = (
            float(early_metrics["macro_aupr"]),
            tail_aupr,
            -float(train_loss),
        )
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "early_macro_aupr": float(early_metrics["macro_aupr"]),
            "early_tail_aupr": tail_aupr,
            "early_macro_f1_0p5": float(early_metrics["macro_f1"]),
            "early_hamming_0p5": float(early_metrics["hamming_loss"]),
            "early_weak_recall_0p5": float(early_metrics["weak_component_recall"]),
        }
        history.append(row)
        print(json.dumps({"target": args.target_label, "cell": args.cell, **row}))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "protocol": protocol,
                    "early_metrics": row,
                },
                run_dir / "best.pt",
            )

    train_dataset.close()
    early_dataset.close()
    write_json(run_dir / "history.json", history)
    append_csv(run_dir / "history.csv", history)

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    calibration_true, calibration_probability, _, _ = evaluate_level(
        model,
        files,
        labels,
        roles["calibration"],
        primary_level,
        args.seed,
        args.batch_size,
        device,
    )
    calibrated_thresholds = select_f1_thresholds(
        calibration_true, calibration_probability
    )
    write_json(
        run_dir / "thresholds.json",
        {
            "mode": "per_label_f1_calibration_role",
            "primary_level": int(primary_level),
            "thresholds": calibrated_thresholds.tolist(),
            "locked_test_used": False,
        },
    )

    power_rows = []
    label_rows = []
    combination_rows = []
    prediction_payload = {}
    primary_metrics = None
    for level in POWER_LEVELS:
        true, probability, weak, sample_indices = evaluate_level(
            model,
            files,
            labels,
            roles["development"],
            level,
            args.seed,
            args.batch_size,
            device,
        )
        fixed = metrics_from_probabilities(true, probability, np.full(6, 0.5), weak)
        calibrated = metrics_from_probabilities(
            true, probability, calibrated_thresholds, weak
        )
        row = {
            "level": level,
            "macro_aupr": calibrated["macro_aupr"],
            "target_aupr": calibrated["per_label_aupr"][label_index],
            "macro_f1_calibrated": calibrated["macro_f1"],
            "hamming_calibrated": calibrated["hamming_loss"],
            "subset_accuracy_calibrated": calibrated["subset_accuracy"],
            "weak_recall_calibrated": calibrated["weak_component_recall"],
            "target_fpr_calibrated": calibrated["per_label_fpr"][label_index],
            "weak_count": calibrated["weak_component_count"],
            "macro_f1_0p5": fixed["macro_f1"],
            "hamming_0p5": fixed["hamming_loss"],
            "weak_recall_0p5": fixed["weak_component_recall"],
        }
        power_rows.append(row)
        if level == primary_level:
            primary_metrics = row
        predicted = calibrated.pop("predicted")
        fixed.pop("predicted")
        for index, name in enumerate(CLASS_NAMES):
            label_rows.append(
                {
                    "level": level,
                    "label_index": index,
                    "label": name,
                    "support": int(true[:, index].sum()),
                    "aupr": float(calibrated["per_label_aupr"][index]),
                    "accuracy": float((predicted[:, index] == true[:, index]).mean()),
                    "precision": float(sk_metrics.precision_score(true[:, index], predicted[:, index], zero_division=0)),
                    "recall": float(sk_metrics.recall_score(true[:, index], predicted[:, index], zero_division=0)),
                    "f1": float(sk_metrics.f1_score(true[:, index], predicted[:, index], zero_division=0)),
                    "fpr": float(calibrated["per_label_fpr"][index]),
                }
            )
        combo_keys = np.asarray([combo_bits(label) for label in true], dtype=object)
        for key in sorted(set(combo_keys.tolist())):
            mask = combo_keys == key
            combination_rows.append(
                {
                    "level": level,
                    "combination_bits": key,
                    "active_labels": "+".join(
                        CLASS_NAMES[index] for index, bit in enumerate(key) if bit == "1"
                    ),
                    "cardinality": key.count("1"),
                    "support": int(mask.sum()),
                    "exact_accuracy": float((predicted[mask] == true[mask]).all(axis=1).mean()),
                    "mean_label_accuracy": float((predicted[mask] == true[mask]).mean()),
                }
            )
        prediction_payload[f"labels_{level}"] = true
        prediction_payload[f"probabilities_{level}"] = probability.astype(np.float32)
        prediction_payload[f"weak_mask_{level}"] = weak
        prediction_payload[f"sample_indices_{level}"] = sample_indices

    if primary_metrics is None:
        raise AssertionError("Primary level was not evaluated")
    append_csv(run_dir / "power_metrics.csv", power_rows)
    append_csv(run_dir / "per_label_power_metrics.csv", label_rows)
    append_csv(run_dir / "per_combination_metrics.csv", combination_rows)
    np.savez_compressed(run_dir / "development_predictions.npz", **prediction_payload)
    summary = {
        **protocol,
        "best_epoch": int(best_epoch),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "primary_macro_aupr": float(primary_metrics["macro_aupr"]),
        "primary_target_aupr": float(primary_metrics["target_aupr"]),
        "primary_macro_f1_calibrated": float(primary_metrics["macro_f1_calibrated"]),
        "primary_hamming_calibrated": float(primary_metrics["hamming_calibrated"]),
        "primary_weak_recall_calibrated": float(primary_metrics["weak_recall_calibrated"]),
        "primary_target_fpr_calibrated": float(primary_metrics["target_fpr_calibrated"]),
        "thresholds": calibrated_thresholds.tolist(),
        "status": "complete",
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
