"""Four-class composition-disjoint validation on paired 25/30/35 dB I/Q views.

This module intentionally reads only ``closedset_6class_train.mat`` from the
power-sweep folders.  The four-class task keeps FMB, SawFM, MFT and ISRJ and
removes every identity containing SinFM or VFT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cj0724.data import SIGNAL_LENGTH, complex_rms_normalize, read_labels
from cj0724.models import build_benchmark_model
from cj0724.train import get_device, set_seed


CLASS_NAMES = ("FMB", "SawFM", "MFT", "ISRJ")
CLASS_COLUMNS = np.asarray([0, 1, 3, 5], dtype=np.int64)
EXCLUDED_COLUMNS = np.asarray([2, 4], dtype=np.int64)
POWER_LEVELS = (25, 30, 35)
DEFAULT_POWER_ROOT = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
POWER_DIRECTORIES = {
    25: "level_04__jnr_25dB",
    30: "level_05__jnr_30dB",
    35: "level_06__jnr_35dB",
}

# Each fold contains five completely unseen combinations.  Across three folds
# every one of the 15 non-empty four-class combinations is evaluated once.
UNSEEN_COMBINATION_FOLDS = {
    0: (1, 6, 10, 13, 15),
    1: (2, 5, 7, 9, 14),
    2: (3, 4, 8, 11, 12),
}


def combination_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    return labels.dot(1 << np.arange(labels.shape[1], dtype=np.int64)).astype(np.int64)


def combination_name(combination_id: int) -> str:
    active = [name for index, name in enumerate(CLASS_NAMES) if combination_id & (1 << index)]
    return "+".join(active) if active else "none"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_receiver_noise_seed(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"][:], dtype=np.float64).reshape(-1)


def load_and_audit_labels(paths: dict[int, Path]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    reference_labels = read_labels(paths[30])
    reference_seed = read_receiver_noise_seed(paths[30])
    per_power: dict[str, object] = {}
    for power, path in paths.items():
        labels = read_labels(path)
        noise_seed = read_receiver_noise_seed(path)
        if not np.array_equal(labels, reference_labels):
            raise RuntimeError(f"label_selected is not aligned between 30 dB and {power} dB")
        if not np.array_equal(noise_seed, reference_seed):
            raise RuntimeError(f"receiver_noise_seed is not aligned between 30 dB and {power} dB")
        per_power[str(power)] = {
            "path": str(path),
            "samples": int(len(labels)),
            "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "receiver_noise_seed_sha256": hashlib.sha256(noise_seed.tobytes()).hexdigest(),
        }

    keep = (
        (reference_labels[:, EXCLUDED_COLUMNS].sum(axis=1) == 0)
        & (reference_labels[:, CLASS_COLUMNS].sum(axis=1) > 0)
    )
    source_indices = np.flatnonzero(keep).astype(np.int64)
    labels4 = reference_labels[source_indices][:, CLASS_COLUMNS].astype(np.uint8, copy=True)
    combo_ids = combination_ids(labels4)
    counts = {str(int(combo)): int((combo_ids == combo).sum()) for combo in np.unique(combo_ids)}
    if set(map(int, counts)) != set(range(1, 16)) or set(counts.values()) != {400}:
        raise RuntimeError(f"Expected 15 combinations x 400 identities, got {counts}")
    audit = {
        "class_names": CLASS_NAMES,
        "class_columns_in_six_class_data": CLASS_COLUMNS.tolist(),
        "excluded_classes": ("SinFM", "VFT"),
        "excluded_columns": EXCLUDED_COLUMNS.tolist(),
        "four_class_identities": int(len(source_indices)),
        "combination_counts": counts,
        "power_alignment": per_power,
        "paired_identity_rule": "same source index and receiver_noise_seed stay in one role across all powers",
        "locked_test_used": False,
    }
    return source_indices, labels4, audit


def make_composition_disjoint_roles(
    source_indices: np.ndarray,
    labels: np.ndarray,
    *,
    fold: int,
    seed: int,
    train_per_combo: int = 280,
    early_per_combo: int = 40,
    calibration_per_combo: int = 40,
    seen_confirm_per_combo: int = 40,
    unseen_per_combo: int = 400,
) -> dict[str, np.ndarray]:
    if fold not in UNSEEN_COMBINATION_FOLDS:
        raise ValueError(f"fold must be one of {tuple(UNSEEN_COMBINATION_FOLDS)}")
    total_seen = train_per_combo + early_per_combo + calibration_per_combo + seen_confirm_per_combo
    if total_seen > 400 or unseen_per_combo > 400:
        raise ValueError("Requested more than 400 identities from a combination")
    combo_ids = combination_ids(labels)
    unseen_combos = set(UNSEEN_COMBINATION_FOLDS[fold])
    rng = np.random.default_rng(int(seed))
    roles: dict[str, list[np.ndarray]] = {
        "train": [], "early": [], "calibration": [], "seen_confirm": [], "unseen": []
    }
    for combo in range(1, 16):
        group = source_indices[combo_ids == combo]
        shuffled = rng.permutation(group)
        if combo in unseen_combos:
            roles["unseen"].append(np.sort(shuffled[:unseen_per_combo]))
            continue
        cursor = 0
        for role, count in (
            ("train", train_per_combo),
            ("early", early_per_combo),
            ("calibration", calibration_per_combo),
            ("seen_confirm", seen_confirm_per_combo),
        ):
            roles[role].append(np.sort(shuffled[cursor: cursor + count]))
            cursor += count
    output = {
        role: np.sort(np.concatenate(parts)).astype(np.int64) if parts else np.empty(0, dtype=np.int64)
        for role, parts in roles.items()
    }
    for left_index, left in enumerate(output):
        for right in list(output)[left_index + 1:]:
            if np.intersect1d(output[left], output[right]).size:
                raise RuntimeError(f"identity leakage between {left} and {right}")
    return output


class PairedPowerIdentityDataset(Dataset):
    """Yield all three power views for one source identity in a single item."""

    def __init__(self, paths: dict[int, Path], indices: Iterable[int]) -> None:
        self.paths = tuple(paths[power] for power in POWER_LEVELS)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        labels6 = read_labels(self.paths[1])
        self.labels = labels6[:, CLASS_COLUMNS].astype(np.float32, copy=False)
        self._handles: list[h5py.File] | None = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def _get_handles(self) -> list[h5py.File]:
        if self._handles is None:
            self._handles = [h5py.File(path, "r") for path in self.paths]
        return self._handles

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        source_index = int(self.indices[int(position)])
        views = []
        for handle in self._get_handles():
            raw = np.asarray(handle["data_legacy"][:, source_index], dtype=np.float32)
            views.append(complex_rms_normalize(raw.reshape(2, SIGNAL_LENGTH)))
        return {
            "iq": torch.from_numpy(np.stack(views, axis=0)),
            "label": torch.from_numpy(self.labels[source_index].copy()),
            "source_index": source_index,
        }

    def close(self) -> None:
        if self._handles is not None:
            for handle in self._handles:
                handle.close()
            self._handles = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = None
        return state

    def __del__(self) -> None:
        self.close()


class FourClassMLCNN(nn.Module):
    """MLCNN with an unchanged common classifier and train-only prototype parameters."""

    def __init__(self, method: str, head_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        if method not in {"baseline", "prototype_relation"}:
            raise ValueError("method must be baseline or prototype_relation")
        base = build_benchmark_model(
            "legacy_mlcnn", num_classes=4, head_dim=head_dim, dropout=dropout
        )
        self.method = method
        self.backbone = base.backbone
        self.feature_adapter = base.feature_adapter
        self.classifier = base.classifier
        if method == "prototype_relation":
            self.positive_prototypes = nn.Parameter(torch.randn(4, head_dim) / math.sqrt(head_dim))
            self.negative_prototypes = nn.Parameter(torch.randn(4, head_dim) / math.sqrt(head_dim))

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.backbone(iq)
        embedding = self.feature_adapter(raw)
        return self.classifier(embedding), embedding


@dataclass(frozen=True)
class LossWeights:
    prototype: float = 0.20
    relation: float = 0.05
    hard_negative: float = 0.20
    separation: float = 0.05
    power_consistency: float = 0.05


def train_loss(
    model: FourClassMLCNN,
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_identities: int,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = F.binary_cross_entropy_with_logits(logits, labels)
    components = {"bce": float(base.detach())}
    if model.method == "baseline":
        return base, components

    normalized = F.normalize(embeddings, dim=1)
    positive = F.normalize(model.positive_prototypes, dim=1)
    negative = F.normalize(model.negative_prototypes, dim=1)
    prototype_logits = 8.0 * (normalized @ positive.T - normalized @ negative.T)
    prototype = F.binary_cross_entropy_with_logits(prototype_logits, labels)

    views = len(POWER_LEVELS)
    identity_embedding = F.normalize(embeddings.reshape(batch_identities, views, -1).mean(dim=1), dim=1)
    identity_labels = labels.reshape(batch_identities, views, -1)[:, 0]
    intersection = identity_labels @ identity_labels.T
    cardinality = identity_labels.sum(dim=1, keepdim=True)
    union = cardinality + cardinality.T - intersection
    target_relation = intersection / union.clamp_min(1.0)
    predicted_relation = (identity_embedding @ identity_embedding.T + 1.0) * 0.5
    off_diagonal = ~torch.eye(batch_identities, dtype=torch.bool, device=logits.device)
    relation = F.mse_loss(predicted_relation[off_diagonal], target_relation[off_diagonal])

    hard_terms = []
    for class_index in range(labels.shape[1]):
        negative_mask = labels[:, class_index] < 0.5
        positive_count = int((~negative_mask).sum().item())
        negative_logits = logits[negative_mask, class_index]
        if positive_count and negative_logits.numel():
            count = min(positive_count, int(negative_logits.numel()))
            selected = torch.topk(negative_logits, k=count, largest=True).values
            hard_terms.append(F.softplus(selected).mean())
    hard_negative = torch.stack(hard_terms).mean() if hard_terms else logits.sum() * 0.0

    separation = F.relu((positive * negative).sum(dim=1) - 0.0).mean()
    view_embeddings = embeddings.reshape(batch_identities, views, -1)
    embedding_center = view_embeddings.mean(dim=1, keepdim=True)
    view_probabilities = torch.sigmoid(logits).reshape(batch_identities, views, -1)
    probability_center = view_probabilities.mean(dim=1, keepdim=True)
    power_consistency = (
        F.mse_loss(view_embeddings, embedding_center.expand_as(view_embeddings))
        + F.mse_loss(view_probabilities, probability_center.expand_as(view_probabilities))
    )

    total = (
        base
        + weights.prototype * prototype
        + weights.relation * relation
        + weights.hard_negative * hard_negative
        + weights.separation * separation
        + weights.power_consistency * power_consistency
    )
    components.update(
        prototype=float(prototype.detach()),
        relation=float(relation.detach()),
        hard_negative=float(hard_negative.detach()),
        separation=float(separation.detach()),
        power_consistency=float(power_consistency.detach()),
    )
    return total, components


def metric_summary(probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    predicted = probabilities >= thresholds[None, :]
    tp = (predicted & (labels == 1)).sum(axis=0).astype(np.float64)
    fp = (predicted & (labels == 0)).sum(axis=0).astype(np.float64)
    fn = ((~predicted) & (labels == 1)).sum(axis=0).astype(np.float64)
    per_f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1.0)
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    per_aupr = [average_precision_score(labels[:, index], probabilities[:, index]) for index in range(4)]
    return {
        "subset_accuracy": float((predicted == labels).all(axis=1).mean()),
        "macro_f1": float(per_f1.mean()),
        "micro_f1": float(2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1.0)),
        "macro_aupr": float(np.mean(per_aupr)),
        "hamming_loss": float((predicted != labels).mean()),
    }


def optimize_thresholds(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.10, 0.90, 33)
    thresholds = np.full(4, 0.5, dtype=np.float64)
    best = metric_summary(probabilities, labels, thresholds)
    for _ in range(3):
        changed = False
        for class_index in range(4):
            current = thresholds[class_index]
            current_key = (best["subset_accuracy"], best["macro_f1"], -best["hamming_loss"])
            for candidate in grid:
                trial = thresholds.copy()
                trial[class_index] = candidate
                metrics = metric_summary(probabilities, labels, trial)
                key = (metrics["subset_accuracy"], metrics["macro_f1"], -metrics["hamming_loss"])
                if key > current_key:
                    current, current_key, best = float(candidate), key, metrics
                    changed = True
            thresholds[class_index] = current
        if not changed:
            break
    return thresholds


@torch.no_grad()
def predict(model: FourClassMLCNN, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    all_probability, all_label, all_index = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        batch_size, views = iq.shape[:2]
        logits, _ = model(iq.reshape(batch_size * views, 2, SIGNAL_LENGTH))
        probability = torch.sigmoid(logits).reshape(batch_size, views, 4)
        all_probability.append(probability.cpu().numpy())
        all_label.append(batch["label"].numpy())
        all_index.append(np.asarray(batch["source_index"], dtype=np.int64))
    return {
        "view_probabilities": np.concatenate(all_probability),
        "probabilities": np.concatenate(all_probability).mean(axis=1),
        "labels": np.concatenate(all_label).astype(np.uint8),
        "source_indices": np.concatenate(all_index),
    }


def per_class_rows(method: str, role: str, probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
    predicted = probabilities >= thresholds[None, :]
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        target = labels[:, index] == 1
        estimate = predicted[:, index]
        tp = int((estimate & target).sum())
        fp = int((estimate & ~target).sum())
        fn = int((~estimate & target).sum())
        tn = int((~estimate & ~target).sum())
        rows.append({
            "method": method, "role": role, "class": name,
            "aupr": float(average_precision_score(target, probabilities[:, index])),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "fpr": float(fp / max(fp + tn, 1)),
            "threshold": float(thresholds[index]),
            "positive_support": int(target.sum()),
        })
    return rows


def per_combination_rows(method: str, role: str, probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
    predicted = probabilities >= thresholds[None, :]
    ids = combination_ids(labels)
    rows = []
    for combo in sorted(np.unique(ids)):
        mask = ids == combo
        rows.append({
            "method": method, "role": role, "combination_id": int(combo),
            "combination": combination_name(int(combo)),
            "cardinality": int(bin(int(combo)).count("1")),
            "support": int(mask.sum()),
            "exact_accuracy": float((predicted[mask] == labels[mask]).all(axis=1).mean()),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    *,
    method: str,
    seed: int,
    paths: dict[int, Path],
    roles: dict[str, np.ndarray],
    run_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    head_dim: int,
    dropout: float,
    weights: LossWeights,
) -> dict[str, object]:
    set_seed(seed)
    datasets = {role: PairedPowerIdentityDataset(paths, indices) for role, indices in roles.items()}
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        role: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=role == "train",
            generator=generator if role == "train" else None,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        for role, dataset in datasets.items()
    }
    model = FourClassMLCNN(method, head_dim=head_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None
    history = []
    best_key = None
    stale = 0
    method_dir = run_dir / f"{method}__seed{seed}"
    method_dir.mkdir(parents=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses, components = [], []
        for batch in loaders["train"]:
            iq = batch["iq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            identities, views = iq.shape[:2]
            expanded_labels = labels[:, None, :].expand(identities, views, 4).reshape(-1, 4)
            autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
            with autocast:
                logits, embeddings = model(iq.reshape(identities * views, 2, SIGNAL_LENGTH))
                loss, parts = train_loss(model, logits, embeddings, expanded_labels, identities, weights)
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
            losses.append(float(loss.detach()))
            components.append(parts)
        scheduler.step()
        early = predict(model, loaders["early"], device)
        early_metrics = metric_summary(early["probabilities"], early["labels"], np.full(4, 0.5))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "early": early_metrics,
            "loss_components": {
                key: float(np.mean([part[key] for part in components if key in part]))
                for key in components[0]
            },
        }
        history.append(row)
        print(json.dumps({"method": method, "seed": seed, **row}, ensure_ascii=False))
        key = (early_metrics["subset_accuracy"], early_metrics["macro_f1"], early_metrics["macro_aupr"])
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            torch.save({
                "method": method, "seed": seed, "epoch": epoch,
                "model_state": model.state_dict(), "early_metrics": early_metrics,
            }, method_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    (method_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(method_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    predictions = {role: predict(model, loaders[role], device) for role in ("calibration", "seen_confirm", "unseen")}
    thresholds = optimize_thresholds(predictions["calibration"]["probabilities"], predictions["calibration"]["labels"])
    np.save(method_dir / "thresholds.npy", thresholds)
    metrics: dict[str, object] = {}
    class_rows, combo_rows, power_rows = [], [], []
    for role, result in predictions.items():
        summary = metric_summary(result["probabilities"], result["labels"], thresholds)
        summary["mean_probability_std_across_power"] = float(result["view_probabilities"].std(axis=1).mean())
        metrics[role] = summary
        class_rows.extend(per_class_rows(method, role, result["probabilities"], result["labels"], thresholds))
        combo_rows.extend(per_combination_rows(method, role, result["probabilities"], result["labels"], thresholds))
        for view_index, power in enumerate(POWER_LEVELS):
            power_rows.append({
                "method": method, "role": role, "power_dB": power,
                **metric_summary(result["view_probabilities"][:, view_index], result["labels"], thresholds),
            })
        np.savez_compressed(
            method_dir / f"{role}_predictions.npz",
            probabilities=result["probabilities"],
            view_probabilities=result["view_probabilities"],
            labels=result["labels"],
            source_indices=result["source_indices"],
            thresholds=thresholds,
        )
    metrics["seen_unseen_subset_gap"] = float(
        metrics["seen_confirm"]["subset_accuracy"] - metrics["unseen"]["subset_accuracy"]
    )
    metrics["best_epoch"] = int(checkpoint["epoch"])
    metrics["parameters"] = int(sum(parameter.numel() for parameter in model.parameters()))
    (method_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(method_dir / "per_class.csv", class_rows)
    write_csv(method_dir / "per_combination.csv", combo_rows)
    write_csv(method_dir / "per_power.csv", power_rows)
    for dataset in datasets.values():
        dataset.close()
    del model, optimizer, scheduler, scaler, loaders, datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"method": method, "seed": seed, **metrics}


def aggregate_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, object] = {"method": method, "seeds": [row["seed"] for row in selected]}
        for role in ("seen_confirm", "unseen"):
            for metric in ("subset_accuracy", "macro_f1", "macro_aupr", "hamming_loss"):
                values = np.asarray([row[role][metric] for row in selected], dtype=np.float64)
                item[f"{role}_{metric}_mean"] = float(values.mean())
                item[f"{role}_{metric}_std"] = float(values.std(ddof=0))
        values = np.asarray([row["seen_unseen_subset_gap"] for row in selected], dtype=np.float64)
        item["seen_unseen_subset_gap_mean"] = float(values.mean())
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/four_class_composition_disjoint"))
    parser.add_argument("--methods", nargs="+", choices=("baseline", "prototype_relation"), default=("baseline", "prototype_relation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(2041, 2042, 2043))
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16, help="Identity batch size; actual view batch is 3x larger")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-per-combo", type=int, default=280)
    parser.add_argument("--early-per-combo", type=int, default=40)
    parser.add_argument("--calibration-per-combo", type=int, default=40)
    parser.add_argument("--seen-confirm-per-combo", type=int, default=40)
    parser.add_argument("--unseen-per-combo", type=int, default=400)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive")
    paths = {
        power: args.power_root / POWER_DIRECTORIES[power] / "closedset_6class_train.mat"
        for power in POWER_LEVELS
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_indices, labels, data_audit = load_and_audit_labels(paths)
    device = get_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"fold{args.fold}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    weights = LossWeights()
    protocol = {
        **vars(args),
        "power_root": str(args.power_root), "output_root": str(args.output_root),
        "device_resolved": str(device), "class_names": CLASS_NAMES,
        "class_definition": "keep FMB,SawFM,MFT,ISRJ; exclude every identity containing SinFM or VFT",
        "power_levels_dB": POWER_LEVELS,
        "data_files": {str(power): str(path) for power, path in paths.items()},
        "unseen_combination_ids": UNSEEN_COMBINATION_FOLDS[args.fold],
        "unseen_combinations": [combination_name(value) for value in UNSEEN_COMBINATION_FOLDS[args.fold]],
        "selection": "seen early-stop subset accuracy, then macro-F1, then macro-AUPR",
        "threshold_calibration": "seen calibration identities only; coordinate search for subset accuracy",
        "primary_endpoint": "identity-aggregated unseen-combination subset accuracy",
        "auxiliary_modules": "training only; inference remains MLCNN embedding plus common linear classifier",
        "loss_weights": weights.__dict__,
        "locked_test_used": False,
        "locked_test_authorized": False,
    }
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str), encoding="utf-8")
    (run_dir / "data_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")
    all_rows = []
    for seed in args.seeds:
        roles = make_composition_disjoint_roles(
            source_indices, labels, fold=args.fold, seed=seed,
            train_per_combo=args.train_per_combo, early_per_combo=args.early_per_combo,
            calibration_per_combo=args.calibration_per_combo,
            seen_confirm_per_combo=args.seen_confirm_per_combo,
            unseen_per_combo=args.unseen_per_combo,
        )
        np.savez_compressed(run_dir / f"roles_seed{seed}.npz", **roles)
        role_receipt = {}
        reference_labels = read_labels(paths[30])[:, CLASS_COLUMNS]
        for role, indices in roles.items():
            ids = combination_ids(reference_labels[indices])
            role_receipt[role] = {
                "identities": int(len(indices)), "views": int(len(indices) * 3),
                "combination_ids": sorted(map(int, np.unique(ids))),
                "combinations": [combination_name(int(value)) for value in sorted(np.unique(ids))],
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            }
        (run_dir / f"role_receipt_seed{seed}.json").write_text(json.dumps(role_receipt, indent=2), encoding="utf-8")
        for method in args.methods:
            all_rows.append(train_one(
                method=method, seed=seed, paths=paths, roles=roles, run_dir=run_dir,
                device=device, epochs=args.epochs, batch_size=args.batch_size, workers=args.workers,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay, patience=args.patience,
                head_dim=args.head_dim, dropout=args.dropout, weights=weights,
            ))
            (run_dir / "runs.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    aggregate = aggregate_runs(all_rows)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    flat_rows = []
    for item in aggregate:
        flat_rows.append({key: value for key, value in item.items() if key != "seeds"} | {"seeds": ",".join(map(str, item["seeds"]))})
    write_csv(run_dir / "aggregate.csv", flat_rows)
    receipt = {
        "run_dir": str(run_dir), "protocol_sha256": sha256_file(run_dir / "protocol.json"),
        "data_audit_sha256": sha256_file(run_dir / "data_audit.json"),
        "locked_test_used": False,
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Four-class composition-disjoint run complete: {run_dir}")


if __name__ == "__main__":
    main()
