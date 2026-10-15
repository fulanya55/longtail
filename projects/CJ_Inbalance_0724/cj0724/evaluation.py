"""Detailed per-label and per-combination evaluation for backbone benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES
from cj0724.data import V73IQDataset
from cj0724.models import build_benchmark_model


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_wide_table(
    path: Path,
    rows: list[dict[str, object]],
    key_fields: list[str],
    value_field: str,
    model_order: list[str],
) -> None:
    indexed: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        output = indexed.setdefault(key, {field: row[field] for field in key_fields})
        output[str(row["model"])] = row[value_field]
    fields = key_fields + model_order
    _write_rows(path, list(indexed.values()), fields)


def predict_validation(
    model: torch.nn.Module,
    dataset: V73IQDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
    )
    logits_parts, label_parts, index_parts = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            logits_parts.append(model(iq).float().cpu())
            label_parts.append(batch["label"].float().cpu())
            indices = batch["sample_index"]
            index_parts.append(indices.cpu() if torch.is_tensor(indices) else torch.as_tensor(indices))
    return (
        torch.cat(logits_parts).numpy(),
        torch.cat(label_parts).numpy().astype(np.uint8),
        torch.cat(index_parts).numpy().astype(np.int64),
    )


def detailed_metrics(
    model_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probabilities >= float(threshold)).astype(np.uint8)
    exact_correct = np.all(predictions == labels, axis=1)
    label_rows: list[dict[str, object]] = []

    for label_index, label_name in enumerate(CLASS_NAMES):
        target = labels[:, label_index].astype(bool)
        predicted = predictions[:, label_index].astype(bool)
        tp = int(np.sum(target & predicted))
        tn = int(np.sum(~target & ~predicted))
        fp = int(np.sum(~target & predicted))
        fn = int(np.sum(target & ~predicted))
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        label_rows.append(
            {
                "model": model_name,
                "label_index": label_index,
                "label": label_name,
                "support_positive": int(np.sum(target)),
                "support_negative": int(np.sum(~target)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": _safe_ratio(tp + tn, len(target)),
                "precision": precision,
                "recall": recall,
                "specificity": _safe_ratio(tn, tn + fp),
                "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
            }
        )

    combination_rows: list[dict[str, object]] = []
    combination_ids = labels.dot(1 << np.arange(labels.shape[1], dtype=np.int64))
    for combination_id in sorted(np.unique(combination_ids).tolist()):
        mask = combination_ids == combination_id
        bits = labels[np.flatnonzero(mask)[0]]
        bit_string = "".join(str(int(value)) for value in bits)
        active_labels = "+".join(name for name, active in zip(CLASS_NAMES, bits) if active)
        combination_rows.append(
            {
                "model": model_name,
                "combination_id": int(combination_id),
                "combination_bits": bit_string,
                "active_labels": active_labels,
                "cardinality": int(bits.sum()),
                "support": int(mask.sum()),
                "exact_accuracy": float(exact_correct[mask].mean()),
                "mean_label_accuracy": float((predictions[mask] == labels[mask]).mean()),
                "mean_predicted_cardinality": float(predictions[mask].sum(axis=1).mean()),
            }
        )

    label_accuracies = [float(row["accuracy"]) for row in label_rows]
    label_f1s = [float(row["f1"]) for row in label_rows]
    combination_accuracies = [float(row["exact_accuracy"]) for row in combination_rows]
    summary = {
        "model": model_name,
        "overall_subset_accuracy": float(exact_correct.mean()),
        "macro_label_accuracy": float(np.mean(label_accuracies)),
        "macro_label_f1": float(np.mean(label_f1s)),
        "macro_combination_exact_accuracy": float(np.mean(combination_accuracies)),
        "worst_combination_exact_accuracy": float(np.min(combination_accuracies)),
        "best_combination_exact_accuracy": float(np.max(combination_accuracies)),
        "evaluated_combinations": len(combination_rows),
    }
    return label_rows, combination_rows, summary


def evaluate_benchmark_run(
    run_dir: str | Path,
    device_name: str = "auto",
    models: list[str] | None = None,
    workers: int = 0,
) -> Path:
    run_path = Path(run_dir).resolve()
    protocol = json.loads((run_path / "protocol.json").read_text(encoding="utf-8"))
    split = np.load(run_path / "split_indices.npz")
    val_indices = split["val_indices"]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    summary_path = run_path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Benchmark summary is not available yet: {summary_path}")
    benchmark_rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if models is not None:
        requested = set(models)
        benchmark_rows = [row for row in benchmark_rows if row["model"] in requested]
    model_order = [str(row["model"]) for row in benchmark_rows]
    dataset = V73IQDataset(protocol["data_file"], val_indices)
    all_label_rows: list[dict[str, object]] = []
    all_combination_rows: list[dict[str, object]] = []
    detailed_summaries: list[dict[str, object]] = []

    for benchmark_row in benchmark_rows:
        model_name = str(benchmark_row["model"])
        model = build_benchmark_model(
            model_name,
            num_classes=len(CLASS_NAMES),
            width=int(protocol["width"]),
            dropout=float(protocol["dropout"]),
            head_dim=int(protocol["head_dim"]),
            legacy_attention=str(protocol["legacy_attention"]),
        )
        checkpoint_path = Path(str(benchmark_row["checkpoint"]))
        if not checkpoint_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            checkpoint_path = project_root / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        logits, labels, sample_indices = predict_validation(
            model,
            dataset,
            device,
            batch_size=int(protocol["batch_size"]),
            workers=workers,
        )
        label_rows, combination_rows, summary = detailed_metrics(
            model_name,
            logits,
            labels,
            float(protocol["threshold"]),
        )
        all_label_rows.extend(label_rows)
        all_combination_rows.extend(combination_rows)
        detailed_summaries.append(summary)
        prediction_dir = run_path / model_name
        np.savez_compressed(
            prediction_dir / "validation_predictions.npz",
            sample_indices=sample_indices,
            logits=logits,
            labels=labels,
        )
        del checkpoint, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dataset.close()
    label_fields = [
        "model", "label_index", "label", "support_positive", "support_negative",
        "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity", "f1",
    ]
    combination_fields = [
        "model", "combination_id", "combination_bits", "active_labels", "cardinality",
        "support", "exact_accuracy", "mean_label_accuracy", "mean_predicted_cardinality",
    ]
    _write_rows(run_path / "per_label_metrics.csv", all_label_rows, label_fields)
    _write_rows(run_path / "per_combination_metrics.csv", all_combination_rows, combination_fields)
    _write_rows(
        run_path / "detailed_summary.csv",
        detailed_summaries,
        list(detailed_summaries[0].keys()) if detailed_summaries else ["model"],
    )
    _write_wide_table(
        run_path / "per_label_accuracy_comparison.csv",
        all_label_rows,
        ["label_index", "label"],
        "accuracy",
        model_order,
    )
    _write_wide_table(
        run_path / "per_label_f1_comparison.csv",
        all_label_rows,
        ["label_index", "label"],
        "f1",
        model_order,
    )
    _write_wide_table(
        run_path / "per_combination_accuracy_comparison.csv",
        all_combination_rows,
        ["combination_id", "combination_bits", "active_labels", "cardinality", "support"],
        "exact_accuracy",
        model_order,
    )
    return run_path / "detailed_summary.csv"
