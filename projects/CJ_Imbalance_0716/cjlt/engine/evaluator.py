"""Multilabel metrics with tail-label and weak-component diagnostics."""

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn import metrics as sklearn_metrics

from cjlt.engine.combination_metrics import (
    compute_cardinality_metrics,
    compute_combination_metrics,
)


def _validate_pair(y_true: np.ndarray, y_pred: np.ndarray):
    true = np.asarray(y_true, dtype=np.uint8)
    pred = np.asarray(y_pred, dtype=np.uint8)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.ndim != 2:
        raise ValueError("multilabel arrays must be two-dimensional")
    return true, pred


def _count_distribution(values: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {int(value): int(count) for value, count in zip(unique, counts)}


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    tail_indices: Optional[Sequence[int]] = None,
    weak_component_mask: Optional[np.ndarray] = None,
    train_combo_counts: Optional[Dict[str, int]] = None,
    combo_tiers: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    true, pred = _validate_pair(y_true, y_pred)
    if class_names is None:
        names = [str(index) for index in range(true.shape[1])]
    else:
        if len(class_names) != true.shape[1]:
            raise ValueError("class_names length must match label columns")
        names = [str(name) for name in class_names]

    per_precision, per_recall, per_f1, per_support = (
        sklearn_metrics.precision_recall_fscore_support(
            true, pred, average=None, zero_division=0
        )
    )
    result = {
        "subset_acc": float(sklearn_metrics.accuracy_score(true, pred)),
        "hamming_loss": float(sklearn_metrics.hamming_loss(true, pred)),
        "macro_precision": float(
            sklearn_metrics.precision_score(
                true, pred, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            sklearn_metrics.recall_score(
                true, pred, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            sklearn_metrics.f1_score(true, pred, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            sklearn_metrics.precision_score(
                true, pred, average="micro", zero_division=0
            )
        ),
        "micro_recall": float(
            sklearn_metrics.recall_score(
                true, pred, average="micro", zero_division=0
            )
        ),
        "micro_f1": float(
            sklearn_metrics.f1_score(true, pred, average="micro", zero_division=0)
        ),
        "label_acc": float((true == pred).mean()),
        "per_class_acc": [float(value) for value in (true == pred).mean(axis=0)],
        "true_label_count_distribution": _count_distribution(true.sum(axis=1)),
        "predicted_label_count_distribution": _count_distribution(pred.sum(axis=1)),
        "cardinality_mae": float(
            np.mean(
                np.abs(
                    true.sum(axis=1).astype(np.int64)
                    - pred.sum(axis=1).astype(np.int64)
                )
            )
        ),
    }

    per_aupr = [float("nan")] * true.shape[1]
    if probabilities is not None:
        probability = np.asarray(probabilities, dtype=np.float64)
        if probability.shape != true.shape:
            raise ValueError("probabilities must have the same shape as y_true")
        if not np.isfinite(probability).all():
            raise ValueError("probabilities contain NaN or Inf")
        for index in range(true.shape[1]):
            if 0 < int(true[:, index].sum()) < len(true):
                per_aupr[index] = float(
                    sklearn_metrics.average_precision_score(
                        true[:, index], probability[:, index]
                    )
                )
        finite_aupr = [value for value in per_aupr if np.isfinite(value)]
        result["macro_aupr"] = (
            float(np.mean(finite_aupr)) if finite_aupr else float("nan")
        )
        result["micro_aupr"] = float(
            sklearn_metrics.average_precision_score(
                true.reshape(-1), probability.reshape(-1)
            )
        )
    else:
        result["macro_aupr"] = float("nan")
        result["micro_aupr"] = float("nan")

    per_specificity = []
    per_balanced_accuracy = []
    for index in range(true.shape[1]):
        negative = true[:, index] == 0
        specificity = (
            float((pred[negative, index] == 0).mean()) if negative.any() else float("nan")
        )
        per_specificity.append(specificity)
        per_balanced_accuracy.append(
            float((per_recall[index] + specificity) / 2.0)
            if np.isfinite(specificity)
            else float("nan")
        )

    result["per_label"] = [
        {
            "label_index": int(index),
            "label": names[index],
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "support": int(per_support[index]),
            "accuracy": float((true[:, index] == pred[:, index]).mean()),
            "specificity": float(per_specificity[index]),
            "balanced_accuracy": float(per_balanced_accuracy[index]),
            "aupr": float(per_aupr[index]),
        }
        for index in range(true.shape[1])
    ]

    if tail_indices is None:
        result["tail_label_recall"] = float("nan")
        result["tail_positive_count"] = 0
    else:
        tail = np.asarray(list(tail_indices), dtype=np.int64)
        if tail.size and (tail.min() < 0 or tail.max() >= true.shape[1]):
            raise IndexError("tail_indices contain an out-of-range label")
        tail_true = true[:, tail]
        denominator = int(tail_true.sum())
        result["tail_positive_count"] = denominator
        result["tail_label_recall"] = (
            float(pred[:, tail][tail_true.astype(bool)].mean())
            if denominator
            else float("nan")
        )

    if weak_component_mask is None:
        result["weak_component_recall"] = float("nan")
        result["weak_component_count"] = 0
    else:
        weak = np.asarray(weak_component_mask, dtype=bool)
        if weak.shape != true.shape:
            raise ValueError("weak_component_mask must have the same shape as y_true")
        weak_positive = np.logical_and(weak, true.astype(bool))
        result["weak_component_count"] = int(weak_positive.sum())
        result["weak_component_recall"] = (
            float(pred[weak_positive].mean()) if weak_positive.any() else float("nan")
        )
    result["combination_metrics"] = compute_combination_metrics(
        true,
        pred,
        class_names=names,
        train_combo_counts=train_combo_counts,
        combo_tiers=combo_tiers,
    )
    result["cardinality_metrics"] = compute_cardinality_metrics(true, pred)
    return result
