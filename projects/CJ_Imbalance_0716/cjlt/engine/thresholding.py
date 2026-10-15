"""Threshold selection restricted to the threshold-selection partition."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sklearn import metrics as sklearn_metrics


@dataclass(frozen=True)
class ThresholdSelection:
    thresholds: List[float]
    subset_acc: float
    mode: str
    source_role: str


def _require_threshold_role(source_role: str) -> None:
    if str(source_role) != "threshold_selection":
        raise ValueError(
            "Threshold fitting requires source_role='threshold_selection', got {!r}".format(
                source_role
            )
        )


def _validate_inputs(labels: np.ndarray, probabilities: np.ndarray):
    truth = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(probabilities, dtype=np.float32)
    if truth.shape != scores.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if truth.ndim != 2:
        raise ValueError("threshold arrays must be two-dimensional")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities contain NaN or Inf")
    return truth, scores


def _candidate_grid(grid: Optional[Iterable[float]]) -> List[float]:
    source = np.arange(0.05, 0.951, 0.01) if grid is None else grid
    values = [round(float(value), 10) for value in source]
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("threshold grid values must be in (0, 1)")
    return values


def apply_thresholds(
    probabilities: np.ndarray, thresholds: Sequence[float]
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=np.float32)
    values = np.asarray(thresholds, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    if values.ndim == 0:
        values = np.repeat(values, scores.shape[1])
    if values.shape != (scores.shape[1],):
        raise ValueError("threshold count must match probability columns")
    return (scores >= values[None, :]).astype(np.uint8)


def select_global_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    source_role: str,
    grid: Optional[Iterable[float]] = None,
) -> ThresholdSelection:
    _require_threshold_role(source_role)
    truth, scores = _validate_inputs(labels, probabilities)
    best_threshold = None
    best_score = -1.0
    for candidate in _candidate_grid(grid):
        prediction = apply_thresholds(scores, [candidate] * scores.shape[1])
        subset = float(np.all(prediction == truth, axis=1).mean())
        if subset > best_score or (
            np.isclose(subset, best_score)
            and (
                best_threshold is None
                or abs(candidate - 0.5) < abs(best_threshold - 0.5)
            )
        ):
            best_threshold = candidate
            best_score = subset
    return ThresholdSelection(
        thresholds=[float(best_threshold)] * scores.shape[1],
        subset_acc=float(best_score),
        mode="global",
        source_role="threshold_selection",
    )


def select_per_label_f1_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    source_role: str,
    grid: Optional[Iterable[float]] = None,
) -> ThresholdSelection:
    _require_threshold_role(source_role)
    truth, scores = _validate_inputs(labels, probabilities)
    candidates = _candidate_grid(grid)
    per_label = []
    for label_index in range(scores.shape[1]):
        best_threshold = None
        best_f1 = -1.0
        for candidate in candidates:
            prediction = (scores[:, label_index] >= candidate).astype(np.uint8)
            f1 = float(
                sklearn_metrics.f1_score(
                    truth[:, label_index], prediction, zero_division=0
                )
            )
            if f1 > best_f1 or (
                np.isclose(f1, best_f1)
                and (
                    best_threshold is None
                    or abs(candidate - 0.5) < abs(best_threshold - 0.5)
                )
            ):
                best_f1 = f1
                best_threshold = candidate
        per_label.append(float(best_threshold))
    prediction = apply_thresholds(scores, per_label)
    return ThresholdSelection(
        thresholds=per_label,
        subset_acc=float(np.all(prediction == truth, axis=1).mean()),
        mode="per_label_f1",
        source_role="threshold_selection",
    )


def select_per_label_fbeta_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    beta: float,
    source_role: str,
    grid: Optional[Iterable[float]] = None,
) -> ThresholdSelection:
    """Select one recall-weighted F-beta threshold per label."""

    _require_threshold_role(source_role)
    truth, scores = _validate_inputs(labels, probabilities)
    beta_value = float(beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive")
    candidates = _candidate_grid(grid)
    per_label = []
    for label_index in range(scores.shape[1]):
        best_threshold = None
        best_score = -1.0
        for candidate in candidates:
            prediction = (scores[:, label_index] >= candidate).astype(np.uint8)
            score = float(
                sklearn_metrics.fbeta_score(
                    truth[:, label_index],
                    prediction,
                    beta=beta_value,
                    zero_division=0,
                )
            )
            if score > best_score or (
                np.isclose(score, best_score)
                and (
                    best_threshold is None
                    or abs(candidate - 0.5) < abs(best_threshold - 0.5)
                )
            ):
                best_score = score
                best_threshold = candidate
        per_label.append(float(best_threshold))
    prediction = apply_thresholds(scores, per_label)
    return ThresholdSelection(
        thresholds=per_label,
        subset_acc=float(np.all(prediction == truth, axis=1).mean()),
        mode="per_label_fbeta",
        source_role="threshold_selection",
    )


def select_per_label_youden_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    source_role: str,
    grid: Optional[Iterable[float]] = None,
) -> ThresholdSelection:
    _require_threshold_role(source_role)
    truth, scores = _validate_inputs(labels, probabilities)
    candidates = _candidate_grid(grid)
    per_label = []
    for label_index in range(scores.shape[1]):
        positives = truth[:, label_index] == 1
        negatives = ~positives
        best_threshold = None
        best_youden = -float("inf")
        for candidate in candidates:
            prediction = scores[:, label_index] >= candidate
            tpr = float(prediction[positives].mean()) if positives.any() else 0.0
            fpr = float(prediction[negatives].mean()) if negatives.any() else 0.0
            youden = tpr - fpr
            if youden > best_youden or (
                np.isclose(youden, best_youden)
                and (
                    best_threshold is None
                    or abs(candidate - 0.5) < abs(best_threshold - 0.5)
                )
            ):
                best_youden = youden
                best_threshold = candidate
        per_label.append(float(best_threshold))
    prediction = apply_thresholds(scores, per_label)
    return ThresholdSelection(
        thresholds=per_label,
        subset_acc=float(np.all(prediction == truth, axis=1).mean()),
        mode="per_label_youden",
        source_role="threshold_selection",
    )


def select_per_label_weak_recall_fpr_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weak_component_mask: np.ndarray,
    max_negative_fpr: float,
    source_role: str,
    grid: Optional[Iterable[float]] = None,
) -> ThresholdSelection:
    """Maximize weak-positive recall under a per-label negative-FPR cap."""

    _require_threshold_role(source_role)
    truth, scores = _validate_inputs(labels, probabilities)
    weak = np.asarray(weak_component_mask, dtype=bool)
    if weak.shape != truth.shape:
        raise ValueError("weak_component_mask must match labels")
    fpr_cap = float(max_negative_fpr)
    if fpr_cap < 0.0 or fpr_cap >= 1.0:
        raise ValueError("max_negative_fpr must be in [0, 1)")
    candidates = _candidate_grid(grid)
    per_label = []
    for label_index in range(scores.shape[1]):
        positives = truth[:, label_index] == 1
        negatives = ~positives
        weak_positives = positives & weak[:, label_index]
        recall_target = weak_positives if weak_positives.any() else positives
        best_key = None
        best_threshold = None
        for candidate in candidates:
            prediction = scores[:, label_index] >= candidate
            fpr = float(prediction[negatives].mean()) if negatives.any() else 0.0
            if fpr > fpr_cap + 1e-12:
                continue
            recall = (
                float(prediction[recall_target].mean())
                if recall_target.any()
                else 0.0
            )
            f1 = float(
                sklearn_metrics.f1_score(
                    truth[:, label_index], prediction, zero_division=0
                )
            )
            key = (recall, f1, -abs(candidate - 0.5))
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = candidate
        if best_threshold is None:
            realized = []
            for candidate in candidates:
                prediction = scores[:, label_index] >= candidate
                fpr = float(prediction[negatives].mean()) if negatives.any() else 0.0
                realized.append((fpr, -candidate, candidate))
            best_threshold = min(realized)[2]
        per_label.append(float(best_threshold))
    prediction = apply_thresholds(scores, per_label)
    return ThresholdSelection(
        thresholds=per_label,
        subset_acc=float(np.all(prediction == truth, axis=1).mean()),
        mode="per_label_weak_recall_fpr",
        source_role="threshold_selection",
    )


def select_shrinkage_per_label_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    positive_counts: Sequence[int],
    beta: float,
    source_role: str,
    grid: Optional[Iterable[float]] = None,
) -> ThresholdSelection:
    _require_threshold_role(source_role)
    truth, scores = _validate_inputs(labels, probabilities)
    counts = np.asarray(positive_counts, dtype=np.float64)
    if counts.shape != (scores.shape[1],) or np.any(counts < 0):
        raise ValueError("positive_counts must contain one nonnegative value per label")
    beta_value = float(beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive")
    candidates = _candidate_grid(grid)
    global_result = select_global_threshold(
        truth,
        scores,
        source_role="threshold_selection",
        grid=candidates,
    )
    per_label = select_per_label_f1_thresholds(
        truth,
        scores,
        source_role="threshold_selection",
        grid=candidates,
    ).thresholds
    alpha = counts / (counts + beta_value)
    global_threshold = float(global_result.thresholds[0])
    shrunk = alpha * np.asarray(per_label) + (1.0 - alpha) * global_threshold
    prediction = apply_thresholds(scores, shrunk)
    subset = float(np.all(prediction == truth, axis=1).mean())
    return ThresholdSelection(
        thresholds=[float(value) for value in shrunk],
        subset_acc=subset,
        mode="shrinkage_per_label",
        source_role="threshold_selection",
    )
