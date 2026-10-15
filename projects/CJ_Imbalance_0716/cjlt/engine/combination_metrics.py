"""Combination, cardinality, and combination-power diagnostics.

These metrics deliberately operate on already-frozen predictions.  They do not
select thresholds and therefore are safe to use on early-stop/development
roles without introducing a second optimization path.
"""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn import metrics as sklearn_metrics


def _validate(y_true, y_pred):
    true = np.asarray(y_true, dtype=np.uint8)
    pred = np.asarray(y_pred, dtype=np.uint8)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("y_true and y_pred must be matching [N, C] arrays")
    if not np.isin(true, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("combination metrics require binary arrays")
    return true, pred


def combination_key(row: np.ndarray, class_names: Sequence[str]) -> str:
    active = [str(class_names[index]) for index in np.flatnonzero(row)]
    return "+".join(active) if active else "<empty>"


def _most_common_error(mask, values, class_names):
    counts = values[mask].sum(axis=0)
    if not mask.any() or int(counts.max()) == 0:
        return None
    return str(class_names[int(np.argmax(counts))])


def compute_combination_metrics(
    y_true,
    y_pred,
    class_names: Optional[Sequence[str]] = None,
    train_combo_counts: Optional[Mapping[str, int]] = None,
    combo_tiers: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    true, pred = _validate(y_true, y_pred)
    names = (
        [str(index) for index in range(true.shape[1])]
        if class_names is None
        else [str(value) for value in class_names]
    )
    if len(names) != true.shape[1]:
        raise ValueError("class_names length must match label columns")

    keys = np.asarray([combination_key(row, names) for row in true], dtype=object)
    rows = []
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        group_true = true[mask]
        group_pred = pred[mask]
        exact = np.all(group_true == group_pred, axis=1)
        missed = np.logical_and(group_true == 1, group_pred == 0)
        false_positive = np.logical_and(group_true == 0, group_pred == 1)
        rows.append(
            {
                "combination": key,
                "cardinality": int(group_true[0].sum()),
                "evaluation_count": int(mask.sum()),
                "train_count": (
                    None
                    if train_combo_counts is None
                    else int(train_combo_counts.get(key, 0))
                ),
                "tier": (
                    None if combo_tiers is None else combo_tiers.get(key, "unseen")
                ),
                "seen_in_training": (
                    None
                    if train_combo_counts is None
                    else int(train_combo_counts.get(key, 0)) > 0
                ),
                "exact_match_accuracy": float(exact.mean()),
                "sample_f1": float(
                    sklearn_metrics.f1_score(
                        group_true, group_pred, average="samples", zero_division=0
                    )
                ),
                "hamming_loss": float(
                    sklearn_metrics.hamming_loss(group_true, group_pred)
                ),
                "most_frequently_missed_label": _most_common_error(
                    np.ones(len(group_true), dtype=bool), missed, names
                ),
                "most_frequent_false_positive_label": _most_common_error(
                    np.ones(len(group_true), dtype=bool), false_positive, names
                ),
            }
        )

    exact_values = np.asarray(
        [row["exact_match_accuracy"] for row in rows], dtype=np.float64
    )
    summary = {
        "combination_count": int(len(rows)),
        "macro_combination_accuracy": float(exact_values.mean()),
        "worst_5_combination_accuracy": float(
            np.sort(exact_values)[: min(5, len(exact_values))].mean()
        ),
    }
    if combo_tiers is not None:
        for tier in ("head", "middle", "tail"):
            selected = [
                row["exact_match_accuracy"] for row in rows if row["tier"] == tier
            ]
            summary["{}_combination_accuracy".format(tier)] = (
                float(np.mean(selected)) if selected else float("nan")
            )
    if train_combo_counts is not None:
        for name, seen in (("seen", True), ("unseen", False)):
            selected = [
                row["exact_match_accuracy"]
                for row in rows
                if row["seen_in_training"] is seen
            ]
            summary["{}_combination_accuracy".format(name)] = (
                float(np.mean(selected)) if selected else float("nan")
            )
    return {"summary": summary, "per_combination": rows}


def compute_cardinality_metrics(y_true, y_pred) -> Dict[str, object]:
    true, pred = _validate(y_true, y_pred)
    rows = []
    cardinality = true.sum(axis=1)
    for value in sorted(np.unique(cardinality).tolist()):
        mask = cardinality == value
        group_true = true[mask]
        group_pred = pred[mask]
        rows.append(
            {
                "cardinality": int(value),
                "count": int(mask.sum()),
                "subset_accuracy": float(
                    sklearn_metrics.accuracy_score(group_true, group_pred)
                ),
                "sample_f1": float(
                    sklearn_metrics.f1_score(
                        group_true, group_pred, average="samples", zero_division=0
                    )
                ),
                "hamming_loss": float(
                    sklearn_metrics.hamming_loss(group_true, group_pred)
                ),
                "cardinality_mae": float(
                    np.abs(
                        group_true.sum(axis=1).astype(np.int64)
                        - group_pred.sum(axis=1).astype(np.int64)
                    ).mean()
                ),
            }
        )
    return {"per_cardinality": rows}


def compute_combination_power_metrics(
    y_true,
    y_pred,
    power_bin,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    true, pred = _validate(y_true, y_pred)
    bins = np.asarray(power_bin)
    if bins.shape != (len(true),):
        raise ValueError("power_bin must have shape [N]")
    names = (
        [str(index) for index in range(true.shape[1])]
        if class_names is None
        else [str(value) for value in class_names]
    )
    rows = []
    combo = np.asarray([combination_key(row, names) for row in true], dtype=object)
    for key in sorted(set(combo.tolist())):
        for level in sorted(set(bins[combo == key].tolist())):
            mask = np.logical_and(combo == key, bins == level)
            rows.append(
                {
                    "combination": key,
                    "power_bin": float(level),
                    "count": int(mask.sum()),
                    "exact_match_accuracy": float(
                        np.all(true[mask] == pred[mask], axis=1).mean()
                    ),
                }
            )
    return {"per_combination_power": rows}
