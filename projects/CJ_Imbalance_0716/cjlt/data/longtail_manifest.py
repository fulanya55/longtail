"""Deterministic train-only manifests for label and combination long tails."""

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from cjlt.data.split_builder import combo_key, group_indices_by_combo


@dataclass
class LongTailManifest:
    protocol: str
    selected_indices: np.ndarray
    seed: int
    source_split_sha256: str
    combo_counts: Dict[str, int]
    sha256: str
    imbalance_factor: Optional[float] = None
    target_label_counts: Optional[np.ndarray] = None
    achieved_label_counts: Optional[np.ndarray] = None
    label_count_error: Optional[np.ndarray] = None
    target_sample_count: Optional[int] = None
    achieved_sample_count: Optional[int] = None
    sample_count_error: Optional[float] = None
    label_counts: Optional[np.ndarray] = None
    weak_label_counts: Optional[np.ndarray] = None
    label_tiers: Optional[Dict[str, str]] = None
    combo_tiers: Optional[Dict[str, str]] = None
    weak_threshold_db: Optional[float] = None
    sample_id_sha256: Optional[str] = None
    selection_note: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "protocol": self.protocol,
            "selected_indices": [int(value) for value in self.selected_indices],
            "seed": int(self.seed),
            "source_split_sha256": self.source_split_sha256,
            "combo_counts": dict(self.combo_counts),
            "sha256": self.sha256,
            "imbalance_factor": self.imbalance_factor,
            "target_label_counts": (
                None
                if self.target_label_counts is None
                else [int(value) for value in self.target_label_counts]
            ),
            "achieved_label_counts": (
                None
                if self.achieved_label_counts is None
                else [int(value) for value in self.achieved_label_counts]
            ),
            "label_count_error": (
                None
                if self.label_count_error is None
                else [float(value) for value in self.label_count_error]
            ),
            "target_sample_count": self.target_sample_count,
            "achieved_sample_count": self.achieved_sample_count,
            "sample_count_error": self.sample_count_error,
            "label_counts": (
                None
                if self.label_counts is None
                else [int(value) for value in self.label_counts]
            ),
            "weak_label_counts": (
                None
                if self.weak_label_counts is None
                else [int(value) for value in self.weak_label_counts]
            ),
            "label_tiers": self.label_tiers,
            "combo_tiers": self.combo_tiers,
            "weak_threshold_db": self.weak_threshold_db,
            "sample_id_sha256": self.sample_id_sha256,
            "selection_note": self.selection_note,
        }


def _validate_inputs(
    labels: np.ndarray, eligible_indices: Iterable[int]
) -> tuple:
    matrix = np.asarray(labels)
    if matrix.ndim != 2 or not np.isin(matrix, [0, 1]).all():
        raise ValueError("labels must be a binary [N, C] array")
    indices = np.asarray(list(eligible_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("eligible_indices must be a nonempty one-dimensional sequence")
    if indices.min() < 0 or indices.max() >= len(matrix):
        raise IndexError("eligible_indices contain an out-of-range sample index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("eligible_indices must not contain duplicates")
    return matrix.astype(np.uint8, copy=False), indices


def _combo_counts(labels: np.ndarray, indices: np.ndarray) -> Dict[str, int]:
    counts = {}
    for index in indices:
        key = combo_key(labels[int(index)])
        counts[key] = counts.get(key, 0) + 1
    return {key: int(counts[key]) for key in sorted(counts)}


def _manifest_sha(payload: Dict[str, object]) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def sample_id_sha256(indices: Iterable[int]) -> str:
    values = np.asarray(list(indices), dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _frequency_tiers(counts: Dict[str, int]) -> Dict[str, str]:
    ordered = sorted(counts, key=lambda key: (-int(counts[key]), str(key)))
    if not ordered:
        return {}
    boundaries = np.array_split(np.asarray(ordered, dtype=object), 3)
    result = {}
    for tier, values in zip(("head", "middle", "tail"), boundaries):
        for value in values.tolist():
            result[str(value)] = tier
    return result


def compute_weak_component_mask(
    labels: np.ndarray,
    component_power_db: np.ndarray,
    weak_threshold_db: float,
) -> np.ndarray:
    matrix = np.asarray(labels, dtype=np.uint8)
    power = np.asarray(component_power_db, dtype=np.float64)
    if matrix.shape != power.shape or matrix.ndim != 2:
        raise ValueError("labels and component_power_db must match [N, C]")
    threshold = abs(float(weak_threshold_db))
    weak = np.zeros_like(matrix, dtype=bool)
    for row_index in range(len(matrix)):
        positive = np.flatnonzero(matrix[row_index])
        if len(positive) < 2:
            continue
        positive_power = power[row_index, positive]
        if not np.isfinite(positive_power).all():
            raise ValueError("positive component powers must be finite")
        for position, label_index in enumerate(positive):
            other = np.delete(positive_power, position)
            relative = float(positive_power[position] - other.max())
            weak[row_index, label_index] = relative <= -threshold
    return weak


def profile_multi_axis_imbalance(
    labels: np.ndarray,
    eligible_indices: Iterable[int],
    component_power_db: Optional[np.ndarray] = None,
    weak_threshold_db: float = 10.0,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    matrix, indices = _validate_inputs(labels, eligible_indices)
    names = (
        [str(index) for index in range(matrix.shape[1])]
        if class_names is None
        else [str(value) for value in class_names]
    )
    if len(names) != matrix.shape[1]:
        raise ValueError("class_names length must match label columns")
    selected = matrix[indices]
    label_counts = selected.sum(axis=0).astype(np.int64)
    combo_counts = _combo_counts(matrix, indices)
    weak_counts = np.zeros(matrix.shape[1], dtype=np.int64)
    if component_power_db is not None:
        weak = compute_weak_component_mask(
            matrix, component_power_db, weak_threshold_db
        )
        weak_counts = weak[indices].sum(axis=0).astype(np.int64)
    label_count_map = {names[i]: int(label_counts[i]) for i in range(len(names))}
    return {
        "sample_count": int(len(indices)),
        "sample_ids": [int(value) for value in indices],
        "sample_id_sha256": sample_id_sha256(indices),
        "label_counts": label_count_map,
        "weak_label_counts": {
            names[i]: int(weak_counts[i]) for i in range(len(names))
        },
        "combination_counts": combo_counts,
        "label_tiers": _frequency_tiers(label_count_map),
        "combo_tiers": _frequency_tiers(combo_counts),
        "weak_threshold_db": float(abs(weak_threshold_db)),
    }


def build_power_longtail(
    labels: np.ndarray,
    eligible_indices: Iterable[int],
    component_power_db: np.ndarray,
    weak_threshold_db: float,
    seed: int,
    source_split_sha256: str = "",
) -> LongTailManifest:
    matrix, indices = _validate_inputs(labels, eligible_indices)
    weak = compute_weak_component_mask(
        matrix, component_power_db, weak_threshold_db
    )
    profile = profile_multi_axis_imbalance(
        matrix, indices, component_power_db, weak_threshold_db
    )
    payload = {
        "protocol": "power_lt",
        "selected_indices": indices.tolist(),
        "seed": int(seed),
        "source_split_sha256": str(source_split_sha256),
        "combo_counts": profile["combination_counts"],
        "weak_label_counts": weak[indices].sum(axis=0).astype(int).tolist(),
        "weak_threshold_db": float(abs(weak_threshold_db)),
        "sample_id_sha256": profile["sample_id_sha256"],
    }
    return LongTailManifest(
        protocol="power_lt",
        selected_indices=indices,
        seed=int(seed),
        source_split_sha256=str(source_split_sha256),
        combo_counts=profile["combination_counts"],
        sha256=_manifest_sha(payload),
        label_counts=matrix[indices].sum(axis=0).astype(np.int64),
        weak_label_counts=weak[indices].sum(axis=0).astype(np.int64),
        label_tiers=profile["label_tiers"],
        combo_tiers=profile["combo_tiers"],
        weak_threshold_db=float(abs(weak_threshold_db)),
        sample_id_sha256=profile["sample_id_sha256"],
        selection_note="Power-LT preserves identities and annotates weak components; power views are not independently split.",
    )


def build_mixed_longtail(
    labels: np.ndarray,
    eligible_indices: Iterable[int],
    component_power_db: np.ndarray,
    label_imbalance_factor: float,
    combo_imbalance_factor: float,
    weak_threshold_db: float,
    seed: int,
    source_split_sha256: str = "",
) -> LongTailManifest:
    """Deterministically induce label/combo tails and annotate power weakness.

    Combination thinning is applied first.  A second label-aware retention pass
    uses the maximum retention probability among a sample's positive labels so
    that a tail label is not discarded merely because it co-occurs with a head
    label.  Achieved counts are always recorded instead of claiming exact IF.
    """

    matrix, indices = _validate_inputs(labels, eligible_indices)
    combo = build_combo_longtail(
        matrix,
        indices,
        imbalance_factor=float(combo_imbalance_factor),
        seed=int(seed),
        source_split_sha256=source_split_sha256,
    )
    candidates = combo.selected_indices
    counts = matrix[candidates].sum(axis=0).astype(np.float64)
    order = np.argsort(-counts, kind="stable")
    denominator = max(len(order) - 1, 1)
    target = counts.copy()
    maximum = float(counts[order[0]])
    for rank, label_index in enumerate(order):
        desired = maximum * float(label_imbalance_factor) ** (
            -float(rank) / float(denominator)
        )
        target[label_index] = min(counts[label_index], max(1.0, desired))
    retention = np.divide(
        target,
        np.maximum(counts, 1.0),
        out=np.ones_like(target),
        where=counts > 0,
    )
    rng = np.random.RandomState(int(seed) + 7919)
    keep = np.zeros(len(candidates), dtype=bool)
    grouped = group_indices_by_combo(matrix, candidates)
    forced = {int(values[0]) for values in grouped.values() if len(values)}
    for position, sample_index in enumerate(candidates):
        positive = np.flatnonzero(matrix[int(sample_index)])
        probability = float(retention[positive].max()) if len(positive) else 1.0
        keep[position] = int(sample_index) in forced or rng.rand() <= probability
    selected = candidates[keep].copy()
    rng.shuffle(selected)
    weak = compute_weak_component_mask(matrix, component_power_db, weak_threshold_db)
    label_counts = matrix[selected].sum(axis=0).astype(np.int64)
    weak_counts = weak[selected].sum(axis=0).astype(np.int64)
    combo_counts = _combo_counts(matrix, selected)
    label_map = {str(index): int(value) for index, value in enumerate(label_counts)}
    payload = {
        "protocol": "mixed_lt",
        "selected_indices": selected.tolist(),
        "seed": int(seed),
        "source_split_sha256": str(source_split_sha256),
        "combo_counts": combo_counts,
        "label_counts": label_counts.tolist(),
        "weak_label_counts": weak_counts.tolist(),
        "label_imbalance_factor": float(label_imbalance_factor),
        "combo_imbalance_factor": float(combo_imbalance_factor),
        "weak_threshold_db": float(abs(weak_threshold_db)),
    }
    return LongTailManifest(
        protocol="mixed_lt",
        selected_indices=selected,
        seed=int(seed),
        source_split_sha256=str(source_split_sha256),
        combo_counts=combo_counts,
        sha256=_manifest_sha(payload),
        imbalance_factor=float(label_imbalance_factor),
        label_counts=label_counts,
        weak_label_counts=weak_counts,
        label_tiers=_frequency_tiers(label_map),
        combo_tiers=_frequency_tiers(combo_counts),
        weak_threshold_db=float(abs(weak_threshold_db)),
        sample_id_sha256=sample_id_sha256(selected),
        selection_note="Mixed-LT uses combo thinning plus tail-preserving label retention; achieved counts are authoritative.",
    )


def build_combo_longtail(
    labels: np.ndarray,
    eligible_indices: Iterable[int],
    imbalance_factor: float,
    seed: int,
    source_split_sha256: str = "",
    min_samples: int = 3,
) -> LongTailManifest:
    matrix, indices = _validate_inputs(labels, eligible_indices)
    imbalance = float(imbalance_factor)
    if imbalance < 1.0:
        raise ValueError("imbalance_factor must be at least 1")
    groups = group_indices_by_combo(matrix, indices)
    combo_names = sorted(groups)
    if len(combo_names) < 2:
        raise ValueError("combo long-tail construction requires at least two combinations")
    max_samples = min(len(groups[name]) for name in combo_names)
    rng = np.random.RandomState(int(seed))
    ranked_names = list(combo_names)
    rng.shuffle(ranked_names)

    selected = []
    selected_counts = {}
    denominator = float(len(ranked_names) - 1)
    for rank, name in enumerate(ranked_names):
        target = int(
            round(max_samples * imbalance ** (-float(rank) / denominator))
        )
        target = max(int(min_samples), min(target, len(groups[name])))
        candidates = groups[name].copy()
        rng.shuffle(candidates)
        chosen = candidates[:target]
        selected.extend(chosen.tolist())
        selected_counts[name] = int(len(chosen))

    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    combo_counts = {key: selected_counts[key] for key in sorted(selected_counts)}
    payload = {
        "protocol": "combo_lt",
        "selected_indices": selected_array.tolist(),
        "seed": int(seed),
        "source_split_sha256": str(source_split_sha256),
        "combo_counts": combo_counts,
        "imbalance_factor": imbalance,
    }
    return LongTailManifest(
        protocol="combo_lt",
        selected_indices=selected_array,
        seed=int(seed),
        source_split_sha256=str(source_split_sha256),
        combo_counts=combo_counts,
        imbalance_factor=imbalance,
        sha256=_manifest_sha(payload),
    )


def build_label_longtail(
    labels: np.ndarray,
    eligible_indices: Iterable[int],
    target_label_counts: Sequence[int],
    seed: int,
    tolerance: float = 0.05,
    source_split_sha256: str = "",
    target_sample_count: Optional[int] = None,
    sample_count_tolerance: float = 0.0,
) -> LongTailManifest:
    matrix, indices = _validate_inputs(labels, eligible_indices)
    target = np.asarray(target_label_counts, dtype=np.int64)
    if target.shape != (matrix.shape[1],):
        raise ValueError(
            "target_label_counts must have shape ({},), got {}".format(
                matrix.shape[1], target.shape
            )
        )
    if np.any(target <= 0):
        raise ValueError("target_label_counts must be positive")
    tolerance_value = float(tolerance)
    if tolerance_value < 0.0 or tolerance_value >= 1.0:
        raise ValueError("tolerance must be in [0, 1)")
    sample_tolerance_value = float(sample_count_tolerance)
    if sample_tolerance_value < 0.0 or sample_tolerance_value >= 1.0:
        raise ValueError("sample_count_tolerance must be in [0, 1)")
    requested_sample_count = None
    if target_sample_count is not None:
        requested_sample_count = int(target_sample_count)
        if requested_sample_count <= 0:
            raise ValueError("target_sample_count must be positive")
        if requested_sample_count > len(indices):
            raise ValueError("target_sample_count exceeds eligible sample count")

    candidate_labels = matrix[indices].astype(np.int64, copy=False)
    available = candidate_labels.sum(axis=0)
    if np.any(target > available):
        bad = np.flatnonzero(target > available).tolist()
        raise ValueError(
            "Requested target exceeds available positives for labels {}".format(bad)
        )

    grouped = group_indices_by_combo(matrix, indices)
    combo_names = sorted(grouped)
    combo_vectors = np.asarray(
        [[int(bit) for bit in name] for name in combo_names], dtype=np.float64
    )
    available_by_combo = np.asarray(
        [len(grouped[name]) for name in combo_names], dtype=np.float64
    )
    combo_count = len(combo_names)
    label_count = matrix.shape[1]
    constraint = np.zeros((label_count, combo_count + 2 * label_count))
    constraint[:, :combo_count] = combo_vectors.T
    constraint[:, combo_count : combo_count + label_count] = np.eye(label_count)
    constraint[:, combo_count + label_count :] = -np.eye(label_count)
    lower_constraint = target.astype(np.float64)
    upper_constraint = target.astype(np.float64)
    if requested_sample_count is not None:
        sample_constraint = np.zeros((1, combo_count + 2 * label_count))
        sample_constraint[0, :combo_count] = 1.0
        constraint = np.vstack([constraint, sample_constraint])
        sample_error = requested_sample_count * sample_tolerance_value
        lower_constraint = np.concatenate(
            [lower_constraint, [requested_sample_count - sample_error]]
        )
        upper_constraint = np.concatenate(
            [upper_constraint, [requested_sample_count + sample_error]]
        )
    allowed_error = target.astype(np.float64) * tolerance_value
    objective = np.concatenate(
        [
            np.full(combo_count, 1e-8),
            1.0 / target.astype(np.float64),
            1.0 / target.astype(np.float64),
        ]
    )
    solution = milp(
        objective,
        integrality=np.concatenate(
            [np.ones(combo_count), np.zeros(2 * label_count)]
        ),
        bounds=Bounds(
            np.zeros(combo_count + 2 * label_count),
            np.concatenate([available_by_combo, allowed_error, allowed_error]),
        ),
        constraints=LinearConstraint(
            constraint, lower_constraint, upper_constraint
        ),
        options={"time_limit": 30.0},
    )
    if not solution.success or solution.x is None:
        raise ValueError("Unable to solve label long-tail combination counts")
    selected_combo_counts = np.rint(solution.x[:combo_count]).astype(np.int64)
    rng = np.random.RandomState(int(seed))
    selected_parts = []
    for name, count in zip(combo_names, selected_combo_counts):
        candidates = grouped[name].copy()
        rng.shuffle(candidates)
        selected_parts.append(candidates[: int(count)])
    selected_array = np.concatenate(selected_parts).astype(np.int64)
    rng.shuffle(selected_array)
    achieved = matrix[selected_array].sum(axis=0).astype(np.int64)
    error = np.abs(achieved - target).astype(np.float64) / target
    if np.any(error > tolerance_value + 1e-12):
        raise ValueError(
            "Unable to meet target label counts within tolerance: target={}, "
            "achieved={}, relative_error={}".format(
                target.tolist(), achieved.tolist(), error.tolist()
            )
        )
    achieved_sample_count = int(len(selected_array))
    sample_error = None
    if requested_sample_count is not None:
        sample_error = abs(achieved_sample_count - requested_sample_count) / float(
            requested_sample_count
        )
        if sample_error > sample_tolerance_value + 1e-12:
            raise ValueError(
                "Unable to meet target sample count within tolerance: target={}, "
                "achieved={}, relative_error={}".format(
                    requested_sample_count, achieved_sample_count, sample_error
                )
            )
    combo_counts = _combo_counts(matrix, selected_array)
    payload = {
        "protocol": "label_lt",
        "selected_indices": selected_array.tolist(),
        "seed": int(seed),
        "source_split_sha256": str(source_split_sha256),
        "combo_counts": combo_counts,
        "target_label_counts": target.tolist(),
        "achieved_label_counts": achieved.tolist(),
        "label_count_error": error.tolist(),
        "target_sample_count": requested_sample_count,
        "achieved_sample_count": achieved_sample_count,
        "sample_count_error": sample_error,
    }
    return LongTailManifest(
        protocol="label_lt",
        selected_indices=selected_array,
        seed=int(seed),
        source_split_sha256=str(source_split_sha256),
        combo_counts=combo_counts,
        target_label_counts=target,
        achieved_label_counts=achieved,
        label_count_error=error,
        target_sample_count=requested_sample_count,
        achieved_sample_count=achieved_sample_count,
        sample_count_error=sample_error,
        sha256=_manifest_sha(payload),
    )
