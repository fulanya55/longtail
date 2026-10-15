"""Frozen P5f execution utilities: single base forward and union-positive lazy head."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.p5b1_jnr_physical_oracle import FeatureRole
from cj0724.p5b3_bounded_jnr_attachment import TARGET_CLASSES


def collect_shared_base_outputs(dataset, model, normalizer, device: torch.device):
    """Collect logits and all parameters from one frozen multi-task forward per batch."""
    loader = DataLoader(dataset, batch_size=126, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    logits, parameters = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            raw_logits, normalized = model(batch["iq"].to(device))
            logits.append(raw_logits.float().cpu().numpy())
            parameters.append(normalizer.inverse_transform(normalized.float().cpu()).numpy())
    return np.concatenate(logits), np.concatenate(parameters)


def union_positive_rows(decision: np.ndarray) -> np.ndarray:
    decision = np.asarray(decision, dtype=bool)
    if decision.ndim != 2 or decision.shape[1] < max(TARGET_CLASSES) + 1:
        raise ValueError("decision must be [N, 6] or wider")
    return decision[:, TARGET_CLASSES].any(axis=1)


def lazy_estimate(role: FeatureRole, model: torch.nn.Module, state: dict, device: torch.device, active_rows: np.ndarray) -> np.ndarray:
    """Run the frozen attachment head only for union-positive sample rows."""
    active_rows = np.asarray(active_rows, dtype=bool)
    if active_rows.shape != (len(role.feature),):
        raise ValueError("active_rows must align with role rows")
    output = np.full((len(role.feature), len(TARGET_CLASSES)), np.nan, dtype=np.float32)
    selected = np.flatnonzero(active_rows)
    if not selected.size:
        return output
    inputs = np.concatenate([role.feature[selected], role.composition[selected]], axis=1).astype(np.float32)
    inputs = (inputs - state["feature_mean"]) / state["feature_scale"]
    with torch.inference_mode():
        value = model(torch.from_numpy(inputs).to(device)).float().cpu().numpy()
    output[selected] = value * state["target_scale"] + state["target_mean"]
    return output


def attach_target_values(base_full: np.ndarray, values: np.ndarray, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scatter values only into pre-existing target-positive cells, returning the allowed mask."""
    base_full = np.asarray(base_full)
    values = np.asarray(values)
    decision = np.asarray(decision, dtype=bool)
    if base_full.ndim != 3 or base_full.shape[1:] != (6, 2):
        raise ValueError("base_full must have shape [N, 6, 2]")
    if values.shape != (base_full.shape[0], len(TARGET_CLASSES)) or decision.shape != base_full.shape[:2]:
        raise ValueError("values/decision shapes are incompatible with base_full")
    candidate = base_full.copy()
    allowed = np.zeros_like(base_full, dtype=bool)
    for local, class_index in enumerate(TARGET_CLASSES):
        selected = decision[:, class_index]
        if selected.any() and not np.isfinite(values[selected, local]).all():
            raise ValueError("lazy head did not produce a finite target value for a selected cell")
        candidate[selected, class_index, 0] = values[selected, local]
        allowed[:, class_index, 0] = selected
    return candidate, allowed
