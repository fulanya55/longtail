"""Validate Route-B R1 baseline code paths on a fixed development split.

This is a preflight/smoke stage, not a performance result.  It creates no
test-v2 role and does not select a model or threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset, read_labels
from cj0724.multitask import (
    PARAMETER_COLUMNS,
    masked_parameter_smooth_l1,
    build_multitask_model,
    reshape_selected_parameters,
)
from cj0724.models import build_benchmark_model
from cj0724.multitask_benchmark import fit_normalizer, make_combination_stratified_split


PROJECT = Path(__file__).resolve().parents[1]
DOCS = PROJECT / "docs"
REPORTS = DOCS / "reports"
RESULTS = DOCS / "results"
RECEIPTS = DOCS / "receipts"
CONFIGS = PROJECT / "configs"


VARIANTS = {
    "B0": {"kind": "classification_only", "model": "legacy_mlcnn"},
    "B1": {"kind": "shared_backbone_shared_head", "model": "legacy_mlcnn_global"},
    "B2": {"kind": "shared_backbone_class_conditioned_head", "model": "legacy_mlcnn_class_conditioned"},
    "B3": {"kind": "class_conditioned_parameter_stop_gradient", "model": "legacy_mlcnn_class_conditioned_detached"},
    "B4": {"kind": "class_conditioned_parameter_loss_warmup", "model": "legacy_mlcnn_class_conditioned"},
}


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main() -> None:
    for directory in (REPORTS, RESULTS, RECEIPTS, CONFIGS):
        directory.mkdir(parents=True, exist_ok=True)
    labels = read_labels(DEFAULT_TRAIN_MAT)
    train_indices, val_indices = make_combination_stratified_split(labels, 0.1, 20260826)
    config = {
        "stage": "R1_preflight",
        "status": "proposal/not-validated_for_performance",
        "data_file": str(DEFAULT_TRAIN_MAT),
        "class_names": list(CLASS_NAMES),
        "input_shape": [2, 3000],
        "development_role": "development-v2 candidate split from training MAT; no test-v2 created",
        "split_seed": 20260826,
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "train_indices_sha256": sha256_array(train_indices),
        "val_indices_sha256": sha256_array(val_indices),
        "optimizer": "AdamW",
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "parameter_loss": "masked SmoothL1 on label=1 AND parameter_mask=1 AND finite target",
        "parameter_weight": 0.1,
        "warmup_variant": "B4 uses the B2 architecture; schedule is not performance-validated in this preflight",
        "threshold": 0.5,
        "locked_test_used": False,
    }
    (CONFIGS / "route_b_r1_preregister.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    normalizer = fit_normalizer(DEFAULT_TRAIN_MAT, train_indices, labels)
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant, spec in VARIANTS.items():
        if spec["kind"] == "classification_only":
            dataset = V73IQDataset(DEFAULT_TRAIN_MAT, val_indices[:4])
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
            model = build_benchmark_model(spec["model"], num_classes=6, width=8, head_dim=32).to(device)
            logits = model(batch["iq"].to(device))
            loss = nn.BCEWithLogitsLoss()(logits, batch["label"].to(device))
            output_shapes = {"logits": list(logits.shape)}
            mask_count = None
        else:
            dataset = V73IQDataset(DEFAULT_TRAIN_MAT, val_indices[:4], PARAMETER_COLUMNS)
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
            model = build_multitask_model(spec["model"], width=8, head_dim=32, parameter_hidden_dim=16).to(device)
            logits, prediction = model(batch["iq"].to(device))
            target = reshape_selected_parameters(batch["parameter_values"]).to(device)
            mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device)
            labels_batch = batch["label"].to(device)
            normalized_target = normalizer.transform(target)
            loss = nn.BCEWithLogitsLoss()(logits, labels_batch)
            loss = loss + 0.1 * masked_parameter_smooth_l1(prediction, normalized_target, labels_batch, mask)
            output_shapes = {"logits": list(logits.shape), "parameters": list(prediction.shape)}
            mask_count = int(mask.sum().item())
        loss.backward()
        finite_grads = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        rows.append({
            "variant": variant,
            "kind": spec["kind"],
            "model": spec["model"],
            "status": "verified_smoke" if finite_grads and all(dim > 0 for shape in output_shapes.values() for dim in shape) else "failed",
            "device": str(device),
            "output_shapes": json.dumps(output_shapes, sort_keys=True),
            "valid_parameter_entries_in_batch": mask_count if mask_count is not None else "",
            "loss": float(loss.detach().cpu()),
            "finite_gradients": finite_grads,
            "performance_claim": "none",
            "locked_test_used": False,
        })
        dataset.close()
        del model, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (RESULTS / "route_b_r1_baseline.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = "verified_smoke_only" if all(row["status"] == "verified_smoke" for row in rows) else "stopped/failed"
    receipt = {
        "stage": "R1_preflight",
        "status": status,
        "variants": rows,
        "config": str(CONFIGS / "route_b_r1_preregister.json"),
        "performance_validated": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (RECEIPTS / "route_b_r1_preflight_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    report = f"""# Route B R1 baseline preflight\n\n- Status: **{status}**\n- This is a forward/backward smoke only; no performance result is claimed.\n- Development split: `{len(train_indices)}` train / `{len(val_indices)}` validation, split seed `20260826`.\n- Test-v2 created/opened: `False`\n- Locked test used: `False`\n\n| Variant | Architecture path | Smoke status |\n|---|---|---|\n"""
    for row in rows:
        report += f"| {row['variant']} | {row['kind']} | {row['status']} |\n"
    report += "\nThe next R1 step is multi-seed development training with the frozen protocol in `configs/route_b_r1_preregister.json`. The current CSV must not be cited as accuracy, AUPR, NMAE, or confirmation evidence.\n"
    (REPORTS / "route_b_r1_preflight.md").write_text(report, encoding="utf-8")
    print(json.dumps({"stage": "R1_preflight", "status": status, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
