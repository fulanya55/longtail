"""R2 execution smoke for the preregistered SawFM dual-axis model matrix."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from cjlt.losses.classification import (  # noqa: E402
    DistributionBalancedLoss,
    MultiLabelClassificationLoss,
)
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.dual_axis_sawfm import RoutedSawFMModel  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402


DATA_ROOT = Path(r"E:\LJQ_code\干扰数据\dual-axis-sawfm-development-v1")
VIEW_PATHS = {
    "VH": DATA_ROOT / "sawfm_level_05__jnr_30dB/closedset_6class_train.mat",
    "VL": DATA_ROOT / "sawfm_level_03__jnr_20dB/closedset_6class_train.mat",
}
AUDIT_ROOT = ROOT / "outputs/route_a_dual_axis_next/r1_data_audit"
MANIFEST_PATHS = {
    "FB": AUDIT_ROOT / "frequency_balanced_manifest.json",
    "LT": AUDIT_ROOT / "label_long_tail_manifest.json",
}
OUTPUT_ROOT = ROOT / "outputs/route_a_dual_axis_next/r2_smoke"
CELLS = (("FB", "VH"), ("LT", "VL"))
SEED = 260811


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def loader(path: Path, indices: np.ndarray, shuffle: bool, seed: int, batch_size: int):
    dataset = V73SignalDataset(path, indices, class_mode="six", normalization="complex_rms")
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return dataset, DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


def loss_value(criterion, logits, targets):
    element = criterion.elementwise(logits, targets)
    common = element[:, torch.as_tensor((0, 2, 3, 4, 5), device=logits.device)].mean()
    return common + 0.2 * element[:, 1].mean()


def train_core(path, train_indices, early_indices, initial_state, loss_name, device):
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    classifier_ids = {id(value) for value in model.classifier.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [value for value in model.parameters() if id(value) not in classifier_ids], "lr": 5e-5},
            {"params": list(model.classifier.parameters()), "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    labels = read_labels(path)[train_indices]
    criterion = (
        MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0)
        if loss_name == "focal"
        else DistributionBalancedLoss(labels.sum(axis=0), len(labels))
    ).to(device)
    train_dataset, train_loader = loader(path, train_indices, True, SEED + 17, 64)
    history = []
    try:
        for epoch in range(1, 3):
            model.train()
            losses = []
            for batch in train_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                value = loss_value(criterion, model(signal), targets)
                if not torch.isfinite(value):
                    raise FloatingPointError("non-finite core loss")
                value.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite core gradient")
                optimizer.step()
                losses.append(float(value.detach().cpu()))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    finally:
        train_dataset.close()
    return model, history


def train_raw(path, train_indices, device):
    torch.manual_seed(SEED + 91)
    model = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=16, topk=16, dropout=0.1, descriptor_mode="full"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_dataset, train_loader = loader(path, train_indices, True, SEED + 17, 64)
    history = []
    try:
        for epoch in range(1, 3):
            model.train()
            losses = []
            for batch in train_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                targets = batch["label"][:, 1].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(signal[:, 0:1], signal[:, 1:2])
                value = F.binary_cross_entropy_with_logits(logits, targets)
                if not torch.isfinite(value):
                    raise FloatingPointError("non-finite RawSaw loss")
                value.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite RawSaw gradient")
                optimizer.step()
                losses.append(float(value.detach().cpu()))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    finally:
        train_dataset.close()
    return model, history


def predict(model, path, indices, device):
    dataset, data_loader = loader(path, indices, False, 0, 128)
    logits, targets = [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in data_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                logits.append(model(signal).detach().cpu().numpy())
                targets.append(batch["label"].numpy())
    finally:
        dataset.close()
    return np.concatenate(logits), np.concatenate(targets)


def metrics(logits, targets):
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    per_label = [average_precision_score(targets[:, index], probability[:, index]) for index in range(6)]
    saw_truth = targets[:, 1].astype(np.uint8)
    saw_auroc = roc_auc_score(saw_truth, probability[:, 1])
    return {
        "macro_aupr": float(np.mean(per_label)),
        "sawfm_aupr": float(per_label[1]),
        "sawfm_presence_auroc": float(saw_auroc),
        "logits_finite": bool(np.isfinite(logits).all()),
    }


def run_cell(frequency, visibility, device):
    cell = f"{frequency}-{visibility}"
    path = VIEW_PATHS[visibility]
    manifest = json.loads(MANIFEST_PATHS[frequency].read_text(encoding="utf-8"))
    roles = np.load(AUDIT_ROOT / "role_indices.npz")
    rng = np.random.RandomState(SEED + (0 if frequency == "FB" else 100))
    train_indices = np.asarray(manifest["selected_indices"], dtype=np.int64)
    rng.shuffle(train_indices)
    train_indices = train_indices[:256]
    early_indices = roles["early"].astype(np.int64).copy()
    rng.shuffle(early_indices)
    early_indices = early_indices[:256]

    torch.manual_seed(SEED)
    template = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    initial_state = copy.deepcopy(template.state_dict())
    initial_sha = state_sha256(initial_state)
    focal, focal_history = train_core(path, train_indices, early_indices, initial_state, "focal", device)
    db, db_history = train_core(path, train_indices, early_indices, initial_state, "db", device)
    raw, raw_history = train_raw(path, train_indices, device)
    mv = RoutedSawFMModel(focal, raw).to(device)
    mfv = RoutedSawFMModel(db, raw).to(device)

    outputs = {}
    for name, model in (("M0_B0_Focal", focal), ("MF_B0_DB", db), ("MV_B0_RawSaw", mv), ("MFV_B0_DB_RawSaw", mfv)):
        logits, targets = predict(model, path, early_indices, device)
        outputs[name] = metrics(logits, targets)
    with torch.no_grad():
        probe = torch.randn(3, 2, 3000, device=device)
        routed, components = mv(probe, return_components=True)
        other_exact = bool(torch.equal(routed[:, (0, 2, 3, 4, 5)], components["core_logits"][:, (0, 2, 3, 4, 5)]))
    cell_root = OUTPUT_ROOT / cell
    cell_root.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": focal.state_dict(), "initial_state_sha256": initial_sha}, cell_root / "m0_focal.pt")
    torch.save({"model_state_dict": db.state_dict(), "initial_state_sha256": initial_sha}, cell_root / "mf_db.pt")
    torch.save({"model_state_dict": raw.state_dict()}, cell_root / "rawsaw.pt")
    history = {"focal": focal_history, "db": db_history, "rawsaw": raw_history}
    write_json(cell_root / "history.json", history)
    return {
        "cell": cell,
        "frequency": frequency,
        "visibility": visibility,
        "train_identity_count": int(len(train_indices)),
        "early_identity_count": int(len(early_indices)),
        "shared_core_initial_state_sha256": initial_sha,
        "core_parameter_count": int(sum(value.numel() for value in focal.parameters())),
        "rawsaw_parameter_count": int(sum(value.numel() for value in raw.parameters())),
        "routed_non_saw_logits_bitwise_equal_to_core": other_exact,
        "metrics": outputs,
        "history": history,
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("R2 smoke requires CUDA")
    rows = [run_cell(frequency, visibility, device) for frequency, visibility in CELLS]
    checks = {
        "all_losses_finite": all(
            np.isfinite(item["train_loss"])
            for row in rows for history in row["history"].values() for item in history
        ),
        "all_logits_finite": all(
            metric["logits_finite"]
            for row in rows for metric in row["metrics"].values()
        ),
        "shared_core_initialization_within_cell": all(bool(row["shared_core_initial_state_sha256"]) for row in rows),
        "rawsaw_parameter_count_5521": all(row["rawsaw_parameter_count"] == 5521 for row in rows),
        "non_saw_routing_invariant": all(row["routed_non_saw_logits_bitwise_equal_to_core"] for row in rows),
        "formal_training_admitted": False,
    }
    checks["formal_training_admitted"] = all(value for key, value in checks.items() if key != "formal_training_admitted")
    payload = {
        "status": "completed" if checks["formal_training_admitted"] else "stopped",
        "evidence_level": "smoke_only_not_performance_evidence",
        "experiment": "RouteA_DualAxis_SawFM_R2_Smoke",
        "device": str(device),
        "seed": SEED,
        "cells": ["FB-VH", "LT-VL"],
        "epochs": 2,
        "max_train_identities": 256,
        "max_early_identities": 256,
        "rows": rows,
        "checks": checks,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    summary_path = OUTPUT_ROOT / "smoke_summary.json"
    write_json(summary_path, payload)
    write_json(OUTPUT_ROOT / "run_receipt.json", {
        "status": payload["status"],
        "decision": "admit_formal_r2" if checks["formal_training_admitted"] else "stop_before_formal_r2",
        "summary_sha256": sha256(summary_path),
        "script_sha256": sha256(Path(__file__)),
        "elapsed_seconds": float(time.time() - started),
        "model_trained": True,
        "evidence_level": "smoke_only_not_performance_evidence",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    })
    print(json.dumps({"status": payload["status"], "checks": checks, "rows": rows}, indent=2, ensure_ascii=False))
    if not checks["formal_training_admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
