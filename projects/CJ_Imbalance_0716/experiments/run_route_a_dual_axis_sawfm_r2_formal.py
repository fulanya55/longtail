"""Formal five-seed Route-A SawFM frequency x visibility primary matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from cjlt.engine.thresholding import (  # noqa: E402
    apply_thresholds,
    select_per_label_weak_recall_fpr_thresholds,
)
from cjlt.losses.classification import (  # noqa: E402
    DistributionBalancedLoss,
    MultiLabelClassificationLoss,
)
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.dual_axis_sawfm import RoutedSawFMModel  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
OTHER_INDICES = (0, 2, 3, 4, 5)
SAW_INDEX = 1
SEEDS = (260811, 260812, 260813, 260814, 260815)
CELLS = (("FB", "VH"), ("FB", "VL"), ("LT", "VH"), ("LT", "VL"))
VARIANTS = ("M0_B0_Focal", "MF_B0_DB", "MV_B0_RawSaw", "MFV_B0_DB_RawSaw")

DATA_ROOT = Path(r"E:\LJQ_code\干扰数据\dual-axis-sawfm-development-v1")
VIEW_PATHS = {
    "VH": DATA_ROOT / "sawfm_level_05__jnr_30dB/closedset_6class_train.mat",
    "VL": DATA_ROOT / "sawfm_level_03__jnr_20dB/closedset_6class_train.mat",
}
R1_ROOT = ROOT / "outputs/route_a_dual_axis_next/r1_data_audit"
MANIFEST_PATHS = {
    "FB": R1_ROOT / "frequency_balanced_manifest.json",
    "LT": R1_ROOT / "label_long_tail_manifest.json",
}
DEFAULT_OUTPUT = ROOT / "outputs/route_a_dual_axis_next/r2_formal_primary"
SEAL_PATH = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float32)


def weak_mask(labels: np.ndarray, view: str) -> np.ndarray:
    matrix = np.asarray(labels, dtype=np.uint8)
    result = np.zeros_like(matrix, dtype=bool)
    if view == "VL":
        result[:, SAW_INDEX] = (matrix[:, SAW_INDEX] == 1) & (matrix.sum(axis=1) > 1)
    elif view != "VH":
        raise ValueError(f"unknown visibility view: {view}")
    return result


def dataset_loader(
    path: Path,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    sampler=None,
):
    dataset = V73SignalDataset(path, indices, class_mode="six", normalization="complex_rms")
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    return dataset, loader


def core_loss_value(criterion, logits, targets):
    element = criterion.elementwise(logits, targets)
    common = element[:, torch.as_tensor(OTHER_INDICES, device=logits.device)].mean()
    return common + 0.2 * element[:, SAW_INDEX].mean()


def macro_aupr(targets: np.ndarray, logits: np.ndarray) -> float:
    return float(
        np.mean(
            [
                average_precision_score(targets[:, index], logits[:, index])
                for index in range(targets.shape[1])
            ]
        )
    )


def saw_weak_auroc(targets, logits, weak):
    positive = (targets[:, SAW_INDEX] == 1) & weak[:, SAW_INDEX]
    negative = targets[:, SAW_INDEX] == 0
    if not positive.any() or not negative.any():
        return float("nan")
    truth = np.concatenate([np.ones(positive.sum()), np.zeros(negative.sum())])
    scores = np.concatenate([logits[positive, SAW_INDEX], logits[negative, SAW_INDEX]])
    return float(roc_auc_score(truth, scores))


def predict_core(model, path, indices, device, batch_size):
    dataset, loader = dataset_loader(path, indices, batch_size)
    logits, targets, sample_ids = [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                logits.append(model(signal).detach().cpu().float().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {
        "logits": np.concatenate(logits),
        "targets": np.concatenate(targets),
        "sample_ids": np.concatenate(sample_ids),
    }


def predict_raw(model, path, indices, device, batch_size):
    dataset, loader = dataset_loader(path, indices, batch_size)
    logits, targets, sample_ids = [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                output = model(signal[:, 0:1], signal[:, 1:2])
                logits.append(output.detach().cpu().float().numpy())
                targets.append(batch["label"][:, SAW_INDEX].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {
        "logits": np.concatenate(logits),
        "targets": np.concatenate(targets),
        "sample_ids": np.concatenate(sample_ids),
    }


def train_core(
    path,
    train_indices,
    early_indices,
    initial_state,
    loss_name,
    seed,
    device,
    args,
):
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) not in classifier_ids],
                "lr": 5e-5,
            },
            {"params": list(model.classifier.parameters()), "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    train_labels = read_labels(path)[train_indices]
    criterion = (
        MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0)
        if loss_name == "focal"
        else DistributionBalancedLoss(train_labels.sum(axis=0), len(train_labels))
    ).to(device)
    train_dataset, train_loader = dataset_loader(
        path, train_indices, args.batch_size, shuffle=True, seed=seed + 73000
    )
    best_state = None
    best_key = None
    best_epoch = 0
    stale = 0
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = core_loss_value(criterion, model(signal), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite core loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite core gradient")
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            early = predict_core(model, path, early_indices, device, args.eval_batch_size)
            macro = macro_aupr(early["targets"], early["logits"])
            saw = float(average_precision_score(early["targets"][:, SAW_INDEX], early["logits"][:, SAW_INDEX]))
            key = (macro, saw)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "early_macro_aupr": macro,
                    "early_sawfm_aupr": saw,
                    "backbone_lr": 5e-5,
                    "classifier_lr": 5e-4,
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
    finally:
        train_dataset.close()
    if best_state is None:
        raise RuntimeError("core did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    return model, best_state, best_epoch, history


def train_raw(path, train_indices, early_indices, initial_state, view, seed, device, args):
    model = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=16, topk=16, dropout=0.1, descriptor_mode="full"
    ).to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    labels = read_labels(path)[train_indices, SAW_INDEX]
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("RawSaw weighted sampler requires both positive and negative samples")
    weights = np.where(
        labels == 1,
        0.5 / float(positive_count),
        0.5 / float(negative_count),
    ).astype(np.float64)
    generator = torch.Generator().manual_seed(seed + 91000)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=2400,
        replacement=True,
        generator=generator,
    )
    train_dataset, train_loader = dataset_loader(
        path, train_indices, args.batch_size, sampler=sampler
    )
    full_labels = read_labels(path)
    early_weak = weak_mask(full_labels[early_indices], view)
    best_state = None
    best_key = None
    best_epoch = 0
    stale = 0
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                targets = batch["label"][:, SAW_INDEX].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(signal[:, 0:1], signal[:, 1:2])
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite RawSaw loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite RawSaw gradient")
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            early = predict_raw(model, path, early_indices, device, args.eval_batch_size)
            aupr = float(average_precision_score(early["targets"], early["logits"]))
            if view == "VL":
                positive = (early["targets"] == 1) & early_weak[:, SAW_INDEX]
                negative = early["targets"] == 0
                truth = np.concatenate([np.ones(positive.sum()), np.zeros(negative.sum())])
                scores = np.concatenate([early["logits"][positive], early["logits"][negative]])
                weak_auroc = float(roc_auc_score(truth, scores))
            else:
                weak_auroc = float("nan")
            key = (aupr, weak_auroc if np.isfinite(weak_auroc) else -1.0)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "early_sawfm_aupr": aupr,
                    "early_sawfm_weak_auroc": weak_auroc,
                    "learning_rate": 5e-4,
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
    finally:
        train_dataset.close()
    if best_state is None:
        raise RuntimeError("RawSaw did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    return model, best_state, best_epoch, history


def fit_platt(logits, targets):
    slopes, intercepts, iterations = [], [], []
    for index in range(logits.shape[1]):
        truth = targets[:, index].astype(np.uint8)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
        model.fit(logits[:, index : index + 1], truth)
        slopes.append(float(model.coef_[0, 0]))
        intercepts.append(float(model.intercept_[0]))
        iterations.append(int(model.n_iter_[0]))
    return {
        "method": "per_label_platt",
        "C": 1e6,
        "solver": "lbfgs",
        "max_iter": 1000,
        "slopes": slopes,
        "intercepts": intercepts,
        "iterations": iterations,
        "source_role": "paired_VH_VL_calibration",
    }


def apply_platt(logits, calibration):
    slopes = np.asarray(calibration["slopes"], dtype=np.float64)
    intercepts = np.asarray(calibration["intercepts"], dtype=np.float64)
    return sigmoid(np.asarray(logits, dtype=np.float64) * slopes + intercepts)


def expected_calibration_error(truth, probability, bins=15):
    target = np.asarray(truth, dtype=np.uint8)
    scores = np.asarray(probability, dtype=np.float64)
    result = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if selected.any():
            result += float(selected.mean()) * abs(float(scores[selected].mean()) - float(target[selected].mean()))
    return float(result)


def view_metrics(logits, targets, weak, probabilities, thresholds):
    prediction = apply_thresholds(probabilities, thresholds)
    per_label = {}
    for index, name in enumerate(CLASS_NAMES):
        positive = targets[:, index] == 1
        negative = ~positive
        weak_positive = positive & weak[:, index]
        strong_positive = positive & ~weak[:, index]
        if weak_positive.any() and negative.any():
            truth_rank = np.concatenate([np.ones(weak_positive.sum()), np.zeros(negative.sum())])
            score_rank = np.concatenate([logits[weak_positive, index], logits[negative, index]])
            weak_auroc = float(roc_auc_score(truth_rank, score_rank))
        else:
            weak_auroc = float("nan")
        per_label[name] = {
            "aupr": float(average_precision_score(targets[:, index], logits[:, index])),
            "weak_vs_negative_auroc": weak_auroc,
            "weak_recall": float(prediction[weak_positive, index].mean()) if weak_positive.any() else float("nan"),
            "strong_recall": float(prediction[strong_positive, index].mean()) if strong_positive.any() else float("nan"),
            "negative_fpr": float(prediction[negative, index].mean()),
            "ece": expected_calibration_error(targets[:, index], probabilities[:, index]),
            "brier": float(np.mean((probabilities[:, index] - targets[:, index]) ** 2)),
            "positive_count": int(positive.sum()),
            "weak_positive_count": int(weak_positive.sum()),
            "strong_positive_count": int(strong_positive.sum()),
            "negative_count": int(negative.sum()),
        }
    return {
        "macro_aupr": float(np.mean([per_label[name]["aupr"] for name in CLASS_NAMES])),
        "macro_f1": float(
            np.mean(
                [
                    2.0 * np.sum(prediction[:, index] & (targets[:, index] == 1))
                    / max(
                        2.0 * np.sum(prediction[:, index] & (targets[:, index] == 1))
                        + np.sum(prediction[:, index] & (targets[:, index] == 0))
                        + np.sum((~prediction[:, index]) & (targets[:, index] == 1)),
                        1.0,
                    )
                    for index in range(targets.shape[1])
                ]
            )
        ),
        "hamming_loss": float(np.not_equal(prediction, targets).mean()),
        "exact_match": float(np.all(prediction == targets, axis=1).mean()),
        "per_label": per_label,
    }


def predict_all_variants(focal, db, raw, path, indices, device, batch_size):
    focal_result = predict_core(focal, path, indices, device, batch_size)
    db_result = predict_core(db, path, indices, device, batch_size)
    raw_result = predict_raw(raw, path, indices, device, batch_size)
    if not (
        np.array_equal(focal_result["sample_ids"], db_result["sample_ids"])
        and np.array_equal(focal_result["sample_ids"], raw_result["sample_ids"])
        and np.array_equal(focal_result["targets"][:, SAW_INDEX], raw_result["targets"])
    ):
        raise AssertionError("variant prediction identity alignment failed")
    mv = focal_result["logits"].copy()
    mfv = db_result["logits"].copy()
    mv[:, SAW_INDEX] = raw_result["logits"]
    mfv[:, SAW_INDEX] = raw_result["logits"]
    return {
        "sample_ids": focal_result["sample_ids"],
        "targets": focal_result["targets"],
        "logits": {
            "M0_B0_Focal": focal_result["logits"],
            "MF_B0_DB": db_result["logits"],
            "MV_B0_RawSaw": mv,
            "MFV_B0_DB_RawSaw": mfv,
        },
    }


def profile_parameter_counts(focal, db, raw):
    core_focal = int(sum(parameter.numel() for parameter in focal.parameters()))
    core_db = int(sum(parameter.numel() for parameter in db.parameters()))
    expert = int(sum(parameter.numel() for parameter in raw.parameters()))
    if core_focal != 257511 or core_db != 257511 or expert != 5521:
        raise AssertionError("formal model parameter count drift")
    return {
        "M0_B0_Focal": core_focal,
        "MF_B0_DB": core_db,
        "MV_B0_RawSaw": core_focal + expert,
        "MFV_B0_DB_RawSaw": core_db + expert,
        "RawSaw_increment": expert,
    }


def run_seed_cell(args, labels, roles, frequency, view, seed, device, source_hashes):
    cell = f"{frequency}-{view}"
    run_root = args.output_root / f"seed_{seed}" / cell
    receipt_path = run_root / "run_receipt.json"
    summary_path = run_root / "cell_summary.json"
    manifest_path = MANIFEST_PATHS[frequency]
    manifest_sha = sha256(manifest_path)
    if args.resume and receipt_path.is_file() and summary_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") == "completed"
            and receipt.get("manifest_sha256") == manifest_sha
            and receipt.get("script_sha256") == sha256(Path(__file__))
        ):
            print(json.dumps({"seed": seed, "cell": cell, "status": "resumed_completed"}), flush=True)
            return json.loads(summary_path.read_text(encoding="utf-8"))
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base_receipt = {
        "seed": int(seed),
        "cell": cell,
        "frequency": frequency,
        "visibility": view,
        "manifest_sha256": manifest_sha,
        "source_hashes": source_hashes,
        "script_sha256": sha256(Path(__file__)),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_indices = np.asarray(manifest["selected_indices"], dtype=np.int64)
        if len(train_indices) != 2400:
            raise ValueError("formal optimization manifest must contain 2400 identities")
        early_indices = roles["early"]
        calibration_indices = roles["calibration"]
        threshold_indices = roles["threshold"]
        evaluation_indices = roles["evaluation"]
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device)
        core_template = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
        core_initial = copy.deepcopy(core_template.state_dict())
        core_initial_sha = state_sha256(core_initial)
        torch.manual_seed(int(seed) + 500000)
        raw_template = RawStage1SawFMFrontend(
            mode="raw", hidden_channels=16, topk=16, dropout=0.1, descriptor_mode="full"
        )
        raw_initial = copy.deepcopy(raw_template.state_dict())
        raw_initial_sha = state_sha256(raw_initial)
        path = VIEW_PATHS[view]
        focal, focal_state, focal_epoch, focal_history = train_core(
            path, train_indices, early_indices, core_initial, "focal", seed, device, args
        )
        db, db_state, db_epoch, db_history = train_core(
            path, train_indices, early_indices, core_initial, "db", seed, device, args
        )
        raw, raw_state, raw_epoch, raw_history = train_raw(
            path, train_indices, early_indices, raw_initial, view, seed, device, args
        )
        focal_path = run_root / "m0_focal_core.pt"
        db_path = run_root / "mf_db_core.pt"
        raw_path = run_root / "rawsaw.pt"
        torch.save(
            {"model_state_dict": focal_state, "best_epoch": focal_epoch, "initial_state_sha256": core_initial_sha},
            focal_path,
        )
        torch.save(
            {"model_state_dict": db_state, "best_epoch": db_epoch, "initial_state_sha256": core_initial_sha},
            db_path,
        )
        torch.save(
            {"model_state_dict": raw_state, "best_epoch": raw_epoch, "initial_state_sha256": raw_initial_sha},
            raw_path,
        )
        write_history(run_root / "m0_focal_history.csv", focal_history)
        write_history(run_root / "mf_db_history.csv", db_history)
        write_history(run_root / "rawsaw_history.csv", raw_history)

        role_predictions = {}
        for role_name, indices in (
            ("calibration", calibration_indices),
            ("threshold", threshold_indices),
            ("evaluation", evaluation_indices),
        ):
            role_predictions[role_name] = {
                visibility: predict_all_variants(
                    focal, db, raw, VIEW_PATHS[visibility], indices, device, args.eval_batch_size
                )
                for visibility in ("VH", "VL")
            }

        variant_results = {}
        prediction_payload = {}
        for variant in VARIANTS:
            calibration_logits = np.concatenate(
                [role_predictions["calibration"][visibility]["logits"][variant] for visibility in ("VH", "VL")]
            )
            calibration_targets = np.concatenate(
                [role_predictions["calibration"][visibility]["targets"] for visibility in ("VH", "VL")]
            )
            calibration = fit_platt(calibration_logits, calibration_targets)
            threshold_logits = np.concatenate(
                [role_predictions["threshold"][visibility]["logits"][variant] for visibility in ("VH", "VL")]
            )
            threshold_targets = np.concatenate(
                [role_predictions["threshold"][visibility]["targets"] for visibility in ("VH", "VL")]
            )
            threshold_weak = np.concatenate(
                [weak_mask(role_predictions["threshold"][visibility]["targets"], visibility) for visibility in ("VH", "VL")]
            )
            threshold_probability = apply_platt(threshold_logits, calibration)
            selection = select_per_label_weak_recall_fpr_thresholds(
                threshold_targets,
                threshold_probability,
                threshold_weak,
                max_negative_fpr=0.20,
                source_role="threshold_selection",
                grid=np.arange(0.05, 0.951, 0.01),
            )
            by_view = {}
            for visibility in ("VH", "VL"):
                result = role_predictions["evaluation"][visibility]
                logits = result["logits"][variant]
                targets = result["targets"]
                probabilities = apply_platt(logits, calibration)
                current_weak = weak_mask(targets, visibility)
                by_view[visibility] = view_metrics(
                    logits, targets, current_weak, probabilities, selection.thresholds
                )
                prediction_payload[f"{variant}_{visibility}_logits"] = logits.astype(np.float32)
                prediction_payload[f"{variant}_{visibility}_probabilities"] = probabilities.astype(np.float32)
            variant_results[variant] = {
                "calibration": calibration,
                "thresholds": [float(value) for value in selection.thresholds],
                "threshold_mode": selection.mode,
                "by_view": by_view,
            }
        np.savez_compressed(
            run_root / "evaluation_predictions.npz",
            sample_ids=role_predictions["evaluation"]["VH"]["sample_ids"],
            targets=role_predictions["evaluation"]["VH"]["targets"],
            **prediction_payload,
        )
        with torch.no_grad():
            routed = RoutedSawFMModel(focal, raw).to(device)
            probe = torch.randn(4, 2, 3000, device=device)
            output, parts = routed(probe, return_components=True)
            invariant = bool(
                torch.equal(output[:, OTHER_INDICES], parts["core_logits"][:, OTHER_INDICES])
            )
        summary = {
            "status": "completed",
            "evidence_level": "five_seed_development_component",
            "seed": int(seed),
            "cell": cell,
            "frequency": frequency,
            "training_visibility": view,
            "train_identity_count": int(len(train_indices)),
            "role_identity_counts": {
                "early": int(len(early_indices)),
                "calibration": int(len(calibration_indices)),
                "threshold": int(len(threshold_indices)),
                "evaluation": int(len(evaluation_indices)),
            },
            "core_initial_state_sha256": core_initial_sha,
            "raw_initial_state_sha256": raw_initial_sha,
            "best_epochs": {"M0_B0_Focal": focal_epoch, "MF_B0_DB": db_epoch, "RawSaw": raw_epoch},
            "parameter_counts": profile_parameter_counts(focal, db, raw),
            "non_saw_routing_bitwise_invariant": invariant,
            "checkpoints": {
                "M0_B0_Focal": {"path": str(focal_path), "sha256": sha256(focal_path)},
                "MF_B0_DB": {"path": str(db_path), "sha256": sha256(db_path)},
                "RawSaw": {"path": str(raw_path), "sha256": sha256(raw_path)},
            },
            "predictions_sha256": sha256(run_root / "evaluation_predictions.npz"),
            "variants": variant_results,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "elapsed_seconds": float(time.time() - started),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
        write_json(summary_path, summary)
        write_json(
            receipt_path,
            {
                **base_receipt,
                "status": "completed",
                "summary_sha256": sha256(summary_path),
                "checkpoint_sha256": {
                    name: value["sha256"] for name, value in summary["checkpoints"].items()
                },
                "elapsed_seconds": summary["elapsed_seconds"],
            },
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "cell": cell,
                    "status": "completed",
                    "best_epochs": summary["best_epochs"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return summary
    except Exception as exception:
        write_json(
            receipt_path,
            {
                **base_receipt,
                "status": "aborted_engineering_failure",
                "error_type": type(exception).__name__,
                "error": str(exception),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": float(time.time() - started),
            },
        )
        raise


def mean_ci(values, seed, repetitions=20000):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    samples = rng.choice(array, size=(int(repetitions), len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "bootstrap_95_ci": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "count": int(len(array)),
        "positive_count": int((array > 0).sum()),
    }


def metric(row, variant, view, name, label=None):
    payload = row["variants"][variant]["by_view"][view]
    return float(payload["per_label"][label][name] if label is not None else payload[name])


def aggregate(rows):
    by_key = {(row["seed"], row["cell"]): row for row in rows}
    frequency_rows = []
    visibility_rows = []
    combined_rows = []
    safety_rows = []
    for seed in SEEDS:
        lt_deltas = []
        fb_deltas = []
        for cell in ("LT-VH", "LT-VL"):
            row = by_key[(seed, cell)]
            view = cell.split("-")[1]
            lt_deltas.append(metric(row, "MF_B0_DB", view, "macro_aupr") - metric(row, "M0_B0_Focal", view, "macro_aupr"))
        for cell in ("FB-VH", "FB-VL"):
            row = by_key[(seed, cell)]
            view = cell.split("-")[1]
            fb_deltas.append(metric(row, "MF_B0_DB", view, "macro_aupr") - metric(row, "M0_B0_Focal", view, "macro_aupr"))
        frequency_rows.append({"seed": seed, "lt_macro_aupr_delta": float(np.mean(lt_deltas)), "fb_macro_aupr_delta": float(np.mean(fb_deltas))})

        weak_auroc, weak_recall, candidate_fpr, strong_delta = [], [], [], []
        for cell in ("FB-VL", "LT-VL"):
            row = by_key[(seed, cell)]
            weak_auroc.append(metric(row, "MV_B0_RawSaw", "VL", "weak_vs_negative_auroc", "SawFM") - metric(row, "M0_B0_Focal", "VL", "weak_vs_negative_auroc", "SawFM"))
            weak_recall.append(metric(row, "MV_B0_RawSaw", "VL", "weak_recall", "SawFM") - metric(row, "M0_B0_Focal", "VL", "weak_recall", "SawFM"))
            candidate_fpr.append(metric(row, "MV_B0_RawSaw", "VL", "negative_fpr", "SawFM"))
            strong_delta.append(metric(row, "MV_B0_RawSaw", "VL", "strong_recall", "SawFM") - metric(row, "M0_B0_Focal", "VL", "strong_recall", "SawFM"))
        visibility_rows.append({
            "seed": seed,
            "weak_auroc_delta": float(np.mean(weak_auroc)),
            "weak_recall_delta": float(np.mean(weak_recall)),
            "candidate_fpr": float(np.mean(candidate_fpr)),
            "strong_recall_delta": float(np.mean(strong_delta)),
        })

        row = by_key[(seed, "LT-VL")]
        best_macro = max(metric(row, "MF_B0_DB", "VL", "macro_aupr"), metric(row, "MV_B0_RawSaw", "VL", "macro_aupr"))
        best_recall = max(metric(row, "MF_B0_DB", "VL", "weak_recall", "SawFM"), metric(row, "MV_B0_RawSaw", "VL", "weak_recall", "SawFM"))
        combined_rows.append({
            "seed": seed,
            "macro_aupr_delta_vs_best": metric(row, "MFV_B0_DB_RawSaw", "VL", "macro_aupr") - best_macro,
            "weak_recall_delta_vs_best": metric(row, "MFV_B0_DB_RawSaw", "VL", "weak_recall", "SawFM") - best_recall,
        })
        for label in CLASS_NAMES:
            if label == "SawFM":
                continue
            safety_rows.append({
                "seed": seed,
                "label": label,
                "aupr_delta": metric(row, "MFV_B0_DB_RawSaw", "VL", "aupr", label) - metric(row, "M0_B0_Focal", "VL", "aupr", label),
            })

    frequency = {
        "per_seed": frequency_rows,
        "lt_macro_aupr_delta": mean_ci([row["lt_macro_aupr_delta"] for row in frequency_rows], 26082001),
        "fb_macro_aupr_delta": mean_ci([row["fb_macro_aupr_delta"] for row in frequency_rows], 26082002),
    }
    visibility = {"per_seed": visibility_rows}
    for index, name in enumerate(
        ("weak_auroc_delta", "weak_recall_delta", "candidate_fpr", "strong_recall_delta")
    ):
        visibility[name] = mean_ci(
            [row[name] for row in visibility_rows], 26082100 + index
        )
    combined = {
        "per_seed": combined_rows,
        "macro_aupr_delta_vs_best": mean_ci([row["macro_aupr_delta_vs_best"] for row in combined_rows], 26082201),
        "weak_recall_delta_vs_best": mean_ci([row["weak_recall_delta_vs_best"] for row in combined_rows], 26082202),
    }
    safety_by_label = {
        label: mean_ci([row["aupr_delta"] for row in safety_rows if row["label"] == label], 26082300 + index)
        for index, label in enumerate(CLASS_NAMES)
        if label != "SawFM"
    }
    gates = {
        "MF_frequency_main_effect": (
            frequency["lt_macro_aupr_delta"]["mean"] >= 0.005
            and frequency["lt_macro_aupr_delta"]["positive_count"] >= 4
            and frequency["fb_macro_aupr_delta"]["mean"] >= -0.005
        ),
        "MV_visibility_main_effect": (
            visibility["weak_auroc_delta"]["mean"] >= 0.020
            and visibility["weak_recall_delta"]["mean"] >= 0.050
            and visibility["candidate_fpr"]["mean"] <= 0.20
            and visibility["strong_recall_delta"]["mean"] >= -0.01
        ),
        "MFV_point_effect": (
            combined["macro_aupr_delta_vs_best"]["mean"] >= 0.005
            or combined["weak_recall_delta_vs_best"]["mean"] >= 0.020
        ),
        "non_saw_safety": all(value["mean"] >= -0.01 for value in safety_by_label.values()),
    }
    gates["controls_admitted"] = gates["MF_frequency_main_effect"] and gates["MV_visibility_main_effect"]
    return {
        "frequency": frequency,
        "visibility": visibility,
        "combined": combined,
        "non_saw_safety_by_label": safety_by_label,
        "gates": gates,
    }


def validate_sources():
    data_audit = json.loads((R1_ROOT / "data_audit.json").read_text(encoding="utf-8"))
    if not data_audit["checks"]["r2_smoke_admitted"]:
        raise ValueError("R1 data audit did not admit R2")
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    if not (
        seal["status"] == "sealed"
        and seal["test_v3"]["access_authorized"] is False
        and seal["test_v3"]["analysis_allowed"] is False
        and seal["test_v3"]["used"] is False
    ):
        raise ValueError("test-v3 seal is not intact")
    source_hashes = {}
    for visibility, path in VIEW_PATHS.items():
        current = sha256(path)
        expected = data_audit["files"]["high_visibility" if visibility == "VH" else "low_visibility"]["sha256"]
        if current != expected:
            raise ValueError(f"{visibility} source SHA mismatch")
        source_hashes[visibility] = current
    return data_audit, source_hashes


def run(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    terminal_path = args.output_root / "terminal_receipt.json"
    started = time.time()
    try:
        data_audit, source_hashes = validate_sources()
        labels = read_labels(VIEW_PATHS["VH"])
        if not np.array_equal(labels, read_labels(VIEW_PATHS["VL"])):
            raise ValueError("formal paired-view labels drifted")
        saved_roles = np.load(R1_ROOT / "role_indices.npz")
        roles = {
            name: saved_roles[name].astype(np.int64)
            for name in ("early", "calibration", "threshold", "evaluation")
        }
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise RuntimeError("formal R2 requires CUDA")
        formal_protocol = bool(
            tuple(args.seeds) == SEEDS
            and tuple(tuple(value.split("-")) for value in args.cells) == CELLS
            and args.epochs == 36
            and args.patience == 8
            and args.batch_size == 64
        )
        if not formal_protocol:
            raise ValueError("formal R2 arguments do not match the frozen preregistration")
        config = {
            "status": "frozen",
            "experiment": "RouteA_DualAxis_SawFM_R2_FormalPrimary",
            "evidence_level": "five_seed_development_only",
            "seeds": list(args.seeds),
            "cells": list(args.cells),
            "variants": list(VARIANTS),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "core_optimizer": {"name": "AdamW", "backbone_lr": 5e-5, "classifier_lr": 5e-4, "weight_decay": 1e-4},
            "raw_optimizer": {"name": "AdamW", "lr": 5e-4, "weight_decay": 1e-4},
            "raw_sampler": "SawFM 50/50 positive-negative weighted replacement, 2400 draws/epoch",
            "gradient_clip": 5.0,
            "calibration": "per-label Platt C=1e6 lbfgs max_iter=1000 on paired VH+VL calibration role",
            "threshold": "0.05:0.01:0.95 maximize weak recall with per-label negative FPR<=0.20 on paired VH+VL threshold role",
            "source_hashes": source_hashes,
            "r1_data_audit_sha256": sha256(R1_ROOT / "data_audit.json"),
            "test_files_opened": False,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
        write_json(args.output_root / "resolved_config.json", config)
        rows = []
        for seed in args.seeds:
            for cell_text in args.cells:
                frequency, view = cell_text.split("-")
                rows.append(
                    run_seed_cell(
                        args, labels, roles, frequency, view, int(seed), device, source_hashes
                    )
                )
        result = aggregate(rows)
        payload = {
            "status": "completed",
            "evidence_level": "five_seed_development_only",
            "formal_protocol": True,
            "rows": rows,
            "aggregate": result,
            "next_decision": (
                "run_preregistered_controls"
                if result["gates"]["controls_admitted"]
                else "stop_primary_mechanism_before_controls"
            ),
            "test_files_opened": False,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
        summary_path = args.output_root / "formal_primary_summary.json"
        write_json(summary_path, payload)
        write_json(
            terminal_path,
            {
                "status": "completed",
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "next_decision": payload["next_decision"],
                "elapsed_seconds": float(time.time() - started),
                "locked_test_used": False,
                "test_v2_used": False,
                "test_v3_used": False,
            },
        )
        print(json.dumps({"status": "completed", "aggregate": result, "next_decision": payload["next_decision"]}, ensure_ascii=False, indent=2))
        return payload
    except Exception as exception:
        write_json(
            terminal_path,
            {
                "status": "aborted_engineering_failure",
                "error_type": type(exception).__name__,
                "error": str(exception),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": float(time.time() - started),
                "locked_test_used": False,
                "test_v2_used": False,
                "test_v3_used": False,
            },
        )
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--cells", nargs="+", default=[f"{frequency}-{view}" for frequency, view in CELLS])
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
