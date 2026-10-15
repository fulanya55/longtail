"""Route-A five-class, label-frequency long-tail pilot (SawFM removed).

This runner intentionally has no power-view input or threshold tuning on test.
It materializes auditable manifests and compares the validated B0 backbone under
balanced/Focal, LabelIF4/Focal, and LabelIF4/DB training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.longtail_manifest import build_label_longtail
from cjlt.data.v73_dataset import V73SignalDataset
from cjlt.losses.classification import DistributionBalancedLoss, MultiLabelClassificationLoss
from cjlt.models.backbone_adapter import build_backbone, count_trainable_parameters


DATA_ROOT = Path(r"E:\LJQ_code\干扰数据\复合干扰数据0716")
TRAIN_PATH = DATA_ROOT / "closedset_6class_train.mat"
TEST_PATH = DATA_ROOT / "closedset_6class_test.mat"
OUTPUT_ROOT = ROOT / "outputs" / "route_a_five_class_frequency_lt_v1_equal_budget"
DATASET_ROOT = DATA_ROOT / "route_a_five_class_frequency_lt_v1"
CLASS_NAMES = ("FMB", "SinFM", "MFT", "VFT", "ISRJ")
SEEDS = (271001, 271002, 271003)
ARMS = (
    "B0-Focal-Balanced",
    "B0-Focal-LabelIF4",
    "B0-DB-LabelIF4",
    "B0-DB-Balanced",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_labels6(path: Path) -> np.ndarray:
    with h5py.File(str(path), "r") as handle:
        return np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T


def split_by_combo(indices: np.ndarray, labels5: np.ndarray, seed: int, train_fraction: float = 0.8):
    rng = np.random.RandomState(int(seed))
    train_parts, early_parts = [], []
    for combo in sorted({tuple(row.tolist()) for row in labels5[indices]}):
        group = indices[np.all(labels5[indices] == np.asarray(combo), axis=1)].copy()
        rng.shuffle(group)
        cut = int(round(len(group) * float(train_fraction)))
        cut = min(max(cut, 1), len(group) - 1) if len(group) > 1 else len(group)
        train_parts.append(group[:cut])
        if cut < len(group):
            early_parts.append(group[cut:])
    train = np.concatenate(train_parts).astype(np.int64)
    early = np.concatenate(early_parts).astype(np.int64) if early_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(early)
    return train, early


def ensure_manifests() -> dict:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    train6 = read_labels6(TRAIN_PATH)
    test6 = read_labels6(TEST_PATH)
    if train6.shape[1] != 6 or test6.shape[1] != 6:
        raise ValueError("expected six source labels")
    train_valid = np.flatnonzero(train6[:, 1] == 0).astype(np.int64)
    test_valid = np.flatnonzero(test6[:, 1] == 0).astype(np.int64)
    train5 = train6[:, [0, 2, 3, 4, 5]]
    test5 = test6[:, [0, 2, 3, 4, 5]]
    if not np.all(train5[train_valid].sum(axis=1) > 0):
        raise ValueError("filtered train contains an empty five-class label")
    if not np.all(test5[test_valid].sum(axis=1) > 0):
        raise ValueError("filtered test contains an empty five-class label")

    # IF4 is the strongest exactly feasible geometric target on the symmetric
    # 0716 combination table without fabricating samples or violating combos.
    target = np.asarray([4000, 3000, 2000, 1500, 1000], dtype=np.int64)
    manifest = build_label_longtail(
        train5,
        train_valid,
        target_label_counts=target,
        seed=271000,
        tolerance=0.05,
        source_split_sha256=sha256_file(TRAIN_PATH),
    )
    payload = manifest.to_dict()
    payload.update({
        "source_file": str(TRAIN_PATH),
        "class_names": list(CLASS_NAMES),
        "source_label_order": ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"],
        "selected_source_indices": [0, 2, 3, 4, 5],
        "excluded_class": "SawFM",
        "excluded_positive_samples": True,
        "power_longtail": False,
        "power_view_dirs_used": [],
        "actual_imbalance_factor": float(target.max() / target.min()),
        "requested_label_if": 4.0,
    })
    (DATASET_ROOT / "label_if4_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    source = {
        "protocol": "five_class_sawfm_removed",
        "train_file": str(TRAIN_PATH),
        "test_file": str(TEST_PATH),
        "train_sha256": sha256_file(TRAIN_PATH),
        "test_sha256": sha256_file(TEST_PATH),
        "class_names": list(CLASS_NAMES),
        "excluded_class": "SawFM",
        "excluded_positive_samples": True,
        "train_source_count": int(len(train6)),
        "train_retained_count": int(len(train_valid)),
        "test_source_count": int(len(test6)),
        "test_retained_count": int(len(test_valid)),
        "train_counts_before_filter": train6.sum(axis=0).astype(int).tolist(),
        "train_counts_after_filter": train5[train_valid].sum(axis=0).astype(int).tolist(),
        "test_counts_after_filter": test5[test_valid].sum(axis=0).astype(int).tolist(),
        "train_retained_indices": train_valid.tolist(),
        "test_retained_indices": test_valid.tolist(),
        "power_longtail": False,
        "power_view_dirs_used": [],
    }
    (DATASET_ROOT / "source_filter_audit.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(DATASET_ROOT / "test_v1_indices.npy", test_valid)
    return {"train_labels5": train5, "test_labels5": test5, "train_valid": train_valid, "test_valid": test_valid, "manifest": manifest, "target": target, "source": source}


def loader(path: Path, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int | None = None, draws: int | None = None):
    dataset = V73SignalDataset(path, indices, class_mode="five", normalization="complex_rms")
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if shuffle and draws is not None:
        sampler = WeightedRandomSampler(
            torch.ones(len(dataset), dtype=torch.double),
            num_samples=int(draws),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available(), generator=None if sampler is not None else generator)


def evaluate(model, path: Path, indices: np.ndarray, device: torch.device, batch_size: int = 256):
    dataset, data_loader = loader(path, indices, batch_size, shuffle=False)
    model.eval()
    logits_parts, targets_parts = [], []
    try:
        with torch.no_grad():
            for batch in data_loader:
                logits_parts.append(model(batch["signal"].to(device, non_blocking=True)).detach().cpu().numpy())
                targets_parts.append(batch["label"].numpy())
    finally:
        dataset.close()
    logits = np.concatenate(logits_parts, axis=0)
    targets = np.concatenate(targets_parts, axis=0).astype(np.uint8)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    per = {}
    for i, name in enumerate(CLASS_NAMES):
        per[name] = {
            "aupr": float(average_precision_score(targets[:, i], probabilities[:, i])),
            "auroc": float(roc_auc_score(targets[:, i], probabilities[:, i])),
            "f1_at_0_5": float(f1_score(targets[:, i], probabilities[:, i] >= 0.5, zero_division=0)),
            "positive_count": int(targets[:, i].sum()),
        }
    return {
        "macro_aupr": float(np.mean([v["aupr"] for v in per.values()])),
        "macro_auroc": float(np.mean([v["auroc"] for v in per.values()])),
        "macro_f1_at_0_5": float(np.mean([v["f1_at_0_5"] for v in per.values()])),
        "per_label": per,
        "logits": logits,
        "targets": targets,
    }


def train_arm(arm: str, seed: int, train_indices: np.ndarray, early_indices: np.ndarray, train_labels5: np.ndarray, device: torch.device, args: argparse.Namespace, arm_dir: Path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_backbone("dfms_iq", num_classes=5, width=16, dropout=0.2).to(device)
    if "DB" in arm:
        criterion = DistributionBalancedLoss(train_labels5[train_indices].sum(axis=0), len(train_indices)).to(device)
    else:
        criterion = MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0).to(device)
    classifier_ids = {id(p) for p in model.classifier.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in classifier_ids], "lr": 5e-5},
        {"params": list(model.classifier.parameters()), "lr": 5e-4},
    ], weight_decay=1e-4)
    dataset, data_loader = loader(TRAIN_PATH, train_indices, args.batch_size, shuffle=True, seed=seed + 17, draws=args.draws_per_epoch)
    best_state, best_key, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            for batch in data_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(signal), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            early = evaluate(model, TRAIN_PATH, early_indices, device, args.eval_batch_size)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "early_macro_aupr": early["macro_aupr"], "elapsed_seconds": time.time() - started}
            history.append(row)
            if early["macro_aupr"] > best_key:
                best_key = early["macro_aupr"]; best_epoch = epoch; stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
    finally:
        dataset.close()
    if best_state is None:
        raise RuntimeError(f"no checkpoint for {arm} seed={seed}")
    model.load_state_dict(best_state, strict=True)
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, arm_dir / "model.pt")
    (arm_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    early_full = evaluate(model, TRAIN_PATH, early_indices, device, args.eval_batch_size)
    test = evaluate(model, TEST_PATH, DATA["test_valid"], device, args.eval_batch_size)
    early = {key: value for key, value in early_full.items() if key not in ("logits", "targets")}
    np.savez_compressed(arm_dir / "test_predictions.npz", logits=test.pop("logits"), targets=test.pop("targets"))
    return {"arm": arm, "seed": int(seed), "best_epoch": int(best_epoch), "train_count": int(len(train_indices)), "early_count": int(len(early_indices)), "train_label_counts": train_labels5[train_indices].sum(axis=0).astype(int).tolist(), "parameter_count": count_trainable_parameters(model), "early": early, "test": test, "history": history}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--draws-per-epoch", type=int, default=9920)
    parser.add_argument("--max-seeds", type=int, default=3)
    return parser.parse_args()


def main():
    global DATA
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    DATA = ensure_manifests()
    train5, valid, manifest = DATA["train_labels5"], DATA["train_valid"], DATA["manifest"]
    lt = np.asarray(manifest.selected_indices, dtype=np.int64)
    seeds = SEEDS[: int(args.max_seeds)]
    rows = []
    for seed in seeds:
        split_data = {
            "balanced": split_by_combo(valid, train5, seed + 100, 0.8),
            "longtail": split_by_combo(lt, train5, seed + 200, 0.8),
        }
        for arm in ARMS:
            kind = "longtail" if "LabelIF4" in arm else "balanced"
            train_idx, early_idx = split_data[kind]
            arm_dir = OUTPUT_ROOT / f"seed_{seed}" / arm.replace("-", "_")
            try:
                print(f"START seed={seed} arm={arm} train={len(train_idx)} early={len(early_idx)}", flush=True)
                row = train_arm(arm, seed, train_idx, early_idx, train5, device, args, arm_dir)
                rows.append(row)
                print(f"DONE seed={seed} arm={arm} early={row['early']['macro_aupr']:.6f} test={row['test']['macro_aupr']:.6f}", flush=True)
            except Exception as exc:
                rows.append({"arm": arm, "seed": int(seed), "status": "failed", "error": repr(exc)})
                print(f"FAILED seed={seed} arm={arm}: {exc!r}", flush=True)
    summary = {
        "status": "completed",
        "decision": "admit_frequency_lt_analysis",
        "device": str(device),
        "class_names": list(CLASS_NAMES),
        "arms": list(ARMS),
        "seeds": [int(s) for s in seeds],
        "backbone": {"name": "dfms_iq", "variant": "B0_DelayedFusionMultiScale", "width": 16, "feature_dim": 128, "dropout": 0.2, "num_classes": 5},
        "losses": {"focal": {"gamma_pos": 0.0, "gamma_neg": 3.0}, "db": "DistributionBalancedLoss(default_parameters)"},
        "optimizer": {"name": "AdamW", "backbone_lr": 5e-5, "classifier_lr": 5e-4, "weight_decay": 1e-4, "gradient_clip": 5.0},
        "training_budget": {"draws_per_epoch": int(args.draws_per_epoch), "equalized_across_arms": True, "batch_size": int(args.batch_size), "max_epochs": int(args.epochs), "patience": int(args.patience)},
        "protocol": {"power_longtail": False, "power_view_dirs_used": [], "sawfm_positive_samples_removed": True, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False},
        "label_if4_target_counts": DATA["target"].astype(int).tolist(),
        "label_if4_achieved_counts": train5[lt].sum(axis=0).astype(int).tolist(),
        "source_audit": DATA["source"],
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "run_receipt.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
    print(f"RECEIPT={OUTPUT_ROOT / 'run_receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
