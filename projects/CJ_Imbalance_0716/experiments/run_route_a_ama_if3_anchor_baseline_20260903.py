"""Development baseline on the corrected AMA six-class IF3 dataset.

The script keeps the new six-class MAT contract separate from the historical
nine-class loader.  It creates one deterministic 80/20 split from the train
file, trains ordinary BCE on the IF3 split, and compares it with an Anchor
condition implemented as uniform-combination sampling from the same fit
indices.  The supplied test MAT is development-test only; no locked artifact
is resolved or opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_c1_baseline import initialize_f2, metrics  # noqa: E402

DATA_ROOT = Path(
    r"E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA"
)
TRAIN_PATH = DATA_ROOT / "closedset_6class_train.mat"
TEST_PATH = DATA_ROOT / "closedset_6class_test.mat"
INIT = Path(
    r"E:\LJQ_code\CJ_Imbalance_0716\frozen\20260731_route_a_visibility_balanced_fusion_v2\seed_2039\backbone\best_model.pth"
)
OUT = ROOT / "outputs" / "route_a_ama_if3_anchor_baseline_20260903"
CLASS_NAMES = ("FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mat(path: Path):
    with h5py.File(path, "r") as f:
        labels = np.asarray(f["label_selected"], dtype=np.uint8).T
        stored = np.asarray(f["iq_complex"])
        receiver_seed = np.asarray(f["receiver_noise_seed"]).reshape(-1)
    if stored.dtype.fields is None or not {"real", "imag"}.issubset(stored.dtype.fields):
        raise TypeError(f"{path} iq_complex is not a real/imag compound dataset")
    signal = np.stack(
        [np.asarray(stored["real"], dtype=np.float32).T,
         np.asarray(stored["imag"], dtype=np.float32).T],
        axis=1,
    )
    rms = np.sqrt(np.mean(signal[:, 0] ** 2 + signal[:, 1] ** 2, axis=1))
    signal = signal / np.sqrt(rms[:, None, None] ** 2 + 1e-12)
    if labels.shape != (signal.shape[0], 6):
        raise ValueError(f"label shape {labels.shape} incompatible with {signal.shape}")
    return np.ascontiguousarray(signal), labels.astype(np.float32), receiver_seed


class ArrayDataset(Dataset):
    def __init__(self, signal: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.signal = signal
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return torch.from_numpy(self.signal[j]), torch.from_numpy(self.labels[j])


def combo_codes(labels: np.ndarray) -> np.ndarray:
    return labels.astype(np.uint8) @ (1 << np.arange(6, dtype=np.uint8))


def split_by_combo(labels: np.ndarray, seed: int, val_fraction: float = 0.2):
    rng = np.random.default_rng(seed)
    codes = combo_codes(labels)
    fit, val = [], []
    for code in sorted(np.unique(codes).tolist()):
        group = np.flatnonzero(codes == code)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction)))
        val.extend(group[:n_val].tolist())
        fit.extend(group[n_val:].tolist())
    return np.asarray(sorted(fit), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)


def distribution(labels: np.ndarray, indices: np.ndarray) -> dict:
    y = labels[indices].astype(np.uint8)
    lc = y.sum(axis=0).astype(int)
    cc = np.bincount(combo_codes(y), minlength=64)[1:]
    active = cc[cc > 0]
    return {
        "n": int(len(indices)),
        "label_counts": lc.tolist(),
        "effective_if": float(lc.max() / lc.min()),
        "unique_combinations": int(np.count_nonzero(cc)),
        "combo_min": int(active.min()),
        "combo_max": int(active.max()),
        "effective_cf": float(active.max() / active.min()),
    }


def train_one(condition, seed, signal, labels, fit_idx, val_idx, test_signal, test_labels,
              device, epochs=8, batch_size=32):
    set_seed(seed)
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    init_meta = initialize_f2(model, INIT)
    fit_ds = ArrayDataset(signal, labels, fit_idx)
    val_ds = ArrayDataset(signal, labels, val_idx)
    if condition == "anchor":
        codes = combo_codes(labels[fit_idx])
        counts = {int(c): int((codes == c).sum()) for c in np.unique(codes)}
        weights = np.asarray([1.0 / counts[int(c)] for c in codes], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(fit_idx), replacement=True)
        loader = DataLoader(fit_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=0, pin_memory=True)
    else:
        loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_aupr, best_epoch = -1.0, 0
    history = []
    started = time.time()
    out_dir = OUT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{condition}__seed_{seed}.pth"

    def predict(ds):
        loader_ = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        ys, logits = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader_:
                ys.append(y.numpy())
                logits.append(model(x.to(device, non_blocking=True)).cpu().numpy())
        return np.concatenate(ys).astype(np.uint8), np.concatenate(logits).astype(np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        vy, vl = predict(val_ds)
        vm = metrics(vy, vl)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": vm})
        print(json.dumps({"condition": condition, "seed": seed, "epoch": epoch,
                          "val_macro_aupr": vm["macro_aupr"],
                          "val_macro_f1": vm["macro_f1"],
                          "val_combo": vm["combination_exact_accuracy"]}, ensure_ascii=False), flush=True)
        if vm["macro_aupr"] > best_aupr:
            best_aupr, best_epoch = vm["macro_aupr"], epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation": vm}, checkpoint)

    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    ty, tl = predict(ArrayDataset(test_signal, test_labels,
                                  np.arange(test_signal.shape[0], dtype=np.int64)))
    result = {
        "schema": "route-a-ama-if3-anchor-baseline-v1",
        "status": "completed",
        "condition": condition,
        "seed": int(seed),
        "classes": list(CLASS_NAMES),
        "backbone": "F2_DFMS_Full",
        "interface": "complex_1d_true [B,2,3000]",
        "loss": "BCEWithLogitsLoss",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "epochs": epochs,
        "gradient_clip_norm": 5.0,
        "initialization": init_meta,
        "fit_distribution": distribution(labels, fit_idx),
        "validation_distribution": distribution(labels, val_idx),
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1]["validation"],
        "test": metrics(ty, tl),
        "history": history,
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": float(time.time() - started),
        "locked_test_used": False,
        "test_files_opened": True,
        "frozen_e5_modified": False,
    }
    (out_dir / f"{condition}__seed_{seed}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(f"missing dataset under {DATA_ROOT}")
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    train_signal, train_labels, train_receiver = load_mat(TRAIN_PATH)
    test_signal, test_labels, test_receiver = load_mat(TEST_PATH)
    fit_idx, val_idx = split_by_combo(train_labels, seed=20260903)
    if set(train_receiver[fit_idx].tolist()) & set(train_receiver[val_idx].tolist()):
        raise AssertionError("fit/validation receiver identity overlap")
    rows = []
    for condition in ("if3", "anchor"):
        for seed in args.seeds:
            rows.append(train_one(condition, seed, train_signal, train_labels,
                                  fit_idx, val_idx, test_signal, test_labels,
                                  device, epochs=args.epochs))
    summary = {
        "schema": "route-a-ama-if3-anchor-baseline-summary-v1",
        "status": "completed",
        "data_root": str(DATA_ROOT),
        "train_sha256": sha256(TRAIN_PATH),
        "test_sha256": sha256(TEST_PATH),
        "train_samples": int(train_signal.shape[0]),
        "test_samples": int(test_signal.shape[0]),
        "split_seed": 20260903,
        "fit_indices": int(len(fit_idx)),
        "validation_indices": int(len(val_idx)),
        "fit_distribution": distribution(train_labels, fit_idx),
        "validation_distribution": distribution(train_labels, val_idx),
        "conditions": ["if3", "anchor"],
        "seeds": [int(x) for x in args.seeds],
        "runs": rows,
        "locked_test_used": False,
        "frozen_e5_modified": False,
        "evidence_boundary": "development_train_validation_and_development_test_only",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "terminal_receipt.json").write_text(json.dumps({
        "status": "completed", "summary": str((OUT / "summary.json").resolve()),
        "locked_test_used": False, "frozen_e5_modified": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": str((OUT / "summary.json").resolve()),
                      "runs": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
