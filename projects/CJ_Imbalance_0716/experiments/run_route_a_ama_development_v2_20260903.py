"""Fresh development-v2 confirmation for AMA IF3 versus true Anchor."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_c1_baseline import initialize_f2, metrics  # noqa: E402
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import (  # noqa: E402
    CLASS_NAMES, DATA_ROOT, INIT, sha256, set_seed,
)

SRC_A = Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类\full_power_10_30dB_v1\closedset_9class_train.mat")
SRC_B = Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类\full_power_10_30dB_v1\closedset_9class_test.mat")
OLD_ANCHOR = ROOT / "outputs" / "route_a_ama_true_anchor_20260903" / "manifest.json"
OUT = ROOT / "outputs" / "route_a_ama_development_v2_20260903"
SEL = np.asarray([0, 1, 2, 3, 5, 7], dtype=np.int64)
EXC = np.asarray([4, 6, 8], dtype=np.int64)
IF3_COUNTS = np.asarray(
    json.loads((DATA_ROOT / "audit_summary.json").read_text(encoding="utf-8"))["planner_record"]["counts"],
    dtype=np.int64,
)


class ArrayDataset(Dataset):
    def __init__(self, signal, labels):
        self.signal = signal
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, i):
        return torch.from_numpy(self.signal[i]), torch.from_numpy(self.labels[i])


def pool_meta(path):
    with h5py.File(path, "r") as f:
        y9 = np.asarray(f["known_label"], dtype=np.uint8)
        seeds = np.asarray(f["receiver_noise_seed"]).reshape(-1)
    pure = (y9[SEL].sum(axis=0) > 0) & (y9[EXC].sum(axis=0) == 0)
    indices = np.flatnonzero(pure)
    y6 = y9[SEL][:, indices].T
    codes = y6 @ (1 << np.arange(6, dtype=np.uint8))
    return indices, y6, codes, seeds


def distribution(labels):
    labels = labels.astype(np.uint8)
    lc = labels.sum(axis=0).astype(int)
    codes = labels @ (1 << np.arange(6, dtype=np.uint8))
    counts = np.bincount(codes.astype(int), minlength=64)[1:]
    active = counts[counts > 0]
    return {
        "n": int(len(labels)),
        "label_counts": lc.tolist(),
        "effective_if": float(lc.max() / lc.min()),
        "unique_combinations": int(np.count_nonzero(counts)),
        "combo_min": int(active.min()),
        "combo_max": int(active.max()),
        "effective_cf": float(active.max() / active.min()),
    }


def build_manifest():
    gi_a, y_a, c_a, seed_a = pool_meta(SRC_A)
    gi_b, y_b, c_b, seed_b = pool_meta(SRC_B)
    old = json.loads(OLD_ANCHOR.read_text(encoding="utf-8"))
    old_used = set(old["fit_indices"]) | set(old["validation_indices"])
    rng = np.random.default_rng(20260904)
    anchor, validation, test, if3 = [], [], [], []
    for code in range(1, 64):
        avail_a = np.asarray([x for x in gi_a[c_a == code] if int(x) not in old_used], dtype=np.int64)
        rng.shuffle(avail_a)
        n_anchor = 126 if code == 63 else 127
        need = n_anchor + 32 + 64
        if len(avail_a) < need:
            raise RuntimeError(f"Source-A role infeasible code={code}: {len(avail_a)} < {need}")
        anchor.extend(avail_a[:n_anchor].tolist())
        validation.extend(avail_a[n_anchor:n_anchor + 32].tolist())
        test.extend(avail_a[n_anchor + 32:need].tolist())

        avail_b = gi_b[c_b == code].copy()
        rng.shuffle(avail_b)
        n_if3 = int(IF3_COUNTS[code - 1])
        if len(avail_b) < n_if3:
            raise RuntimeError(f"Source-B IF3 infeasible code={code}: {len(avail_b)} < {n_if3}")
        if3.extend(avail_b[:n_if3].tolist())

    def seed_overlap(a, b):
        return int(len(set(seed_a[a].tolist()) & set(seed_a[b].tolist())))

    manifest = {
        "schema": "route-a-ama-development-v2-manifest-v1",
        "classes": list(CLASS_NAMES),
        "source_a_anchor_val_test": str(SRC_A),
        "source_b_if3_train": str(SRC_B),
        "source_a_sha256": sha256(SRC_A),
        "source_b_sha256": sha256(SRC_B),
        "selection_seed": 20260904,
        "excluded_previous_true_anchor_indices": int(len(old_used)),
        "anchor_indices": sorted(anchor),
        "validation_indices": sorted(validation),
        "development_test_indices": sorted(test),
        "if3_indices": sorted(if3),
        "replacement": False,
        "roles": {
            "anchor_fit_n": len(anchor),
            "validation_n": len(validation),
            "test_n": len(test),
            "if3_fit_n": len(if3),
        },
        "seed_overlap_anchor_validation": seed_overlap(anchor, validation),
        "seed_overlap_anchor_test": seed_overlap(anchor, test),
        "seed_overlap_validation_test": seed_overlap(validation, test),
        "seed_overlap_if3_with_source_a_roles": int(
            len(set(seed_b[if3].tolist()) & (set(seed_a[anchor].tolist()) | set(seed_a[validation].tolist()) | set(seed_a[test].tolist())))
        ),
    }
    manifest["source_a_role_distributions"] = {
        "anchor": distribution(y_a[np.searchsorted(gi_a, np.asarray(anchor))]),
        "validation": distribution(y_a[np.searchsorted(gi_a, np.asarray(validation))]),
        "test": distribution(y_a[np.searchsorted(gi_a, np.asarray(test))]),
    }
    manifest["if3_distribution"] = distribution(y_b[np.searchsorted(gi_b, np.asarray(if3))])
    return manifest


def load_selected(path, indices):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    with h5py.File(path, "r") as f:
        y = np.asarray(f["known_label"][SEL, :][:, sorted_indices], dtype=np.uint8).T
        stored = np.asarray(f["iq_complex"][:, sorted_indices])
        seeds = np.asarray(f["receiver_noise_seed"][:, sorted_indices]).reshape(-1)
    signal = np.stack(
        [np.asarray(stored["real"], dtype=np.float32).T,
         np.asarray(stored["imag"], dtype=np.float32).T], axis=1,
    )
    rms = np.sqrt(np.mean(signal[:, 0] ** 2 + signal[:, 1] ** 2, axis=1))
    signal = signal / np.sqrt(rms[:, None, None] ** 2 + 1e-12)
    inverse = np.argsort(order)
    return np.ascontiguousarray(signal[inverse]), y[inverse].astype(np.float32), seeds[inverse]


def predict(model, signal, labels, device):
    dl = DataLoader(ArrayDataset(signal, labels), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    ys, logits = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            ys.append(y.numpy())
            logits.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return metrics(np.concatenate(ys).astype(np.uint8), np.concatenate(logits).astype(np.float32))


def train_one(condition, seed, fit_signal, fit_labels, val_signal, val_labels, test_signal, test_labels, device):
    set_seed(seed)
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    init = initialize_f2(model, INIT)
    loader = DataLoader(ArrayDataset(fit_signal, fit_labels), batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    out_dir = OUT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{condition}__seed_{seed}.pth"
    best, best_epoch, history = -1.0, 0, []
    started = time.time()
    for epoch in range(1, 9):
        model.train(); losses = []
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        vm = predict(model, val_signal, val_labels, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": vm})
        print(json.dumps({"condition": condition, "seed": seed, "epoch": epoch, "val_macro_aupr": vm["macro_aupr"], "val_macro_f1": vm["macro_f1"], "val_combo": vm["combination_exact_accuracy"]}), flush=True)
        if vm["macro_aupr"] > best:
            best, best_epoch = vm["macro_aupr"], epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation": vm}, checkpoint)
    model.load_state_dict(torch.load(str(checkpoint), map_location=device, weights_only=False)["model_state_dict"])
    tm = predict(model, test_signal, test_labels, device)
    row = {"schema": "route-a-ama-development-v2-run-v1", "status": "completed", "condition": condition, "seed": int(seed), "classes": list(CLASS_NAMES), "backbone": "F2_DFMS_Full", "loss": "BCEWithLogitsLoss", "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 32, "epochs": 8, "gradient_clip_norm": 5.0, "initialization": init, "best_epoch": best_epoch, "validation": history[best_epoch - 1]["validation"], "development_test": tm, "history": history, "checkpoint_sha256": sha256(checkpoint), "elapsed_seconds": float(time.time() - started), "locked_test_used": False, "frozen_e5_modified": False}
    (out_dir / f"{condition}__seed_{seed}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    a_sig, a_lab, a_seed = load_selected(SRC_A, manifest["anchor_indices"])
    v_sig, v_lab, v_seed = load_selected(SRC_A, manifest["validation_indices"])
    t_sig, t_lab, t_seed = load_selected(SRC_A, manifest["development_test_indices"])
    i_sig, i_lab, i_seed = load_selected(SRC_B, manifest["if3_indices"])
    if set(a_seed.tolist()) & set(v_seed.tolist()) or set(a_seed.tolist()) & set(t_seed.tolist()) or set(v_seed.tolist()) & set(t_seed.tolist()) or set(i_seed.tolist()) & (set(a_seed.tolist()) | set(v_seed.tolist()) | set(t_seed.tolist())):
        raise AssertionError("development-v2 identity overlap")
    rows = []
    for condition, fs, fl in (("if3", i_sig, i_lab), ("anchor", a_sig, a_lab)):
        for seed in (2039, 2040, 2041, 2042, 2043):
            rows.append(train_one(condition, seed, fs, fl, v_sig, v_lab, t_sig, t_lab, device))
    summary = {"schema": "route-a-ama-development-v2-summary-v1", "status": "completed", "manifest": manifest, "validation_distribution": distribution(v_lab.astype(np.uint8)), "development_test_distribution": distribution(t_lab.astype(np.uint8)), "runs": rows, "locked_test_used": False, "frozen_e5_modified": False, "evidence_boundary": "development-v2-confirmation_only"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "terminal_receipt.json").write_text(json.dumps({"status": "completed", "summary": str((OUT / "summary.json").resolve()), "locked_test_used": False, "frozen_e5_modified": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": str((OUT / "summary.json").resolve()), "runs": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
