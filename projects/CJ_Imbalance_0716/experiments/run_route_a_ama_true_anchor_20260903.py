"""Build and train a no-replacement, independently generated AMA Anchor."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import (  # noqa: E402
    ArrayDataset, CLASS_NAMES, DATA_ROOT, INIT, load_mat, sha256, set_seed,
    distribution,
)
from experiments.run_route_a_c1_baseline import initialize_f2, metrics  # noqa: E402

POOL = Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类\full_power_10_30dB_v1\closedset_9class_train.mat")
OUT = ROOT / "outputs" / "route_a_ama_true_anchor_20260903"
SELECTED = np.asarray([0, 1, 2, 3, 5, 7], dtype=np.int64)
EXCLUDED = np.asarray([4, 6, 8], dtype=np.int64)


def build_manifest():
    with h5py.File(POOL, "r") as f:
        y9 = np.asarray(f["known_label"], dtype=np.uint8)
        receiver = np.asarray(f["receiver_noise_seed"]).reshape(-1)
    pure = (y9[SELECTED].sum(axis=0) > 0) & (y9[EXCLUDED].sum(axis=0) == 0)
    global_indices = np.flatnonzero(pure)
    y6 = y9[SELECTED][:, global_indices].T
    codes = y6 @ (1 << np.arange(6, dtype=np.uint8))
    rng = np.random.default_rng(20260903)
    # Total counts: 127 for masks 1..62 and 126 for mask 63.  This produces
    # perfectly equal label marginals because the one removed sample carries
    # all six labels.
    extra_val = {1, 62, 2, 61, 3, 60, 4, 59, 5, 58}
    fit, val = [], []
    combo_plan = {}
    for code in range(1, 64):
        available = global_indices[codes == code].copy()
        rng.shuffle(available)
        total = 126 if code == 63 else 127
        n_val = 26 if code in extra_val else 25
        chosen = available[:total]
        val.extend(chosen[:n_val].tolist())
        fit.extend(chosen[n_val:].tolist())
        combo_plan[str(code)] = {"total": total, "fit": total - n_val, "validation": n_val}
    fit = np.asarray(sorted(fit), dtype=np.int64)
    val = np.asarray(sorted(val), dtype=np.int64)
    if len(fit) != 6415 or len(val) != 1585 or np.intersect1d(fit, val).size:
        raise AssertionError("anchor manifest budget/overlap failure")
    manifest = {
        "schema": "route-a-ama-true-anchor-manifest-v1",
        "source": str(POOL), "source_sha256": sha256(POOL),
        "classes": list(CLASS_NAMES), "selected_nineclass_indices": SELECTED.tolist(),
        "excluded_nineclass_indices": EXCLUDED.tolist(), "selection_seed": 20260903,
        "fit_indices": fit.tolist(), "validation_indices": val.tolist(),
        "combo_plan": combo_plan, "replacement": False,
        "receiver_seed_intersection_fit_validation": int(len(set(receiver[fit].tolist()) & set(receiver[val].tolist()))),
        "locked_test_used": False,
    }
    return manifest


def load_selected(path, indices):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    with h5py.File(path, "r") as f:
        y = np.asarray(f["known_label"][SELECTED, :][:, sorted_idx], dtype=np.uint8).T
        stored = np.asarray(f["iq_complex"][:, sorted_idx])
        receiver = np.asarray(f["receiver_noise_seed"][:, sorted_idx]).reshape(-1)
    signal = np.stack([np.asarray(stored["real"], dtype=np.float32).T,
                       np.asarray(stored["imag"], dtype=np.float32).T], axis=1)
    rms = np.sqrt(np.mean(signal[:, 0] ** 2 + signal[:, 1] ** 2, axis=1))
    signal = signal / np.sqrt(rms[:, None, None] ** 2 + 1e-12)
    inverse = np.argsort(order)
    return np.ascontiguousarray(signal[inverse]), y[inverse].astype(np.float32), receiver[inverse]


def predict(model, ds, device):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    ys, ls = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            ys.append(y.numpy()); ls.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(ys).astype(np.uint8), np.concatenate(ls).astype(np.float32)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    all_indices = np.asarray(manifest["fit_indices"] + manifest["validation_indices"], dtype=np.int64)
    signal, labels, receiver = load_selected(POOL, all_indices)
    nfit = len(manifest["fit_indices"])
    fit_idx = np.arange(nfit, dtype=np.int64)
    val_idx = np.arange(nfit, len(all_indices), dtype=np.int64)
    # Manifest concatenation order is fit then validation; load_selected restores it.
    fit_dist = distribution(labels, fit_idx); val_dist = distribution(labels, val_idx)
    if fit_dist["effective_if"] != 1.0 or val_dist["effective_if"] != 1.0:
        raise AssertionError(f"anchor IF failure: {fit_dist}, {val_dist}")
    test_signal, test_labels, test_receiver = load_mat(DATA_ROOT / "closedset_6class_test.mat")
    if set(receiver.tolist()) & set(test_receiver.tolist()):
        raise AssertionError("anchor development/test receiver overlap")
    # Governed forward/backward/step smoke before the full runs.
    set_seed(2039)
    smoke = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    initialize_f2(smoke, INIT)
    opt = torch.optim.AdamW(smoke.parameters(), lr=1e-4, weight_decay=1e-4)
    x = torch.from_numpy(signal[:2]).to(device); y = torch.from_numpy(labels[:2]).to(device)
    loss = nn.BCEWithLogitsLoss()(smoke(x), y); loss.backward(); opt.step()
    smoke_receipt = {"forward_shape": list(smoke(x).shape), "loss": float(loss.detach().cpu()),
                     "backward": True, "optimizer_step": True, "device": str(device)}
    del smoke, opt, x, y
    rows = []
    for seed in (2039, 2040, 2041):
        set_seed(seed)
        model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
        init = initialize_f2(model, INIT)
        dl = DataLoader(ArrayDataset(signal, labels, fit_idx), batch_size=32, shuffle=True,
                        num_workers=0, pin_memory=True)
        val_ds = ArrayDataset(signal, labels, val_idx)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        best, best_epoch, history = -1.0, 0, []
        ck = OUT / f"true_anchor__seed_{seed}.pth"
        started = time.time()
        for epoch in range(1, 9):
            model.train(); losses = []
            for xb, yb in dl:
                xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True); l=criterion(model(xb),yb); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); losses.append(float(l.detach().cpu()))
            vy,vl=predict(model,val_ds,device); vm=metrics(vy,vl)
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":vm})
            print(json.dumps({"seed":seed,"epoch":epoch,"val_macro_aupr":vm["macro_aupr"],"val_macro_f1":vm["macro_f1"],"val_combo":vm["combination_exact_accuracy"]}),flush=True)
            if vm["macro_aupr"]>best:
                best=vm["macro_aupr"]; best_epoch=epoch; torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"validation":vm},ck)
        model.load_state_dict(torch.load(str(ck),map_location=device,weights_only=False)["model_state_dict"])
        ty,tl=predict(model,ArrayDataset(test_signal,test_labels,np.arange(len(test_labels))),device)
        row={"schema":"route-a-ama-true-anchor-v1","status":"completed","seed":seed,
             "classes":list(CLASS_NAMES),"backbone":"F2_DFMS_Full","loss":"BCEWithLogitsLoss",
             "optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch_size":32,"epochs":8,
             "initialization":init,"fit_distribution":fit_dist,"validation_distribution":val_dist,
             "best_epoch":best_epoch,"validation":history[best_epoch-1]["validation"],"test_on_if3_distribution":metrics(ty,tl),
             "history":history,"elapsed_seconds":float(time.time()-started),"checkpoint_sha256":sha256(ck),
             "locked_test_used":False,"frozen_e5_modified":False}
        (OUT/f"true_anchor__seed_{seed}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
        rows.append(row)
    summary={"schema":"route-a-ama-true-anchor-summary-v1","status":"completed","manifest":manifest,
             "smoke":smoke_receipt,"fit_distribution":fit_dist,"validation_distribution":val_dist,
             "test_distribution":"IF3 development-test, frozen and never used for model selection","runs":rows,
             "locked_test_used":False,"frozen_e5_modified":False,"evidence_boundary":"development-only"}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((OUT/"summary.json").resolve()),"locked_test_used":False,"frozen_e5_modified":False},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","summary":str((OUT/"summary.json").resolve()),"runs":len(rows)}),flush=True)


if __name__ == "__main__":
    main()
