"""Validation-only threshold and probability calibration for AMA IF3 runs."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import (  # noqa: E402
    CLASS_NAMES, DATA_ROOT, INIT, OUT, ArrayDataset, load_mat, split_by_combo,
    sha256, set_seed,
)
from experiments.run_route_a_c1_baseline import initialize_f2  # noqa: E402

CAL_OUT = OUT / "calibration"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z.astype(np.float64), -40, 40)))


def ece_binary(y, p, bins=15):
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def evaluate(y, logits, thresholds=None, temperature=1.0, bias=None):
    z = logits.astype(np.float64) / float(temperature)
    if bias is not None:
        z = z + np.asarray(bias, dtype=np.float64)[None, :]
    p = sigmoid(z)
    th = np.full(y.shape[1], 0.5, dtype=np.float64) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    pred = (p >= th[None, :]).astype(np.uint8)
    per = {}
    aupr, auroc, briers, eces = [], [], [], []
    for i, name in enumerate(CLASS_NAMES):
        truth = y[:, i].astype(np.uint8)
        pos, neg = truth == 1, truth == 0
        a = float(average_precision_score(truth, p[:, i]))
        r = float(roc_auc_score(truth, p[:, i])) if len(np.unique(truth)) == 2 else float("nan")
        per[name] = {
            "aupr": a,
            "auroc": r,
            "recall": float(pred[pos, i].mean()) if pos.any() else float("nan"),
            "negative_fpr": float(pred[neg, i].mean()) if neg.any() else float("nan"),
            "f1": float(f1_score(truth, pred[:, i], zero_division=0)),
            "positive_count": int(pos.sum()),
            "ece": ece_binary(truth, p[:, i]),
            "brier": float(np.mean((p[:, i] - truth) ** 2)),
        }
        aupr.append(a); auroc.append(r); briers.append(per[name]["brier"]); eces.append(per[name]["ece"])
    weights = (2 ** np.arange(y.shape[1], dtype=np.int64))[None, :]
    true_code = (y.astype(np.int64) * weights).sum(axis=1)
    pred_code = (pred.astype(np.int64) * weights).sum(axis=1)
    return {
        "macro_aupr": float(np.nanmean(aupr)),
        "macro_auroc": float(np.nanmean(auroc)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "hamming_loss": float(np.mean(pred != y)),
        "combination_exact_accuracy": float(np.mean(true_code == pred_code)),
        "ece_macro": float(np.mean(eces)),
        "brier_macro": float(np.mean(briers)),
        "per_label": per,
    }


def bce_loss(y, logits):
    z = np.clip(logits, -40, 40)
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(y, logits):
    fun = lambda logt: bce_loss(y, logits / math.exp(float(logt)))
    opt = minimize_scalar(fun, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-5})
    return float(math.exp(float(opt.x)))


def fit_bias(y, logits, temperature=1.0):
    out = []
    base = logits / float(temperature)
    for i in range(y.shape[1]):
        fun = lambda b: bce_loss(y[:, i:i+1], (base[:, i:i+1] + float(b)))
        opt = minimize_scalar(fun, bounds=(-8.0, 8.0), method="bounded", options={"xatol": 1e-5})
        out.append(float(opt.x))
    return np.asarray(out, dtype=np.float64)


def best_f1_thresholds(y, logits):
    p = sigmoid(logits)
    out = []
    for i in range(y.shape[1]):
        candidates = np.unique(np.concatenate(([0.0], p[:, i], [1.0])))
        scores = np.asarray([f1_score(y[:, i], (p[:, i] >= t).astype(np.uint8), zero_division=0) for t in candidates])
        best = np.flatnonzero(scores == scores.max())
        # Prefer the candidate closest to 0.5 in ties to avoid extreme thresholds.
        j = best[np.argmin(np.abs(candidates[best] - 0.5))]
        out.append(float(candidates[j]))
    return np.asarray(out, dtype=np.float64)


def coordinate_combo_thresholds(y, logits, passes=2):
    p = sigmoid(logits)
    thresholds = np.full(y.shape[1], 0.5, dtype=np.float64)
    candidates = [np.unique(np.concatenate(([0.0], p[:, i], [1.0]))) for i in range(y.shape[1])]
    weights = (2 ** np.arange(y.shape[1], dtype=np.int64))[None, :]
    true_code = (y.astype(np.int64) * weights).sum(axis=1)
    for _ in range(passes):
        improved = False
        for i in range(y.shape[1]):
            best_score = -1.0; best_t = thresholds[i]
            for t in candidates[i]:
                trial = thresholds.copy(); trial[i] = t
                pred = (p >= trial[None, :]).astype(np.int64)
                score = float(np.mean((pred * weights).sum(axis=1) == true_code))
                if score > best_score or (score == best_score and abs(t - 0.5) < abs(best_t - 0.5)):
                    best_score, best_t = score, float(t)
            if best_t != thresholds[i]:
                improved = True
            thresholds[i] = best_t
        if not improved:
            break
    return thresholds


def predict(model, signal, labels, indices, device):
    ds = ArrayDataset(signal, labels, indices)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    ys, ls = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            ys.append(y.numpy())
            ls.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(ys).astype(np.uint8), np.concatenate(ls).astype(np.float32)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for the governed experiment")
    train_signal, train_labels, _ = load_mat(DATA_ROOT / "closedset_6class_train.mat")
    test_signal, test_labels, _ = load_mat(DATA_ROOT / "closedset_6class_test.mat")
    fit_idx, val_idx = split_by_combo(train_labels, seed=20260903)
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for condition in ("if3", "anchor"):
        for seed in (2039, 2040, 2041):
            run_json = OUT / condition / f"{condition}__seed_{seed}.json"
            stored = json.loads(run_json.read_text(encoding="utf-8"))
            set_seed(seed)
            model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
            init_meta = initialize_f2(model, INIT)
            ck = OUT / condition / f"{condition}__seed_{seed}.pth"
            payload = torch.load(str(ck), map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            vy, vl = predict(model, train_signal, train_labels, val_idx, device)
            ty, tl = predict(model, test_signal, test_labels, np.arange(test_signal.shape[0], dtype=np.int64), device)
            base = evaluate(vy, vl)
            # Reproducibility gate against the saved validation receipt.
            for key in ("macro_aupr", "macro_f1", "combination_exact_accuracy", "hamming_loss"):
                if abs(float(base[key]) - float(stored["validation"][key])) > 2e-5:
                    raise RuntimeError(f"replay mismatch {condition}/{seed}/{key}: {base[key]} vs {stored['validation'][key]}")
            f1_th = best_f1_thresholds(vy, vl)
            combo_th = coordinate_combo_thresholds(vy, vl)
            temp = fit_temperature(vy, vl)
            bias = fit_bias(vy, vl, 1.0)
            temp_bias = fit_bias(vy, vl, temp)
            methods = {
                "fixed_0.5": {"thresholds": np.full(6, 0.5), "temperature": 1.0, "bias": np.zeros(6)},
                "per_class_f1_threshold": {"thresholds": f1_th, "temperature": 1.0, "bias": np.zeros(6)},
                "combo_coordinate_threshold": {"thresholds": combo_th, "temperature": 1.0, "bias": np.zeros(6)},
                "temperature_scaling": {"thresholds": np.full(6, 0.5), "temperature": temp, "bias": np.zeros(6)},
                "logit_bias_bce": {"thresholds": np.full(6, 0.5), "temperature": 1.0, "bias": bias},
                "temperature_then_logit_bias_bce": {"thresholds": np.full(6, 0.5), "temperature": temp, "bias": temp_bias},
            }
            artifact = {"condition": condition, "seed": seed, "validation_targets": vy, "validation_logits": vl,
                        "test_targets": ty, "test_logits": tl, "parameters": {}}
            row = {"condition": condition, "seed": int(seed), "initialization": init_meta, "baseline_replay": base, "methods": {}}
            for name, pars in methods.items():
                row["methods"][name] = {
                    "parameters": {"thresholds": np.asarray(pars["thresholds"]).tolist(), "temperature": float(pars["temperature"]), "bias": np.asarray(pars["bias"]).tolist()},
                    "validation": evaluate(vy, vl, **pars),
                    "test": evaluate(ty, tl, **pars),
                }
                artifact["parameters"][name] = row["methods"][name]["parameters"]
            np.savez_compressed(CAL_OUT / f"{condition}__seed_{seed}_logits.npz", **artifact)
            (CAL_OUT / f"{condition}__seed_{seed}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            all_rows.append(row)
            print(json.dumps({"condition": condition, "seed": seed, "base_val_f1": base["macro_f1"],
                              "f1_threshold_val_f1": row["methods"]["per_class_f1_threshold"]["validation"]["macro_f1"],
                              "combo_threshold_val_combo": row["methods"]["combo_coordinate_threshold"]["validation"]["combination_exact_accuracy"],
                              "temperature": temp}, ensure_ascii=False), flush=True)
    summary = {"schema": "route-a-ama-if3-anchor-calibration-v1", "status": "completed",
               "data_root": str(DATA_ROOT), "train_sha256": sha256(DATA_ROOT / "closedset_6class_train.mat"),
               "test_sha256": sha256(DATA_ROOT / "closedset_6class_test.mat"), "split_seed": 20260903,
               "fit_indices": int(len(fit_idx)), "validation_indices": int(len(val_idx)),
               "methods": ["fixed_0.5", "per_class_f1_threshold", "combo_coordinate_threshold", "temperature_scaling", "logit_bias_bce", "temperature_then_logit_bias_bce"],
               "runs": all_rows, "locked_test_used": False, "frozen_e5_modified": False,
               "evidence_boundary": "validation_fitted; development_test_once_with_frozen_parameters"}
    (CAL_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": str((CAL_OUT / "summary.json").resolve()), "runs": len(all_rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
