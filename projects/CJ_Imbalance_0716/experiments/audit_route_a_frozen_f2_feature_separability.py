"""Frozen F2 feature separability audit across power views.

This is a development diagnostic only: the F2 backbone and classifier are not
changed, no thresholds are tuned, and only train/development roles are read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import (
    CLASS_NAMES,
    SELECTED_INDICES,
    NineClassDataset,
    metrics,
    pure_six_indices,
)
from experiments.run_route_a_nineclass_feature_sufficiency import (
    POWERS_DB,
    find_file,
    receiver_seeds,
    sha256,
    write_json,
)


def extract(model, path, indices, device, batch_size):
    ds = NineClassDataset(
        path, indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ys, logits, feats = [], [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in dl:
                z, f = model(
                    batch["signal"].to(device, non_blocking=True),
                    return_features=True,
                )
                ys.append(batch["label"].numpy())
                logits.append(z.cpu().numpy())
                feats.append(f.cpu().numpy())
    finally:
        ds.close()
    return (
        np.concatenate(ys).astype(np.uint8),
        np.concatenate(logits).astype(np.float32),
        np.concatenate(feats).astype(np.float32),
    )


def make_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=260906,
        ),
    )


def probability_metrics(y, prob):
    pred = (prob >= 0.5).astype(np.uint8)
    per = {}
    for j, name in enumerate(CLASS_NAMES):
        yy = y[:, j]
        pp = prob[:, j]
        per[name] = {
            "aupr": float(average_precision_score(yy, pp)),
            "auroc": float(roc_auc_score(yy, pp)),
            "recall_at_0.5": float(((pred[:, j] == 1) & (yy == 1)).sum() / max(1, (yy == 1).sum())),
            "fpr_at_0.5": float(((pred[:, j] == 1) & (yy == 0)).sum() / max(1, (yy == 0).sum())),
        }
    return {
        "macro_aupr": float(np.mean([v["aupr"] for v in per.values()])),
        "macro_auroc": float(np.mean([v["auroc"] for v in per.values()])),
        "combination_exact_accuracy": float(np.all(pred == y, axis=1).mean()),
        "per_label": per,
    }


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Frozen F2 audit requires CUDA")
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = {p: find_file(args.data_root, p, "train") for p in POWERS_DB}
    base = paths[30]
    pure = pure_six_indices(base)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(pure)
    fit = np.sort(perm[: args.fit_samples])
    used = set(receiver_seeds(base, np.sort(perm[: args.fit_samples + args.val_samples])))
    dev_indices = {}
    for power in POWERS_DB:
        pidx = pure_six_indices(paths[power])
        seeds = np.asarray(receiver_seeds(paths[power], pidx), dtype=np.int64)
        candidates = pidx[~np.isin(seeds, np.fromiter(used, dtype=np.int64))]
        dev_indices[power] = np.sort(
            np.random.default_rng(args.split_seed + power).choice(
                candidates, size=min(args.power_dev_samples, len(candidates)), replace=False
            )
        )

    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    payload = torch.load(str(args.f2_checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"full F2 checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    init = {
        "checkpoint": str(args.f2_checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.f2_checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "classifier_loaded": True,
    }
    y_fit, z_fit, f_fit = extract(model, base, fit, device, args.batch_size)
    records = {}
    for power in POWERS_DB:
        y, z, f = extract(model, paths[power], dev_indices[power], device, args.batch_size)
        frozen_prob = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        rec = {"f2_logits": metrics(y, z), "f2_feature_norm_mean": float(np.linalg.norm(f, axis=1).mean())}

        # Cross-power probe: fit only on 30 dB frozen features, then evaluate
        # each unseen-identity power view. This is not used as a candidate model.
        probe = [make_probe().fit(f_fit, y_fit[:, j]) for j in range(y_fit.shape[1])]
        p30 = np.column_stack([m.predict_proba(f)[:, 1] for m in probe])
        rec["linear_probe_fit30"] = probability_metrics(y, p30)

        # Within-power 4-fold linear probe estimates whether the frozen feature
        # space contains class information at that power at all.
        cv_prob = np.zeros_like(y, dtype=np.float64)
        for j in range(y.shape[1]):
            folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.split_seed + power + j)
            cv_prob[:, j] = cross_val_predict(
                make_probe(), f, y[:, j], cv=folds, method="predict_proba", n_jobs=1
            )[:, 1]
        rec["within_power_cv_probe"] = probability_metrics(y, cv_prob)
        records[str(power)] = rec

    role = {
        "schema": "route-a-frozen-f2-feature-separability-role-v1",
        "selected_classes": list(CLASS_NAMES),
        "backbone": "F2_DFMS_Full/dfms_iq",
        "fit": {"power_db": 30, "count": int(len(fit)), "receiver_noise_seed": sorted(set(receiver_seeds(base, fit)))},
        "development": {"source_role": "train_only_unseen_identity", "count_by_power": {str(p): int(len(i)) for p, i in dev_indices.items()}},
        "test": {"used": False, "locked_test_used": False},
    }
    write_json(args.output_root / "role_manifest.json", role)
    summary = {
        "schema": "route-a-frozen-f2-feature-separability-summary-v1",
        "status": "completed_diagnostic",
        "locked_test_used": False,
        "protocol": {
            "fit_power_db": 30,
            "fit_samples": int(len(fit)),
            "development_samples_per_power": args.power_dev_samples,
            "feature_dim": int(f_fit.shape[1]),
            "no_threshold_tuning": True,
            "full_checkpoint_classifier_loaded": True,
        },
        "initialization": init,
        "records_by_power": records,
    }
    write_json(args.output_root / "frozen_f2_feature_separability_summary.json", summary)
    write_json(
        args.output_root / "terminal_receipt.json",
        {
            "status": "completed",
            "summary": str((args.output_root / "frozen_f2_feature_separability_summary.json").resolve()),
            "locked_test_used": False,
            "test_files_opened": False,
            "script_sha256": sha256(Path(__file__)),
        },
    )
    print(json.dumps({"status": "completed", "summary": str(args.output_root / "frozen_f2_feature_separability_summary.json")}, ensure_ascii=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_frozen_f2_feature_separability_v2_20260905")
    p.add_argument("--f2-checkpoint", type=Path, default=PROJECT_ROOT / "outputs/route_a_c1_baseline_full_20260826/best_model.pth")
    p.add_argument("--split-seed", type=int, default=260905)
    p.add_argument("--fit-samples", type=int, default=4096)
    p.add_argument("--val-samples", type=int, default=2048)
    p.add_argument("--power-dev-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
