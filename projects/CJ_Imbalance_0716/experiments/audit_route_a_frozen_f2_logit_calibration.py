"""Diagnostic: can scalar logit calibration recover cross-power decisions?

The F2 model is frozen and the calibration probes are evaluated only with
within-power cross-validation on train-only development views. No final
threshold or model checkpoint is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.models.backbone_adapter import build_backbone
from experiments.audit_route_a_frozen_f2_feature_separability import extract, probability_metrics
from experiments.run_route_a_c1_baseline import CLASS_NAMES, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import (
    POWERS_DB,
    find_file,
    receiver_seeds,
    sha256,
    write_json,
)


def probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500, class_weight="balanced", solver="liblinear", random_state=260907
        ),
    )


def load_full(model, checkpoint):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "classifier_loaded": True,
    }


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("logit calibration audit requires CUDA")
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = {p: find_file(args.data_root, p, "train") for p in POWERS_DB}
    base = paths[30]
    pure = pure_six_indices(base)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(pure)
    fit = np.sort(perm[: args.fit_samples])
    used = set(receiver_seeds(base, np.sort(perm[: args.fit_samples + args.val_samples])))
    dev = {}
    for power in POWERS_DB:
        idx = pure_six_indices(paths[power])
        rs = np.asarray(receiver_seeds(paths[power], idx), dtype=np.int64)
        candidates = idx[~np.isin(rs, np.fromiter(used, dtype=np.int64))]
        dev[power] = np.sort(
            np.random.default_rng(args.split_seed + power).choice(
                candidates, size=min(args.power_dev_samples, len(candidates)), replace=False
            )
        )

    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    init = load_full(model, args.f2_checkpoint)
    yfit, zfit, _ = extract(model, base, fit, device, args.batch_size)
    fit30_probes = [probe().fit(zfit[:, j : j + 1], yfit[:, j]) for j in range(6)]
    records = {}
    for power in POWERS_DB:
        y, z, _ = extract(model, paths[power], dev[power], device, args.batch_size)
        raw = probability_metrics(y, 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))))
        p30 = np.column_stack(
            [m.predict_proba(z[:, j : j + 1])[:, 1] for j, m in enumerate(fit30_probes)]
        )
        cvp = np.zeros_like(z, dtype=np.float64)
        for j in range(6):
            folds = StratifiedKFold(
                n_splits=4, shuffle=True, random_state=args.split_seed + power + j
            )
            cvp[:, j] = cross_val_predict(
                probe(), z[:, j : j + 1], y[:, j], cv=folds, method="predict_proba", n_jobs=1
            )[:, 1]
        records[str(power)] = {
            "f2_raw": raw,
            "fit30_scalar_logit_calibration": probability_metrics(y, p30),
            "within_power_scalar_logit_cv_calibration": probability_metrics(y, cvp),
        }

    role = {
        "schema": "route-a-frozen-f2-logit-calibration-role-v1",
        "selected_classes": list(CLASS_NAMES),
        "backbone": "F2_DFMS_Full/dfms_iq",
        "fit": {"power_db": 30, "count": int(len(fit)), "receiver_noise_seed": sorted(set(receiver_seeds(base, fit)))},
        "development": {"source_role": "train_only_unseen_identity", "count_by_power": {str(p): int(len(i)) for p, i in dev.items()}},
        "calibration": "diagnostic only; scalar per-label probes; no final threshold tuning",
        "test": {"used": False, "locked_test_used": False},
    }
    write_json(args.output_root / "role_manifest.json", role)
    summary = {
        "schema": "route-a-frozen-f2-logit-calibration-summary-v1",
        "status": "completed_diagnostic",
        "locked_test_used": False,
        "protocol": {
            "fit_power_db": 30,
            "fit_samples": int(len(fit)),
            "development_samples_per_power": args.power_dev_samples,
            "no_final_threshold_tuning": True,
            "calibration_cv_folds": 4,
        },
        "initialization": init,
        "records_by_power": records,
    }
    write_json(args.output_root / "frozen_f2_logit_calibration_summary.json", summary)
    write_json(
        args.output_root / "terminal_receipt.json",
        {
            "status": "completed",
            "summary": str((args.output_root / "frozen_f2_logit_calibration_summary.json").resolve()),
            "locked_test_used": False,
            "test_files_opened": False,
            "script_sha256": sha256(Path(__file__)),
        },
    )
    print(json.dumps({"status": "completed", "summary": str(args.output_root / "frozen_f2_logit_calibration_summary.json")}, ensure_ascii=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_frozen_f2_logit_calibration_20260905")
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
