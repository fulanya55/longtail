"""P3l: fixed-margin anchor-safe pooled scalar calibration diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.models.backbone_adapter import build_backbone
from experiments.audit_route_a_frozen_f2_feature_separability import extract
from experiments.run_route_a_c1_baseline import CLASS_NAMES, metrics, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, find_file, receiver_seeds, sha256, write_json
from experiments.run_route_a_p3k_pooled_logit_calibration import calibrate, fit_calibrator


def load_full(model, checkpoint):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "checkpoint_epoch": payload.get("epoch"), "classifier_loaded": True}


def logit(prob):
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("P3l requires CUDA extraction")
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = {p: find_file(args.data_root, p, "train") for p in POWERS_DB}
    base = paths[30]; pure = pure_six_indices(base); perm = np.random.default_rng(args.split_seed).permutation(pure)
    fit = np.sort(perm[:args.fit_samples]); val = np.sort(perm[args.fit_samples:args.fit_samples + args.val_samples])
    used = set(receiver_seeds(base, np.sort(np.concatenate([fit, val])))); dev = {}
    for p in POWERS_DB:
        idx = pure_six_indices(paths[p]); rs = np.asarray(receiver_seeds(paths[p], idx), dtype=np.int64); cand = idx[~np.isin(rs, np.fromiter(used, dtype=np.int64))]
        dev[p] = np.sort(np.random.default_rng(args.split_seed + p).choice(cand, size=min(args.power_dev_samples, len(cand)), replace=False))
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device); init = load_full(model, args.f2_checkpoint)
    yfit, zfit, _ = extract(model, base, fit, device, args.batch_size)
    pooled_z, pooled_y = [], []
    for p in POWERS_DB:
        yp, zp, _ = extract(model, paths[p], fit, device, args.batch_size); pooled_z.append(zp); pooled_y.append(yp)
    probes = fit_calibrator(np.concatenate(pooled_z), np.concatenate(pooled_y))

    def eval_one(path, idx):
        y, z, _ = extract(model, path, idx, device, args.batch_size)
        cal_z = logit(calibrate(probes, z))
        gate = np.max(np.abs(z), axis=1, keepdims=True) < args.margin
        safe_z = np.where(gate, cal_z, z)
        return {"raw_f2": metrics(y, z), "pooled_calibrated": metrics(y, cal_z), "anchor_safe": metrics(y, safe_z), "gate_fraction": float(gate.mean())}

    records = {"30_validation": eval_one(base, val), "power_development": {str(p): eval_one(paths[p], dev[p]) for p in POWERS_DB}}
    role = {"schema": "route-a-p3l-anchor-safe-calibration-role-manifest-v1", "selected_classes": list(CLASS_NAMES), "backbone": "F2_DFMS_Full/dfms_iq", "calibrator": "pooled scalar probes from train-role 10--50 dB logits", "gate": {"rule": "apply residual only if max absolute raw logit < margin", "margin": args.margin}, "fit": {"power_db": 30, "count": int(len(fit)), "pooled_powers": list(POWERS_DB), "receiver_noise_seed": sorted(set(receiver_seeds(base, fit)))}, "validation": {"power_db": 30, "count": int(len(val)), "receiver_noise_seed": sorted(set(receiver_seeds(base, val)))}, "development": {"source_role": "train_only_unseen_identity", "count_by_power": {str(p): int(len(i)) for p, i in dev.items()}}, "test": {"used": False, "locked_test_used": False}}
    write_json(args.output_root / "role_manifest.json", role)
    summary = {"schema": "route-a-p3l-anchor-safe-calibration-summary-v1", "status": "completed_diagnostic", "locked_test_used": False, "protocol": {"fit_samples_per_power": int(len(fit)), "pooled_powers": list(POWERS_DB), "margin": args.margin, "calibrator_parameters": 12, "no_threshold_tuning": True}, "initialization": init, "records": records}
    write_json(args.output_root / "p3l_anchor_safe_calibration_summary.json", summary)
    write_json(args.output_root / "terminal_receipt.json", {"status": "completed", "summary": str((args.output_root / "p3l_anchor_safe_calibration_summary.json").resolve()), "locked_test_used": False, "test_files_opened": False, "script_sha256": sha256(Path(__file__))})
    print(json.dumps({"status": "completed", "summary": str(args.output_root / "p3l_anchor_safe_calibration_summary.json")}, ensure_ascii=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_p3l_anchor_safe_calibration_20260905")
    p.add_argument("--f2-checkpoint", type=Path, default=PROJECT_ROOT / "outputs/route_a_c1_baseline_full_20260826/best_model.pth")
    p.add_argument("--split-seed", type=int, default=260905); p.add_argument("--fit-samples", type=int, default=4096); p.add_argument("--val-samples", type=int, default=2048); p.add_argument("--power-dev-samples", type=int, default=256); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--margin", type=float, default=2.0); p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
