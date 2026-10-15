"""P3j: raw-I/Q F2 plus a zero-initialized bounded phase residual.

The raw I/Q path remains the main classifier.  A small phase-increment
residual is added with a fixed bound 0.1*tanh(scale), so this screen tests
complementarity without replacing Cartesian evidence or introducing an expert.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, NineClassDataset, metrics, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, find_file, profile_cost, receiver_seeds, sha256, write_json
from experiments.run_route_a_p3_dual_axis_longtail import set_seed

VARIANTS = ("F2_raw_control", "F2_bounded_phase_residual")
SEEDS = (2039, 2040, 2041)


def phase_increment(x):
    real, imag = x[:, 0], x[:, 1]
    pr = torch.cat([real[:, :1], real[:, :-1]], dim=1)
    pi = torch.cat([imag[:, :1], imag[:, :-1]], dim=1)
    dot = real * pr + imag * pi
    cross = imag * pr - real * pi
    denom = torch.sqrt((real.square() + imag.square()) * (pr.square() + pi.square()) + 1e-8)
    out = torch.stack([dot / denom, cross / denom], dim=1)
    out[:, :, 0] = 0.0
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class PhaseResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(4),
            nn.Conv1d(8, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.readout = nn.Linear(8, 6)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = self.net(phase_increment(x)).squeeze(-1)
        return self.readout(h)


class ResidualModel(nn.Module):
    def __init__(self, enabled):
        super().__init__()
        self.base = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
        self.residual = PhaseResidual()
        self.enabled = bool(enabled)

    def forward(self, x):
        z = self.base(x)
        if self.enabled:
            z = z + 0.1 * torch.tanh(self.residual.scale) * self.residual(x)
        return z


def load_full(model, checkpoint):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    missing, unexpected = model.base.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"F2 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "checkpoint_epoch": payload.get("epoch"), "classifier_loaded": True}


def predict(model, path, indices, device, batch):
    ds = NineClassDataset(path, indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    ys, zs = [], []
    try:
        model.eval()
        with torch.no_grad():
            for b in dl:
                ys.append(b["label"].numpy())
                zs.append(model(b["signal"].to(device, non_blocking=True)).cpu().numpy())
    finally:
        ds.close()
    return np.concatenate(ys).astype(np.uint8), np.concatenate(zs).astype(np.float32)


def run_one(args, name, seed, paths, fit, val, power_dev, out):
    set_seed(seed)
    model = ResidualModel(name == "F2_bounded_phase_residual").to(args.device)
    init = load_full(model, args.f2_checkpoint)
    ds = NineClassDataset(paths[30], fit, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist, best, best_ep = [], -1.0, 0
    started = time.time()
    try:
        for ep in range(1, args.epochs + 1):
            model.train(); losses = []
            for b in dl:
                x = b["signal"].to(args.device, non_blocking=True)
                y = b["label"].to(args.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                losses.append(float(loss.detach().cpu()))
            yt, zt = predict(model, paths[30], val, args.device, args.eval_batch_size)
            vm = metrics(yt, zt)
            scale = float((0.1 * torch.tanh(model.residual.scale)).detach().cpu())
            hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "bounded_residual_scale": scale, "validation": vm})
            if vm["macro_aupr"] > best:
                best, best_ep = vm["macro_aupr"], ep
                torch.save(model.state_dict(), out / f"{name}__seed_{seed}.pth")
    finally:
        ds.close()
    model.load_state_dict(torch.load(str(out / f"{name}__seed_{seed}.pth"), map_location=args.device, weights_only=True))
    result = {"variant": name, "seed": seed, "history": hist, "best_epoch": best_ep, "validation": hist[best_ep - 1]["validation"], "power_dev": {}, "total_parameters": int(sum(p.numel() for p in model.parameters())), "cost_profile": profile_cost(model.base, args.device), "initialization": init, "locked_test_used": False, "elapsed_seconds": time.time() - started}
    for p, (path, idx) in power_dev.items():
        y, z = predict(model, path, idx, args.device, args.eval_batch_size)
        result["power_dev"][str(p)] = metrics(y, z)
    result["checkpoint_sha256"] = sha256(out / f"{name}__seed_{seed}.pth")
    write_json(out / f"{name}__seed_{seed}.json", result)
    print(json.dumps({"variant": name, "seed": seed, "val_combo": result["validation"]["combination_exact_accuracy"], "val_macro_aupr": result["validation"]["macro_aupr"]}, ensure_ascii=False), flush=True)
    return result


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("P3j requires CUDA")
    args.device = device; args.output_root.mkdir(parents=True, exist_ok=True)
    paths = {p: find_file(args.data_root, p, "train") for p in POWERS_DB}
    base = paths[30]; pure = pure_six_indices(base); perm = np.random.default_rng(args.split_seed).permutation(pure)
    fit = np.sort(perm[:args.fit_samples]); val = np.sort(perm[args.fit_samples:args.fit_samples + args.val_samples])
    used = set(receiver_seeds(base, np.sort(np.concatenate([fit, val])))); power_dev = {}
    for p in POWERS_DB:
        idx = pure_six_indices(paths[p]); rs = np.asarray(receiver_seeds(paths[p], idx), dtype=np.int64); candidates = idx[~np.isin(rs, np.fromiter(used, dtype=np.int64))]
        power_dev[p] = (paths[p], np.sort(np.random.default_rng(args.split_seed + p).choice(candidates, size=min(args.power_dev_samples, len(candidates)), replace=False)))
    sample_id_sha256 = __import__("experiments.run_route_a_p3_dual_axis_longtail", fromlist=["sample_id_sha256"]).sample_id_sha256
    role = {"schema": "route-a-p3j-bounded-phase-residual-role-manifest-v1", "selected_classes": list(CLASS_NAMES), "backbone": "F2_DFMS_Full/dfms_iq", "input_main": "raw normalized I/Q", "residual": "phase increment cos/sin, bound 0.1*tanh(scale), scale initialized 0", "fit": {"power_db": 30, "count": int(len(fit)), "sample_indices_sha256": sample_id_sha256(fit), "receiver_noise_seed": sorted(set(receiver_seeds(base, fit)))}, "validation": {"power_db": 30, "count": int(len(val)), "sample_indices_sha256": sample_id_sha256(val), "receiver_noise_seed": sorted(set(receiver_seeds(base, val)))}, "power_development": {"source_role": "train_only_unseen_identity", "count_by_power": {str(p): int(len(i)) for p, (_, i) in power_dev.items()}}, "test": {"used": False, "locked_test_used": False}}
    write_json(args.output_root / "role_manifest.json", role)
    rows = []
    for seed in args.seeds:
        for name in VARIANTS:
            out = args.output_root / name; out.mkdir(parents=True, exist_ok=True)
            rows.append(run_one(args, name, int(seed), paths, fit, val, power_dev, out))
    summary = {"schema": "route-a-p3j-bounded-phase-residual-summary-v1", "status": "completed", "locked_test_used": False, "protocol": {"fit_power_db": 30, "fit_samples": int(len(fit)), "val_samples": int(len(val)), "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "residual_bound": 0.1, "threshold": 0.5, "seeds": list(args.seeds)}, "variants": {}}
    for name in VARIANTS:
        rr = [r for r in rows if r["variant"] == name]
        summary["variants"][name] = {"parameters": rr[0]["total_parameters"], "val_macro_aupr_mean": float(np.mean([r["validation"]["macro_aupr"] for r in rr])), "val_macro_f1_mean": float(np.mean([r["validation"]["macro_f1"] for r in rr])), "val_combo_mean": float(np.mean([r["validation"]["combination_exact_accuracy"] for r in rr])), "power_dev_macro_aupr_mean_by_power": {str(p): float(np.mean([r["power_dev"][str(p)]["macro_aupr"] for r in rr])) for p in POWERS_DB}, "power_dev_combo_mean_by_power": {str(p): float(np.mean([r["power_dev"][str(p)]["combination_exact_accuracy"] for r in rr])) for p in POWERS_DB}}
    write_json(args.output_root / "p3j_bounded_phase_residual_summary.json", summary)
    write_json(args.output_root / "terminal_receipt.json", {"status": "completed", "summary": str((args.output_root / "p3j_bounded_phase_residual_summary.json").resolve()), "locked_test_used": False, "test_files_opened": False, "script_sha256": sha256(Path(__file__))})
    print(json.dumps({"status": "completed", "summary": str(args.output_root / "p3j_bounded_phase_residual_summary.json")}, ensure_ascii=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_p3j_bounded_phase_residual_20260905")
    p.add_argument("--f2-checkpoint", type=Path, default=PROJECT_ROOT / "outputs/route_a_c1_baseline_full_20260826/best_model.pth")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); p.add_argument("--split-seed", type=int, default=260905)
    p.add_argument("--fit-samples", type=int, default=4096); p.add_argument("--val-samples", type=int, default=2048); p.add_argument("--power-dev-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--eval-batch-size", type=int, default=32); p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
