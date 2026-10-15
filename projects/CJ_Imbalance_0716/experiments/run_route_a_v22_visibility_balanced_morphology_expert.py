"""Five-seed v22 visibility-balanced morphology-expert validation on new identities."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write
from experiments.run_route_a_development_v16_isrj_width_probe import _load_signals
from experiments.run_route_a_development_v2_h5 import _sha256
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft

VFT, ISRJ = 4, 5


class MorphologyProbe(nn.Module):
    def __init__(self, dim: int = 268, hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            rows.append(model(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def _stratum_loss(logits: torch.Tensor, targets: torch.Tensor, weak: torch.Tensor) -> torch.Tensor:
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    parts = [(targets == 1) & (weak == 1), (targets == 1) & (weak == 0), targets == 0]
    return torch.stack([base[part].mean() for part in parts]).mean()


def _train(name, xfit, yfit, weakfit, xearly, yearly, weakearly, seed, device, args, root):
    _set_seed(seed)
    model = MorphologyProbe(xfit.shape[1], args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    strong = int(((yfit == 1) & (weakfit == 0)).sum()); weak_count = int(((yfit == 1) & (weakfit == 1)).sum())
    weak_weight = float(strong / weak_count) if name == "M_VBF" else 1.0
    generator = torch.Generator(); generator.manual_seed(seed + 902)
    best = {"loss": float("inf"), "epoch": 0, "state": None}; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(xfit), generator=generator).numpy(); losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            xb = torch.from_numpy(xfit[index]).to(device); yb = torch.from_numpy(yfit[index]).to(device); wb = torch.from_numpy(weakfit[index]).to(device)
            weights = torch.ones_like(yb); weights[(yb == 1) & (wb == 1)] = weak_weight
            optimizer.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * weights).mean()
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            early_loss = float(_stratum_loss(model(torch.from_numpy(xearly).to(device)), torch.from_numpy(yearly).to(device), torch.from_numpy(weakearly).to(device)).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "early_three_stratum_loss": early_loss})
        if early_loss < best["loss"] - 1e-10:
            best = {"loss": early_loss, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    output = root / name; output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "best_epoch": best["epoch"], "best_early_three_stratum_loss": best["loss"], "fit_weak_positive_weight": weak_weight, "architecture": {"input_dim": int(xfit.shape[1]), "hidden": args.hidden, "dropout": args.dropout}}, output / "best_model.pth")
    _write(output / "history.json", history)
    return model, best, weak_weight


def _metric(target: np.ndarray, score: np.ndarray) -> dict:
    return {"count": int(len(target)), "auroc": float(roc_auc_score(target, score)), "aupr": float(average_precision_score(target, score))}


def _stratum_metric(labels, weak, ids, score, stratum: str):
    positive = (labels[ids, VFT] == 1) & (weak[ids, VFT] if stratum == "weak" else ~weak[ids, VFT])
    negative = labels[ids, VFT] == 0
    if min(positive.sum(), negative.sum()) < 20:
        raise ValueError(f"v22 missing {stratum} evaluation stratum")
    target = np.r_[np.ones(positive.sum(), dtype=np.uint8), np.zeros(negative.sum(), dtype=np.uint8)]
    values = np.r_[score[positive], score[negative]]
    return {"positive_count": int(positive.sum()), "negative_count": int(negative.sum()), **_metric(target, values)}


def _load(args):
    audit = _read(args.audit); seal = _read(args.test_v3_manifest)
    if not audit.get("identity_audit_passed") or audit.get("test_files_opened"):
        raise ValueError("v22 audit is not a valid train-only identity audit")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("sealed test-v3 cannot be accessed")
    if _sha256(args.data_path).lower() != audit["sha256"].lower():
        raise ValueError("v22 source hash changed")
    roles_file = np.load(args.roles, allow_pickle=False); ids = {name: np.asarray(roles_file[name], dtype=np.int64) for name in ("fit", "early_stop", "evaluation")}
    labels = read_labels(args.data_path); params, _ = read_parameters(args.data_path)
    weak = compute_weak_component_mask(labels, _component_power(params), args.weak_threshold_db)
    with h5py.File(args.data_path, "r") as h5:
        receiver = np.asarray(h5["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    names = tuple(ids)
    if any(np.intersect1d(receiver[ids[a]], receiver[ids[b]]).size for i, a in enumerate(names) for b in names[i + 1 :]):
        raise ValueError("v22 role identity leakage")
    for name, index in ids.items():
        if not np.all(labels[index, ISRJ] == 1): raise ValueError(f"v22 {name} includes non-ISRJ samples")
        for active in ((labels[index, VFT] == 1) & weak[index, VFT], (labels[index, VFT] == 1) & ~weak[index, VFT], labels[index, VFT] == 0):
            if active.sum() < 20: raise ValueError(f"v22 {name} lacks a required stratum")
    return audit, labels, weak, receiver, ids


def _one_seed(args, features, labels, weak, receiver, ids, seed, device):
    root = args.output_root / f"seed_{seed}"; root.mkdir(parents=True, exist_ok=True)
    fit, early, evaluation = (ids[name] for name in ("fit", "early_stop", "evaluation"))
    mean = features[fit].mean(0).astype(np.float32); std = np.maximum(features[fit].std(0), 1e-5).astype(np.float32); x = ((features - mean) / std).astype(np.float32)
    y = labels[:, VFT].astype(np.float32); w = weak[:, VFT].astype(np.float32)
    variants = {}
    for name in ("M_BCE", "M_VBF"):
        model, best, weight = _train(name, x[fit], y[fit], w[fit], x[early], y[early], w[early], seed, device, args, root)
        scores = _predict(model, x, device, args.eval_batch_size)
        output = root / name
        metrics = {"overall": _metric(y[evaluation], scores[evaluation]), "weak_vft_vs_hard_negative": _stratum_metric(labels, weak, evaluation, scores[evaluation], "weak"), "strong_vft_vs_hard_negative": _stratum_metric(labels, weak, evaluation, scores[evaluation], "strong"), "best_epoch": best["epoch"], "best_early_three_stratum_loss": best["loss"], "fit_weak_positive_weight": weight}
        np.savez_compressed(output / "predictions.npz", sample_ids=evaluation, receiver_noise_seed=receiver[evaluation], targets=y[evaluation].astype(np.uint8), weak_vft=w[evaluation].astype(np.uint8), scores=scores[evaluation])
        _write(output / "metrics.json", metrics); variants[name] = metrics
    np.savez_compressed(root / "normalization.npz", mean=mean, std=std)
    return {"seed": seed, "feature_dim": int(x.shape[1]), "parameter_count": int(sum(p.numel() for p in MorphologyProbe(x.shape[1], args.hidden, args.dropout).parameters())), "roles": {k: int(len(v)) for k, v in ids.items()}, "variants": variants, "normalization_sha256": _sha256(root / "normalization.npz"), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v22_used": False}


def run(args):
    audit, labels, weak, receiver, ids = _load(args); device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_ids = np.sort(np.concatenate(tuple(ids.values())))
    signals = _load_signals(args.data_path, all_ids); extracted = _extract_vft(signals, device, args.feature_batch_size)
    features = np.zeros((len(labels), extracted.shape[1]), dtype=np.float32); features[all_ids] = extracted
    args.output_root.mkdir(parents=True, exist_ok=True); rows = [_one_seed(args, features, labels, weak, receiver, ids, seed, device) for seed in args.seeds]
    def delta(section, field, salt): return _mean_ci([row["variants"]["M_VBF"][section][field] - row["variants"]["M_BCE"][section][field] for row in rows], salt)
    result_delta = {"weak": {field: delta("weak_vft_vs_hard_negative", field, 220100 + i) for i, field in enumerate(("auroc", "aupr"))}, "strong": {field: delta("strong_vft_vs_hard_negative", field, 220200 + i) for i, field in enumerate(("auroc", "aupr"))}, "overall": {field: delta("overall", field, 220300 + i) for i, field in enumerate(("auroc", "aupr"))}}
    weak_auc_seed = [row["variants"]["M_VBF"]["weak_vft_vs_hard_negative"]["auroc"] - row["variants"]["M_BCE"]["weak_vft_vs_hard_negative"]["auroc"] for row in rows]
    strong_auc_seed = [row["variants"]["M_VBF"]["strong_vft_vs_hard_negative"]["auroc"] - row["variants"]["M_BCE"]["strong_vft_vs_hard_negative"]["auroc"] for row in rows]
    overall_auc_seed = [row["variants"]["M_VBF"]["overall"]["auroc"] - row["variants"]["M_BCE"]["overall"]["auroc"] for row in rows]
    checks = {"weak_auroc_gain_ge_0p030": result_delta["weak"]["auroc"]["mean"] >= .030, "weak_auroc_ci_positive": result_delta["weak"]["auroc"]["bootstrap_95_ci"][0] > 0, "weak_auroc_positive_4of5": sum(value > 0 for value in weak_auc_seed) >= 4, "weak_aupr_gain_ge_0p020": result_delta["weak"]["aupr"]["mean"] >= .020, "strong_auroc_safe_every_seed": all(value >= -.010 for value in strong_auc_seed), "overall_auroc_safe_every_seed": all(value >= -.010 for value in overall_auc_seed), "matched_architecture": len({row["parameter_count"] for row in rows}) == 1 and len({row["feature_dim"] for row in rows}) == 1, "test_v3_sealed": True}
    passed = bool(all(checks.values()))
    result = {"status": "completed", "experiment": "route_a_v22_visibility_balanced_morphology_expert", "evidence_role": "fresh_identity_isolated_visibility_balanced_training_probe", "data_audit_sha256": _sha256(args.audit), "seeds": list(args.seeds), "architecture": {"input_dim": 268, "hidden": args.hidden, "dropout": args.dropout, "parameter_count": rows[0]["parameter_count"]}, "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs": args.epochs, "checkpoint_criterion": "unweighted_three_stratum_early_loss"}, "rows": rows, "vbf_minus_bce": result_delta, "gate": {"checks": checks, "passed": passed, "decision": "permit_fresh_fusion_design" if passed else "stop_visibility_balanced_morphology_training"}, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v22_used": False}
    _write(args.output_root / "route_a_v22_visibility_balanced_morphology_expert_summary.json", result)
    _write(args.output_root / "run_receipt.json", {"status": "completed", "gate_passed": passed, "decision": result["gate"]["decision"], "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v22_used": False})
    print(json.dumps({"passed": passed, "checks": checks, "output": str(args.output_root)}, ensure_ascii=False, indent=2)); return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v22_visibility_balanced_morphology\closedset_6class_train.mat"))
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v22_visibility_balanced_morphology_generation/identity_audit.json")
    parser.add_argument("--roles", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v22_visibility_balanced_morphology_generation/role_indices.npz")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v22_visibility_balanced_morphology_expert")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None)
    parser.add_argument("--hidden", type=int, default=64); parser.add_argument("--dropout", type=float, default=.10); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--eval-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--weak-threshold-db", type=float, default=10.)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
