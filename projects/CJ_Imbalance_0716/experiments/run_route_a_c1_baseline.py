"""Train a clean six-class Route-A C1 baseline on the audited nine-class MATs.

The selected classes are FMB, AMA, SawFM, SinFM, MFT and ISRJ. Samples that
contain any excluded class (SN, VFT or RangeFakeTarget) are not projected to
negative labels; they are removed from this closed-set baseline. The run is
development evidence and never opens a locked test artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_nineclass_feature_sufficiency import (  # noqa: E402
    CLASS_NAMES as NINE_CLASS_NAMES,
    POWERS_DB,
    NineClassDataset,
    find_file,
    profile_cost,
    receiver_seeds,
    sha256,
    write_json,
)

SELECTED_INDICES = (0, 1, 2, 3, 5, 7)
EXCLUDED_INDICES = (4, 6, 8)
CLASS_NAMES = tuple(NINE_CLASS_NAMES[i] for i in SELECTED_INDICES)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pure_six_indices(path: Path) -> np.ndarray:
    """Return samples with >=1 selected label and no excluded label."""
    with h5py.File(str(path), "r") as handle:
        labels = np.asarray(handle["known_label"][:, :], dtype=np.uint8)
    mask = (labels[list(SELECTED_INDICES)].sum(axis=0) > 0) & (labels[list(EXCLUDED_INDICES)].sum(axis=0) == 0)
    return np.flatnonzero(mask).astype(np.int64)


def metrics(targets: np.ndarray, logits: np.ndarray) -> dict:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    preds = (probs >= 0.5).astype(np.uint8)
    per_label = {}
    auprs, aurocs = [], []
    for i, name in enumerate(CLASS_NAMES):
        truth = targets[:, i]
        positive = truth == 1
        negative = ~positive
        aupr = float(average_precision_score(truth, probs[:, i]))
        auroc = float(roc_auc_score(truth, probs[:, i])) if len(np.unique(truth)) == 2 else float("nan")
        per_label[name] = {
            "aupr": aupr,
            "auroc": auroc,
            "recall": float(preds[positive, i].mean()) if positive.any() else float("nan"),
            "negative_fpr": float(preds[negative, i].mean()) if negative.any() else float("nan"),
            "f1": float(f1_score(truth, preds[:, i], zero_division=0)),
            "positive_count": int(positive.sum()),
        }
        auprs.append(aupr)
        aurocs.append(auroc)
    powers = (2 ** np.arange(6, dtype=np.int64))[None, :]
    true_codes = (targets.astype(np.int64) * powers).sum(axis=1)
    pred_codes = (preds.astype(np.int64) * powers).sum(axis=1)
    transitions = {}
    for truth, pred in zip(true_codes.tolist(), pred_codes.tolist()):
        key = f"{int(truth):02d}->{int(pred):02d}"
        transitions[key] = transitions.get(key, 0) + 1
    errors = sorted(
        [{"transition": k, "count": v} for k, v in transitions.items() if k.split("->")[0] != k.split("->")[1]],
        key=lambda row: (-row["count"], row["transition"]),
    )[:30]
    return {
        "macro_aupr": float(np.nanmean(auprs)),
        "macro_auroc": float(np.nanmean(aurocs)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "hamming_loss": float(np.mean(preds != targets)),
        "subset_accuracy": float(np.mean(np.all(preds == targets, axis=1))),
        "combination_exact_accuracy": float(np.mean(true_codes == pred_codes)),
        "target_unique_combinations": int(len(np.unique(true_codes))),
        "predicted_unique_combinations": int(len(np.unique(pred_codes))),
        "combination_error_top30": errors,
        "per_label": per_label,
    }


def initialize_f2(model: nn.Module, checkpoint: Path):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    full_state = payload["model_state_dict"]
    shared = {key[len("backbone."):]: value for key, value in full_state.items() if key.startswith("backbone.") and not key.startswith("backbone.classifier.")}
    missing, unexpected = model.load_state_dict(shared, strict=False)
    missing = [key for key in missing if not key.startswith("classifier.")]
    unexpected = [key for key in unexpected if not key.startswith(("label_feature_bank.", "prototype_memory."))]
    if missing or unexpected:
        raise RuntimeError(f"F2 initialization mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "best_epoch": payload.get("best_epoch")}


def predict(model, path: Path, indices: np.ndarray, device: torch.device, batch_size: int):
    dataset = NineClassDataset(path, indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logits, targets = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                logits.append(model(batch["signal"].to(device, non_blocking=True)).cpu().numpy())
                targets.append(batch["label"].numpy())
    finally:
        dataset.close()
    return np.concatenate(targets).astype(np.uint8), np.concatenate(logits).astype(np.float32)


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("C1 baseline requires CUDA")
    set_seed(args.seed)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    train_path = find_file(args.data_root, args.train_power, "train")
    pure_train = pure_six_indices(train_path)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(pure_train))
    n_fit = int(round(len(pure_train) * (1.0 - args.val_fraction)))
    fit_indices = np.sort(pure_train[perm[:n_fit]])
    val_indices = np.sort(pure_train[perm[n_fit:]])
    if args.max_fit_samples > 0 and len(fit_indices) > args.max_fit_samples:
        fit_indices = fit_indices[: args.max_fit_samples]
    test_indices_by_power = {}
    test_paths = {}
    for power in POWERS_DB:
        path = find_file(args.data_root, power, "test")
        pure = pure_six_indices(path)
        if power == args.primary_test_power:
            if args.max_primary_test_samples > 0 and len(pure) > args.max_primary_test_samples:
                indices = np.sort(np.random.default_rng(args.split_seed + power).choice(pure, size=args.max_primary_test_samples, replace=False))
            else:
                indices = pure
        else:
            indices = np.sort(np.random.default_rng(args.split_seed + power).choice(pure, size=min(args.diagnostic_samples, len(pure)), replace=False))
        test_paths[power] = path
        test_indices_by_power[power] = indices
    fit_seed_set = set(receiver_seeds(train_path, fit_indices))
    val_seed_set = set(receiver_seeds(train_path, val_indices))
    test_overlap = {str(power): len(fit_seed_set.intersection(receiver_seeds(path, idx))) for power, (path, idx) in ((p, (test_paths[p], test_indices_by_power[p])) for p in POWERS_DB)}
    role_manifest = {
        "schema": "route-a-c1-pure-six-role-manifest-v1",
        "selected_classes": list(CLASS_NAMES),
        "excluded_classes": [NINE_CLASS_NAMES[i] for i in EXCLUDED_INDICES],
        "filter": "at_least_one_selected_and_zero_excluded",
        "fit": {"path": str(train_path), "power_db": args.train_power, "count": int(len(fit_indices)), "sample_indices": fit_indices.tolist(), "receiver_noise_seed": sorted(fit_seed_set)},
        "validation": {"path": str(train_path), "count": int(len(val_indices)), "sample_indices": val_indices.tolist(), "receiver_noise_seed": sorted(val_seed_set)},
        "test": {"paths_by_power": {str(p): str(test_paths[p]) for p in POWERS_DB}, "count_by_power": {str(p): int(len(test_indices_by_power[p])) for p in POWERS_DB}, "sample_indices_by_power": {str(p): test_indices_by_power[p].tolist() for p in POWERS_DB}, "fit_receiver_seed_overlap_by_power": test_overlap},
        "locked_test_used": False,
    }
    write_json(output / "role_manifest.json", role_manifest)
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    init_meta = initialize_f2(model, args.f2_checkpoint_root / f"seed_{args.seed}" / "backbone" / "best_model.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    fit_dataset = NineClassDataset(train_path, fit_indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    fit_loader = DataLoader(fit_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    history = []
    best_val = -float("inf")
    best_epoch = 0
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in fit_loader:
                signal = batch["signal"].to(device, non_blocking=True)
                target = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(signal), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_targets, val_logits = predict(model, train_path, val_indices, device, args.eval_batch_size)
            val_metrics = metrics(val_targets, val_logits)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_metrics}
            history.append(row)
            print(json.dumps({"epoch": epoch, "train_loss": row["train_loss"], "val_macro_aupr": val_metrics["macro_aupr"], "val_subset_accuracy": val_metrics["subset_accuracy"]}, ensure_ascii=False), flush=True)
            if val_metrics["macro_aupr"] > best_val:
                best_val = val_metrics["macro_aupr"]
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation": val_metrics}, output / "best_model.pth")
    finally:
        fit_dataset.close()
    checkpoint = output / "best_model.pth"
    model.load_state_dict(torch.load(str(checkpoint), map_location=device, weights_only=False)["model_state_dict"])
    result = {
        "schema": "route-a-c1-pure-six-baseline-v1",
        "status": "completed",
        "selected_classes": list(CLASS_NAMES),
        "excluded_classes": [NINE_CLASS_NAMES[i] for i in EXCLUDED_INDICES],
        "backbone": "F2_DFMS_Full",
        "interface": "complex_1d_true [B,2,3000]",
        "initialization": init_meta,
        "seed": args.seed,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "test_metrics_by_power": {},
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "cost_profile": profile_cost(model, device),
        "locked_test_used": False,
        "elapsed_seconds": float(time.time() - started),
    }
    for power in POWERS_DB:
        targets, logits = predict(model, test_paths[power], test_indices_by_power[power], device, args.eval_batch_size)
        result["test_metrics_by_power"][str(power)] = metrics(targets, logits)
    torch.save(model.state_dict(), output / "best_model_state_only.pth")
    result["checkpoint_sha256"] = sha256(output / "best_model_state_only.pth")
    result["role_manifest_sha256"] = sha256(output / "role_manifest.json")
    result["script_sha256"] = sha256(Path(__file__))
    write_json(output / "baseline_result.json", result)
    write_json(output / "experiment_config.json", {
        "schema": "route-a-c1-pure-six-baseline-v1",
        "classes": list(CLASS_NAMES), "excluded_classes": [NINE_CLASS_NAMES[i] for i in EXCLUDED_INDICES],
        "backbone": "F2_DFMS_Full", "input": "[B,2,3000]", "loss": "BCEWithLogitsLoss", "optimizer": "AdamW",
        "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs, "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size, "train_power": args.train_power, "primary_test_power": args.primary_test_power,
        "val_fraction": args.val_fraction, "diagnostic_samples": args.diagnostic_samples, "seed": args.seed,
        "max_fit_samples": args.max_fit_samples,
        "max_primary_test_samples": args.max_primary_test_samples,
        "split_seed": args.split_seed, "locked_test_used": False,
    })
    write_json(output / "terminal_receipt.json", {"status": "completed", "result": str((output / "baseline_result.json").resolve()), "locked_test_used": False, "test_files_opened": True, "script_sha256": sha256(Path(__file__))})
    print(json.dumps({"status": "completed", "result": str(output / "baseline_result.json"), "best_epoch": best_epoch}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    parser.add_argument("--f2-checkpoint-root", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_c1_baseline_20260826")
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument("--split-seed", type=int, default=260826)
    parser.add_argument("--train-power", type=int, default=30)
    parser.add_argument("--primary-test-power", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--diagnostic-samples", type=int, default=2048)
    parser.add_argument("--max-fit-samples", type=int, default=0, help="0 uses all split fit samples")
    parser.add_argument("--max-primary-test-samples", type=int, default=0, help="0 uses all primary test samples")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
