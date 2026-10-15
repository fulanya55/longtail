"""Route-A six-class combination screen on the audited nine-class source.

All six candidates use the same initialized F2_DFMS_Full backbone, input
interface, BCE loss, optimizer, split and threshold.  This is development
evidence only and never opens a locked-test artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

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
    F2HeadWrapper,
    find_file,
    profile_cost,
    receiver_seeds,
    sha256,
    write_json,
)

SUBSETS = {
    "C0_original_six": (0, 2, 3, 5, 6, 7),
    "C1_replace_VFT_AMA": (0, 1, 2, 3, 5, 7),
    "C2_replace_VFT_SN": (0, 2, 3, 4, 5, 7),
    "C3_replace_VFT_RangeFakeTarget": (0, 2, 3, 5, 7, 8),
    "C4_replace_ISRJ_AMA": (0, 1, 2, 3, 5, 6),
    "C5_replace_ISRJ_SN": (0, 2, 3, 4, 5, 6),
}
SEEDS = (2039, 2040, 2041)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(targets: np.ndarray, logits: np.ndarray, names: tuple[str, ...]) -> dict:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    predictions = (probabilities >= 0.5).astype(np.uint8)
    per_label = {}
    auprs, aurocs = [], []
    for index, name in enumerate(names):
        truth = targets[:, index]
        positive = truth == 1
        negative = ~positive
        aupr = float(average_precision_score(truth, probabilities[:, index]))
        auroc = float(roc_auc_score(truth, probabilities[:, index])) if len(np.unique(truth)) == 2 else float("nan")
        per_label[name] = {
            "aupr": aupr,
            "auroc": auroc,
            "recall": float(predictions[positive, index].mean()) if positive.any() else float("nan"),
            "negative_fpr": float(predictions[negative, index].mean()) if negative.any() else float("nan"),
            "f1": float(f1_score(truth, predictions[:, index], zero_division=0)),
            "positive_count": int(positive.sum()),
        }
        auprs.append(aupr)
        aurocs.append(auroc)
    powers = (2 ** np.arange(targets.shape[1], dtype=np.int64))[None, :]
    true_codes = (targets.astype(np.int64) * powers).sum(axis=1)
    pred_codes = (predictions.astype(np.int64) * powers).sum(axis=1)
    transitions = {}
    for truth, pred in zip(true_codes.tolist(), pred_codes.tolist()):
        key = f"{int(truth):02d}->{int(pred):02d}"
        transitions[key] = transitions.get(key, 0) + 1
    top_errors = sorted(
        [{"transition": key, "count": value} for key, value in transitions.items() if key.split("->")[0] != key.split("->")[1]],
        key=lambda row: (-row["count"], row["transition"]),
    )[:20]
    return {
        "macro_aupr": float(np.nanmean(auprs)),
        "macro_auroc": float(np.nanmean(aurocs)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(np.mean(predictions != targets)),
        "subset_accuracy": float(np.mean(np.all(predictions == targets, axis=1))),
        "label_cardinality_error": float(np.mean(np.abs(predictions.sum(1) - targets.sum(1)))),
        "combination_exact_accuracy": float(np.mean(true_codes == pred_codes)),
        "target_unique_combinations": int(len(np.unique(true_codes))),
        "predicted_unique_combinations": int(len(np.unique(pred_codes))),
        "combination_error_top20": top_errors,
        "per_label": per_label,
    }


def predict(model, path: Path, indices: np.ndarray, label_indices: tuple[int, ...], device: torch.device, batch_size: int):
    dataset = NineClassDataset(path, indices, label_indices=np.asarray(label_indices, dtype=np.int64))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logits, targets = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                logits.append(model(batch["signal"].to(device, non_blocking=True)).cpu().numpy())
                targets.append(batch["label"].numpy())
    finally:
        dataset.close()
    return np.concatenate(targets).astype(np.uint8), np.concatenate(logits).astype(np.float32)


def initialize_f2(model: nn.Module, checkpoint: Path):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    full_state = payload["model_state_dict"]
    shared = {
        key[len("backbone."):]: value
        for key, value in full_state.items()
        if key.startswith("backbone.") and not key.startswith("backbone.classifier.")
    }
    missing, unexpected = model.load_state_dict(shared, strict=False)
    missing = [key for key in missing if not key.startswith("classifier.")]
    unexpected = [key for key in unexpected if not key.startswith(("label_feature_bank.", "prototype_memory."))]
    if missing or unexpected:
        raise RuntimeError(f"F2 initialization mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "best_epoch": payload.get("best_epoch")}


def run_candidate(args, candidate: str, seed: int, train_indices: np.ndarray, eval_sets: dict[int, tuple[Path, np.ndarray]], device: torch.device, output: Path):
    set_seed(seed)
    label_indices = SUBSETS[candidate]
    names = tuple(NINE_CLASS_NAMES[index] for index in label_indices)
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    init_meta = initialize_f2(model, args.f2_checkpoint_root / f"seed_{seed}" / "backbone" / "best_model.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    dataset = NineClassDataset(find_file(args.data_root, 30, "train"), train_indices, label_indices=np.asarray(label_indices))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    history = []
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                values = batch["signal"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(values), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    finally:
        dataset.close()
    result = {
        "candidate": candidate,
        "seed": int(seed),
        "class_indices": list(label_indices),
        "class_names": list(names),
        "initialization": init_meta,
        "status": "completed",
        "history": history,
        "power_metrics": {},
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "cost_profile": profile_cost(model, device),
        "locked_test_used": False,
        "elapsed_seconds": float(time.time() - started),
    }
    for power, (path, indices) in eval_sets.items():
        targets, logits = predict(model, path, indices, label_indices, device, args.eval_batch_size)
        result["power_metrics"][str(power)] = metrics(targets, logits, names)
    checkpoint = output / f"{candidate}__seed_{seed}.pth"
    torch.save(model.state_dict(), checkpoint)
    result["checkpoint_sha256"] = sha256(checkpoint)
    write_json(output / f"{candidate}__seed_{seed}.json", result)
    return result


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("six-class screen requires CUDA")
    candidate_names = tuple(args.candidates) if args.candidates else tuple(SUBSETS)
    unknown = [name for name in candidate_names if name not in SUBSETS]
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    train_path = find_file(args.data_root, 30, "train")
    with __import__("h5py").File(str(train_path), "r") as handle:
        train_count = int(handle["iq_complex"].shape[1])
    permutation = np.random.default_rng(args.split_seed).permutation(train_count)
    train_indices = np.sort(permutation[: args.train_samples])
    eval_sets = {}
    for power in POWERS_DB:
        path = find_file(args.data_root, power, "test")
        with __import__("h5py").File(str(path), "r") as handle:
            count = int(handle["iq_complex"].shape[1])
        indices = np.sort(np.random.default_rng(args.split_seed + power).permutation(count)[: args.eval_samples])
        eval_sets[power] = (path, indices)
    role_manifest = {
        "schema": "route-a-sixclass-combo-role-manifest-v1",
        "fit": {"path": str(train_path), "count": int(len(train_indices)), "sample_indices": train_indices.tolist(), "receiver_noise_seed": receiver_seeds(train_path, train_indices)},
        "development_by_power": {str(power): {"path": str(path), "sample_indices": indices.tolist(), "receiver_noise_seed": receiver_seeds(path, indices)} for power, (path, indices) in eval_sets.items()},
        "validation": {"used": False}, "early": {"used": False}, "calibration": {"used": False}, "test": {"used": False, "locked_test_used": False},
    }
    write_json(output / "role_manifest.json", role_manifest)
    write_json(output / "experiment_config.json", {
        "schema": "route-a-sixclass-combo-screen-v1",
        "backbone": "F2_DFMS_Full",
        "backbone_interface": "complex_1d_true [B,2,3000]",
        "loss": "BCEWithLogitsLoss",
        "optimizer": "AdamW",
        "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "candidates": {key: {"indices": list(SUBSETS[key]), "names": [NINE_CLASS_NAMES[index] for index in SUBSETS[key]]} for key in candidate_names},
        "seeds": list(args.seeds), "split_seed": args.split_seed,
        "data_root": str(args.data_root), "f2_checkpoint_root": str(args.f2_checkpoint_root),
        "locked_test_used": False, "test_files_opened": False,
    })
    rows = []
    for candidate in candidate_names:
        candidate_dir = output / candidate
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            artifact = candidate_dir / f"{candidate}__seed_{seed}.json"
            if args.resume and artifact.is_file():
                rows.append(json.loads(artifact.read_text(encoding="utf-8")))
            else:
                rows.append(run_candidate(args, candidate, int(seed), train_indices, eval_sets, device, candidate_dir))
    summary = {"schema": "route-a-sixclass-combo-summary-v1", "status": "development_screen_completed", "locked_test_used": False, "candidates": {}}
    for candidate in candidate_names:
        label_indices = SUBSETS[candidate]
        cells = [row for row in rows if row["candidate"] == candidate]
        summary["candidates"][candidate] = {
            "class_names": [NINE_CLASS_NAMES[index] for index in label_indices],
            "macro_aupr_by_power": {str(power): [row["power_metrics"][str(power)]["macro_aupr"] for row in cells] for power in POWERS_DB},
            "macro_f1_by_power": {str(power): [row["power_metrics"][str(power)]["macro_f1"] for row in cells] for power in POWERS_DB},
            "combination_accuracy_by_power": {str(power): [row["power_metrics"][str(power)]["combination_exact_accuracy"] for row in cells] for power in POWERS_DB},
            "mean_macro_aupr_30dB": float(np.mean([row["power_metrics"]["30"]["macro_aupr"] for row in cells])),
            "std_macro_aupr_30dB": float(np.std([row["power_metrics"]["30"]["macro_aupr"] for row in cells], ddof=1)),
            "mean_worst_power_macro_aupr": float(np.mean([min(row["power_metrics"][str(power)]["macro_aupr"] for power in POWERS_DB) for row in cells])),
            "mean_30dB_combination_accuracy": float(np.mean([row["power_metrics"]["30"]["combination_exact_accuracy"] for row in cells])),
            "per_label_30dB": {name: {"aupr": [row["power_metrics"]["30"]["per_label"][name]["aupr"] for row in cells], "f1": [row["power_metrics"]["30"]["per_label"][name]["f1"] for row in cells]} for name in [NINE_CLASS_NAMES[index] for index in label_indices]},
        }
    write_json(output / "sixclass_combo_summary.json", summary)
    write_json(output / "terminal_receipt.json", {"status": "completed", "summary": str((output / "sixclass_combo_summary.json").resolve()), "locked_test_used": False, "test_files_opened": False, "script_sha256": sha256(Path(__file__))})
    print(json.dumps({"status": "completed", "summary": str(output / 'sixclass_combo_summary.json')}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    parser.add_argument("--f2-checkpoint-root", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_sixclass_combo_screen_20260826")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--candidates", type=str, nargs="+", choices=tuple(SUBSETS), default=None)
    parser.add_argument("--split-seed", type=int, default=260826)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
