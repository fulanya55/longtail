"""Fixed T0/T1/T2 five-seed H3 development model comparison.

This runner is intentionally ranking-only: it fits no calibrator and chooses
no threshold.  Its target-only VFT expert is tested against a same-capacity
timestamp-permuted control on the fresh H3 three-strata source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cjlt.models.multipulse_vft_expert import MultiPulseVFTExpertClassifier

MODES = ("T0-h3", "T1-h3", "T2-h3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class H3Dataset(Dataset):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle: h5py.File | None = None
        with h5py.File(self.path, "r") as h5:
            self.length = int(h5["labels"].shape[0])

    def _file(self) -> h5py.File:
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return self.handle

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        h5 = self._file()
        iq = np.stack((h5["iq_real"][index], h5["iq_imag"][index]), axis=0).astype(np.float32, copy=False)
        return {
            "iq": torch.from_numpy(iq),
            "labels": torch.from_numpy(np.asarray(h5["labels"][index], dtype=np.float32)),
            "arm": int(h5["vft_visibility_arm"][index]),
            "pair_seed": int(h5["pair_seed"][index]),
        }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    logits, labels, arms, pairs = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits.append(model(batch["iq"].to(device)).cpu().numpy())
            labels.append(batch["labels"].numpy())
            arms.extend(batch["arm"].tolist())
            pairs.extend(batch["pair_seed"].tolist())
    raw_logits = np.concatenate(logits)
    probabilities = 1.0 / (1.0 + np.exp(-raw_logits))
    target = np.concatenate(labels).astype(bool)
    arm = np.asarray(arms, dtype=np.uint8)
    per_label = []
    for index in range(6):
        y, score = target[:, index], probabilities[:, index]
        per_label.append({"aupr": float(average_precision_score(y, score)), "auroc": float(roc_auc_score(y, score))})
    absent, weak, strong = arm == 0, arm == 1, arm == 2
    def vft_metric(present: np.ndarray) -> dict[str, float]:
        selected = absent | present
        y, score = target[selected, 4], probabilities[selected, 4]
        return {"auroc": float(roc_auc_score(y, score)), "aupr": float(average_precision_score(y, score))}
    return {"macro_aupr": float(np.mean([entry["aupr"] for entry in per_label])), "per_label": per_label, "weak_vft_vs_absent": vft_metric(weak), "strong_vft_vs_absent": vft_metric(strong), "raw_logits": raw_logits, "probabilities": probabilities, "labels": target, "arm": arm, "pair_seed": np.asarray(pairs, dtype=np.uint64)}


def run_arm(mode: str, train_path: Path, development_path: Path, *, seed: int, epochs: int, learning_rate: float, batch_size: int, width: int, expert_dim: int, device: torch.device, output: Path) -> dict[str, object]:
    _seed(seed)
    train, development = H3Dataset(train_path), H3Dataset(development_path)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    development_loader = DataLoader(development, batch_size=batch_size, shuffle=False, num_workers=0)
    model = MultiPulseVFTExpertClassifier(mode, width=width, expert_dim=expert_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[dict[str, float | int]] = []
    best: dict[str, object] | None = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["iq"].to(device))
            loss = loss_fn(prediction, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(prediction); count += len(prediction)
        metric = _evaluate(model, development_loader, device)
        row = {"epoch": epoch, "train_bce": total / max(count, 1), "macro_aupr": metric["macro_aupr"], "weak_vft_auroc": metric["weak_vft_vs_absent"]["auroc"], "strong_vft_auroc": metric["strong_vft_vs_absent"]["auroc"]}
        history.append(row)
        print(json.dumps({"event": "epoch_complete", "mode": mode, "seed": seed, **row}), flush=True)
        if best is None or row["macro_aupr"] > best["metrics"]["macro_aupr"]:
            best = {"epoch": epoch, "metrics": metric, "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
    assert best is not None
    model.load_state_dict(best.pop("state"))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    torch.save({"mode": mode, "seed": seed, "state_dict": model.state_dict(), "best_epoch": best["epoch"], "history": history}, checkpoint)
    metric = best["metrics"]
    np.savez_compressed(output / "development_predictions.npz", raw_logits=metric.pop("raw_logits"), probabilities=metric.pop("probabilities"), labels=metric.pop("labels"), vft_visibility_arm=metric.pop("arm"), pair_seed=metric.pop("pair_seed"))
    return {"mode": mode, "seed": seed, "best_epoch": best["epoch"], "history": history, "metrics": metric, "parameter_count": int(sum(value.numel() for value in model.parameters())), "checkpoint": str(checkpoint), "train_seconds": time.perf_counter() - started}


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.train_path.resolve() == args.development_path.resolve() and not args.smoke:
        raise ValueError("formal train/development paths must differ")
    train_audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    development_audit = json.loads(args.development_audit.read_text(encoding="utf-8"))
    if not all(train_audit["contracts"].values()) or not all(development_audit["contracts"].values()):
        raise ValueError("H3 source audit failed")
    if any(train_audit.get(key) or development_audit.get(key) for key in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError("H3 source audit violates test boundary")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = []
    for seed in args.seeds:
        for mode in args.modes:
            result.append(run_arm(mode, args.train_path, args.development_path, seed=seed, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, width=args.width, expert_dim=args.expert_dim, device=device, output=args.output_root / f"{mode}_seed{seed}"))
    receipt = {"status": "completed_smoke_only" if args.smoke else "completed_development_unaggregated", "experiment": "RouteA_H3_VFTThreeStrataIdentifiability", "modes": list(args.modes), "seeds": list(args.seeds), "source": {"train": str(args.train_path), "development": str(args.development_path), "train_sha256": _sha256(args.train_path), "development_sha256": _sha256(args.development_path), "train_audit": str(args.train_audit), "development_audit": str(args.development_audit)}, "config": {"epochs": args.epochs, "learning_rate": args.learning_rate, "weight_decay": 1e-4, "batch_size": args.batch_size, "width": args.width, "expert_dim": args.expert_dim, "optimizer": "AdamW", "loss": "BCEWithLogitsLoss", "gradient_clip_norm": 5.0, "calibration": "none", "threshold_selection": "none"}, "device": str(device), "results": result, "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "smoke": bool(args.smoke)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--development-path", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261003, 20261011, 20261019, 20261027, 20261104])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--expert-dim", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    value = main(parser.parse_args())
    print(json.dumps({"status": value["status"], "modes": value["modes"], "seeds": value["seeds"]}, ensure_ascii=False))
