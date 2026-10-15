"""Oracle endpoint isolation screen for 10 dB swept-component identifiability."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader, Dataset

from cj0724.dual_axis import complex_rms_normalize, discover_power_train_files, read_identity_metadata
from cj0724.models import IQGroupedConvClassifier


MODES = ("raw_10db", "oracle_swept_endpoint")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


class SweptIsolationDataset(Dataset):
    def __init__(
        self,
        file_10: Path,
        file_50: Path,
        labels: np.ndarray,
        indices: np.ndarray,
        mode: str,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.file_10 = Path(file_10)
        self.file_50 = Path(file_50)
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode
        self._h10 = None
        self._h50 = None
        self._denominator = float(10.0 ** (50.0 / 20.0) - 10.0 ** (10.0 / 20.0))

    def __len__(self) -> int:
        return int(len(self.indices))

    def _handles(self):
        if self._h10 is None:
            self._h10 = h5py.File(self.file_10, "r")
        if self._h50 is None:
            self._h50 = h5py.File(self.file_50, "r")
        return self._h10, self._h50

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        index = int(self.indices[position])
        h10, h50 = self._handles()
        raw10 = np.asarray(h10["data_legacy"][:, index], dtype=np.float32).reshape(2, -1)
        if self.mode == "raw_10db":
            iq = raw10
        else:
            raw50 = np.asarray(h50["data_legacy"][:, index], dtype=np.float32).reshape(2, -1)
            iq = (raw50 - raw10) / self._denominator
        iq = complex_rms_normalize(iq)
        return {
            "iq": torch.from_numpy(iq),
            "label": torch.from_numpy(self.labels[index, 3:6].astype(np.float32, copy=True)),
            "sample_index": index,
        }

    def close(self) -> None:
        if self._h10 is not None:
            self._h10.close()
        if self._h50 is not None:
            self._h50.close()
        self._h10 = self._h50 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h10"] = state["_h50"] = None
        return state

    def __del__(self) -> None:
        self.close()


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, logits = [], []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(iq)
        labels.append(batch["label"].numpy())
        logits.append(output.float().cpu().numpy())
    logit = np.concatenate(logits)
    return np.concatenate(labels).astype(np.uint8), 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def select_fpr_thresholds(labels: np.ndarray, probabilities: np.ndarray, fpr_cap: float) -> np.ndarray:
    thresholds = np.ones(labels.shape[1], dtype=np.float64)
    grid = np.linspace(0.001, 0.999, 999)
    for class_index in range(labels.shape[1]):
        truth = labels[:, class_index].astype(bool)
        best = (-1.0, -1.0, 1.0)
        for threshold in grid:
            predicted = probabilities[:, class_index] >= threshold
            negative = ~truth
            fpr = float(predicted[negative].mean()) if negative.any() else 0.0
            recall = float(predicted[truth].mean()) if truth.any() else 0.0
            if fpr <= fpr_cap + 1e-12:
                candidate = (recall, -fpr, float(threshold))
                if candidate > best:
                    best = candidate
        thresholds[class_index] = best[2]
    return thresholds


def evaluate(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, object]:
    truth = labels.astype(bool)
    predicted = probabilities >= thresholds[None, :]
    per_class = []
    for index, name in enumerate(("SawFM", "VFT", "ISRJ")):
        positive, negative = truth[:, index], ~truth[:, index]
        per_class.append({
            "class": name,
            "aupr": float(sk_metrics.average_precision_score(positive, probabilities[:, index])),
            "recall": float(predicted[positive, index].mean()),
            "fpr": float(predicted[negative, index].mean()),
            "threshold": float(thresholds[index]),
        })
    return {
        "macro_aupr": float(np.mean([row["aupr"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_fpr": float(np.mean([row["fpr"] for row in per_class])),
        "macro_f1": float(sk_metrics.f1_score(truth, predicted, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def train_one(
    mode: str,
    seed: int,
    files: dict[int, Path],
    labels: np.ndarray,
    roles: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed)
    run_dir = Path(args.output_root) / mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model = IQGroupedConvClassifier(num_classes=3, width=args.width, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    datasets = {
        role: SweptIsolationDataset(files[10], files[50], labels, roles[role], mode)
        for role in ("train", "early", "calibration", "confirm")
    }
    train_loader = make_loader(datasets["train"], args.batch_size, True, seed)
    early_loader = make_loader(datasets["early"], args.batch_size, False, seed)
    best = (-float("inf"), -float("inf"))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_samples = 0.0, 0
        for batch in train_loader:
            iq = batch["iq"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(iq), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            total_samples += len(target)
        early_true, early_probability = predict(model, early_loader, device)
        per_aupr = [sk_metrics.average_precision_score(early_true[:, i], early_probability[:, i]) for i in range(3)]
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            "early_macro_aupr": float(np.mean(per_aupr)),
            "early_min_class_aupr": float(np.min(per_aupr)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps({"mode": mode, "seed": seed, **row}))
        key = (row["early_macro_aupr"], row["early_min_class_aupr"])
        if key > best:
            best = key
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "mode": mode}, run_dir / "best.pt")
        scheduler.step()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    calibration_loader = make_loader(datasets["calibration"], args.batch_size, False, seed)
    confirm_loader = make_loader(datasets["confirm"], args.batch_size, False, seed)
    calibration_true, calibration_probability = predict(model, calibration_loader, device)
    thresholds = select_fpr_thresholds(calibration_true, calibration_probability, args.fpr_cap)
    confirm_true, confirm_probability = predict(model, confirm_loader, device)
    result = {
        "mode": mode,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "calibration_fpr_cap": float(args.fpr_cap),
        "thresholds": thresholds.tolist(),
        "confirm": evaluate(confirm_true, confirm_probability, thresholds),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for dataset in datasets.values():
        dataset.close()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def linearity_audit(files: dict[int, Path], indices: np.ndarray) -> dict[str, object]:
    selected = np.asarray(indices[: min(len(indices), 512)], dtype=np.int64)
    handles = {level: h5py.File(path, "r") for level, path in files.items()}
    try:
        x10 = np.asarray(handles[10]["data_legacy"][:, selected], dtype=np.float64)
        x50 = np.asarray(handles[50]["data_legacy"][:, selected], dtype=np.float64)
        a10, a50 = 10.0 ** 0.5, 10.0 ** 2.5
        rows = []
        for level in sorted(files):
            coefficient = (10.0 ** (level / 20.0) - a10) / (a50 - a10)
            reconstructed = x10 + coefficient * (x50 - x10)
            truth = np.asarray(handles[level]["data_legacy"][:, selected], dtype=np.float64)
            relative = np.sqrt(np.mean((reconstructed - truth) ** 2, axis=0)) / (
                np.sqrt(np.mean(truth ** 2, axis=0)) + 1e-12
            )
            rows.append({
                "power_level_db": level,
                "median_relative_rms": float(np.median(relative)),
                "p95_relative_rms": float(np.quantile(relative, 0.95)),
                "maximum_relative_rms": float(np.max(relative)),
            })
        return {
            "sample_count": int(len(selected)),
            "sample_indices_sha256": sha256_indices(selected),
            "maximum_p95_relative_rms": float(max(row["p95_relative_rms"] for row in rows)),
            "per_level": rows,
        }
    finally:
        for handle in handles.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/oracle_swept_isolation_20260725")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fpr-cap", type=float, default=0.10)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    roles_npz = np.load(run_root / "role_indices.npz")
    roles = {name: roles_npz[name].astype(np.int64) for name in roles_npz.files}
    files = discover_power_train_files(args.power_root)
    seeds10, labels10 = read_identity_metadata(files[10])
    seeds50, labels50 = read_identity_metadata(files[50])
    if not np.array_equal(seeds10, seeds50) or not np.array_equal(labels10, labels50):
        raise RuntimeError("10 and 50 dB identities differ")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audit = linearity_audit(files, roles["confirm"])
    protocol = {
        "modes": MODES,
        "oracle_definition": "(X50-X10)/(10^(50/20)-10^(10/20))",
        "interpretation": "identity-paired multi-view upper bound; not deployable",
        "targets": ["SawFM", "VFT", "ISRJ"],
        "roles": {name: {"count": len(value), "sha256": sha256_indices(value)} for name, value in roles.items()},
        "seeds": args.seeds,
        "epochs": args.epochs,
        "fixed_per_class_fpr_cap": args.fpr_cap,
        "linearity_audit": audit,
        "locked_test_used": False,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for mode in MODES:
        for seed in args.seeds:
            results.append(train_one(mode, seed, files, labels10, roles, args, device))
    aggregates = []
    for mode in MODES:
        rows = [row for row in results if row["mode"] == mode]
        aggregate = {"mode": mode, "seed_count": len(rows)}
        for metric in ("macro_aupr", "macro_recall", "macro_fpr", "macro_f1"):
            values = [row["confirm"][metric] for row in rows]
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values))
        for class_index, name in enumerate(("SawFM", "VFT", "ISRJ")):
            for metric in ("aupr", "recall", "fpr"):
                values = [row["confirm"]["per_class"][class_index][metric] for row in rows]
                aggregate[f"{name}_{metric}_mean"] = float(np.mean(values))
                aggregate[f"{name}_{metric}_std"] = float(np.std(values))
        aggregates.append(aggregate)
    raw = next(row for row in aggregates if row["mode"] == "raw_10db")
    oracle = next(row for row in aggregates if row["mode"] == "oracle_swept_endpoint")
    decision = {
        "oracle_macro_recall_at_fpr_0p10_at_least_0p70": bool(oracle["macro_recall_mean"] >= 0.70),
        "oracle_all_class_recall_at_least_0p70": bool(all(oracle[f"{name}_recall_mean"] >= 0.70 for name in ("SawFM", "VFT", "ISRJ"))),
        "three_seed_macro_aupr_same_direction": bool(all(
            next(row for row in results if row["mode"] == "oracle_swept_endpoint" and row["seed"] == seed)["confirm"]["macro_aupr"]
            > next(row for row in results if row["mode"] == "raw_10db" and row["seed"] == seed)["confirm"]["macro_aupr"]
            for seed in args.seeds
        )),
        "macro_aupr_gain": float(oracle["macro_aupr_mean"] - raw["macro_aupr_mean"]),
        "macro_recall_gain": float(oracle["macro_recall_mean"] - raw["macro_recall_mean"]),
    }
    payload = {"aggregates": aggregates, "decision": decision, "per_seed": results}
    (output_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"aggregates": aggregates, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
