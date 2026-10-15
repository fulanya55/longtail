"""Run the frozen three-seed oracle-positive screen for physics parameter experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cj0724 import DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset, read_labels
from cj0724.physics_parameter_specialists import build_physics_specialist, ordinal_count_loss
from cj0724.train import set_seed


SPECIALISTS = {
    "sawfm_period_sequence": {"class_index": 1, "column": 14, "kind": "scalar"},
    "mft_delay_peak_ordinal": {"class_index": 3, "column": 28, "kind": "ordinal"},
    "vft_instantaneous_frequency": {"class_index": 4, "column": 49, "kind": "scalar"},
}


def _sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _stratified_limit(indices: np.ndarray, target: np.ndarray, maximum: int, seed: int, kind: str) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= maximum:
        return indices
    selected_target = target[indices]
    if kind == "ordinal":
        groups = np.rint(selected_target).astype(np.int64)
    else:
        order = np.argsort(selected_target, kind="mergesort")
        groups = np.empty(len(indices), dtype=np.int64)
        groups[order] = np.minimum(np.arange(len(indices)) * 10 // len(indices), 9)
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(indices[groups == group]) for group in np.unique(groups)]
    offsets = np.zeros(len(buckets), dtype=np.int64)
    output: list[int] = []
    while len(output) < maximum:
        for bucket_index, bucket in enumerate(buckets):
            if offsets[bucket_index] < len(bucket):
                output.append(int(bucket[offsets[bucket_index]]))
                offsets[bucket_index] += 1
                if len(output) == maximum:
                    break
    return rng.permutation(np.asarray(output, dtype=np.int64))


def _positive(indices: np.ndarray, labels: np.ndarray, mask: np.ndarray, class_index: int, column: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    keep = (labels[indices, class_index] > 0) & mask[column, indices]
    return indices[keep]


def _loss(output: dict[str, torch.Tensor], target_raw: torch.Tensor, kind: str, mean: float, scale: float) -> torch.Tensor:
    if kind == "ordinal":
        return ordinal_count_loss(output["ordinal_logits"], target_raw)
    target = (target_raw - mean) / scale
    return F.smooth_l1_loss(output["prediction"], target)


@torch.inference_mode()
def _predict(model, loader, device, kind: str, mean: float, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    prediction, target, sample_index = [], [], []
    for batch in loader:
        output = model(batch["iq"].to(device))["prediction"].float()
        if kind != "ordinal":
            output = output * scale + mean
        prediction.append(output.cpu().numpy())
        target.append(batch["parameter_values"][:, 0].numpy())
        sample_index.append(batch["sample_index"].numpy())
    return np.concatenate(prediction), np.concatenate(target), np.concatenate(sample_index)


def _metrics(prediction: np.ndarray, target: np.ndarray, training_median: float) -> dict[str, float]:
    error = np.abs(prediction - target)
    baseline = np.abs(target - training_median)
    correlation = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) > 0 else 0.0
    return {
        "sample_count": int(len(target)),
        "mae": float(error.mean()),
        "constant_median_mae": float(baseline.mean()),
        "relative_mae_reduction_vs_constant": float(1.0 - error.mean() / baseline.mean()),
        "pearson_correlation": correlation,
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--role-file", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/role_indices.npz")
    parser.add_argument("--output-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--max-train-positive", type=int, default=2048)
    parser.add_argument("--max-early-positive", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    roles = np.load(args.role_file)
    labels = read_labels(args.data_file)
    with h5py.File(args.data_file, "r") as handle:
        values = np.asarray(handle["parameter_values"], dtype=np.float64)
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    all_runs = []
    for name, spec in SPECIALISTS.items():
        for seed in args.seeds:
            set_seed(seed)
            train_positive = _positive(roles["train"], labels, mask, spec["class_index"], spec["column"])
            early_positive = _positive(roles["early"], labels, mask, spec["class_index"], spec["column"])
            confirm_positive = _positive(roles["confirm"], labels, mask, spec["class_index"], spec["column"])
            target_values = values[spec["column"]]
            train_indices = _stratified_limit(train_positive, target_values, args.max_train_positive, seed + 101, spec["kind"])
            early_indices = _stratified_limit(early_positive, target_values, args.max_early_positive, seed + 102, spec["kind"])
            training_target = target_values[train_indices]
            training_median = float(np.median(training_target))
            mean = float(np.mean(training_target))
            scale = float(max(np.std(training_target), 1e-8))
            datasets = {
                "train": V73IQDataset(args.data_file, train_indices, [spec["column"]]),
                "early": V73IQDataset(args.data_file, early_indices, [spec["column"]]),
                "confirm": V73IQDataset(args.data_file, confirm_positive, [spec["column"]]),
            }
            generator = torch.Generator().manual_seed(seed)
            loaders = {
                "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator, pin_memory=device.type == "cuda"),
                "early": DataLoader(datasets["early"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
                "confirm": DataLoader(datasets["confirm"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
            }
            model = build_physics_specialist(name, args.width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            run_dir = output_root / name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            best_loss, stale, history = float("inf"), 0, []
            start = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                model.train()
                training_losses = []
                for batch in loaders["train"]:
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch["iq"].to(device))
                    target_raw = batch["parameter_values"][:, 0].to(device)
                    loss = _loss(output, target_raw, spec["kind"], mean, scale)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    training_losses.append(float(loss.detach()))
                model.eval()
                early_losses = []
                with torch.inference_mode():
                    for batch in loaders["early"]:
                        output = model(batch["iq"].to(device))
                        early_losses.append(float(_loss(output, batch["parameter_values"][:, 0].to(device), spec["kind"], mean, scale)))
                early_loss = float(np.mean(early_losses))
                history.append({"epoch": epoch, "train_loss": float(np.mean(training_losses)), "early_loss": early_loss})
                if early_loss < best_loss:
                    best_loss, stale = early_loss, 0
                    torch.save(model.state_dict(), run_dir / "best.pt")
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
            prediction, target, sample_index = _predict(model, loaders["confirm"], device, spec["kind"], mean, scale)
            metrics = _metrics(prediction, target, training_median)
            receipt = {
                "specialist": name, "seed": seed, **spec, **metrics,
                "train_positive_count": int(len(train_indices)), "early_positive_count": int(len(early_indices)),
                "confirm_positive_count": int(len(confirm_positive)), "train_indices_sha256": _sha(train_indices),
                "early_indices_sha256": _sha(early_indices), "confirm_indices_sha256": _sha(confirm_positive),
                "epochs_ran": len(history), "best_early_loss": best_loss,
                "train_seconds": time.perf_counter() - start, "locked_test_used": False,
                "classification_loss_used": False, "oracle_positive_only": True,
            }
            (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            np.savez_compressed(run_dir / "predictions.npz", prediction=prediction, target=target, sample_index=sample_index)
            all_runs.append(receipt)
            print(json.dumps(receipt), flush=True)
            for dataset in datasets.values():
                dataset.close()
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    gates = []
    for name in SPECIALISTS:
        rows = [row for row in all_runs if row["specialist"] == name]
        reductions = [row["relative_mae_reduction_vs_constant"] for row in rows]
        correlations = [row["pearson_correlation"] for row in rows]
        gates.append({
            "specialist": name,
            "mean_relative_mae_reduction_vs_constant": float(np.mean(reductions)),
            "mean_pearson_correlation": float(np.mean(correlations)),
            "positive_mae_direction_seed_count": int(sum(value > 0 for value in reductions)),
            "positive_correlation_seed_count": int(sum(value > 0 for value in correlations)),
            "reattach_passed": bool(len(rows) == 3 and np.mean(reductions) >= 0.10 and np.mean(correlations) >= 0.30 and all(value > 0 for value in reductions) and all(value > 0 for value in correlations)),
        })
    summary = {
        "protocol": vars(args),
        "promotion_gate": {"mae_reduction_minimum": 0.10, "correlation_minimum": 0.30, "same_direction": "3/3 seeds"},
        "gates": gates, "runs": all_runs, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
