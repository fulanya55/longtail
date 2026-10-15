"""Diagnose power behavior, VFT false-positive combinations, and expert diversity."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from cj0724.dual_axis import CLASS_NAMES, PowerViewDataset, discover_power_train_files, read_identity_metadata
from cj0724.dual_axis_train import make_loader
from cj0724.models import build_benchmark_model


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def combo_key(label: np.ndarray) -> str:
    return "".join(str(int(value)) for value in label.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/vft_d11_backbone_loss_20260725"),
    )
    parser.add_argument(
        "--power-root",
        type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    summaries = []
    for path in sorted(args.run_root.glob("VFT/D11/*/seed_*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["run_dir"] = path.parent
        summaries.append(item)
    if not summaries:
        raise FileNotFoundError(f"No summaries under {args.run_root}")

    power_raw = []
    combo_accumulator: dict[tuple[str, str], dict[str, float]] = {}
    target_index = CLASS_NAMES.index("VFT")
    for item in summaries:
        run_dir = Path(item["run_dir"])
        with (run_dir / "power_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                power_raw.append(
                    {
                        "method": item["method"],
                        "seed": int(item["seed"]),
                        "level": int(row["level"]),
                        "macro_aupr": float(row["macro_aupr"]),
                        "target_aupr": float(row["target_aupr"]),
                        "hamming": float(row["hamming_calibrated"]),
                        "weak_recall": float(row["weak_recall_calibrated"]),
                        "target_fpr": float(row["target_fpr_calibrated"]),
                    }
                )
        predictions = np.load(run_dir / "development_predictions.npz")
        labels = predictions["labels_10"].astype(np.uint8)
        probabilities = predictions["probabilities_10"].astype(np.float64)
        threshold_payload = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        threshold = float(threshold_payload["thresholds"][target_index])
        predicted_vft = probabilities[:, target_index] >= threshold
        for key in sorted({combo_key(label) for label in labels if not label[target_index]}):
            mask = np.asarray([combo_key(label) == key for label in labels])
            support = int(mask.sum())
            accumulator = combo_accumulator.setdefault(
                (item["method"], key),
                {"support": 0.0, "false_positives": 0.0, "probability_sum": 0.0},
            )
            accumulator["support"] += support
            accumulator["false_positives"] += int(predicted_vft[mask].sum())
            accumulator["probability_sum"] += float(probabilities[mask, target_index].sum())

    power_rows = []
    for method, level in sorted({(row["method"], row["level"]) for row in power_raw}):
        selected = [row for row in power_raw if row["method"] == method and row["level"] == level]
        aggregate = {"method": method, "level": level, "runs": len(selected)}
        for metric in ("macro_aupr", "target_aupr", "hamming", "weak_recall", "target_fpr"):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            aggregate[f"{metric}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        power_rows.append(aggregate)
    write_csv(args.run_root / "improvement_power_curves.csv", power_rows)

    combo_rows = []
    for (method, key), value in combo_accumulator.items():
        support = int(value["support"])
        combo_rows.append(
            {
                "method": method,
                "combination_bits": key,
                "active_labels": "+".join(
                    CLASS_NAMES[index] for index, bit in enumerate(key) if bit == "1"
                ),
                "support_across_seeds": support,
                "vft_false_positives": int(value["false_positives"]),
                "vft_fpr": float(value["false_positives"] / support),
                "mean_vft_probability": float(value["probability_sum"] / support),
            }
        )
    combo_rows.sort(key=lambda row: (row["method"], -row["vft_fpr"], row["combination_bits"]))
    write_csv(args.run_root / "vft_negative_combo_fpr.csv", combo_rows)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    files = discover_power_train_files(args.power_root)
    _, all_labels = read_identity_metadata(files[30])
    diversity_rows = []
    for item in summaries:
        if item["model"] != "efficientnet_b1_moe3_1d":
            continue
        run_dir = Path(item["run_dir"])
        split = np.load(run_dir / "split_indices.npz")
        indices = split["development"].astype(np.int64)
        dataset = PowerViewDataset(files, all_labels, indices, (10,), int(item["seed"]))
        loader = make_loader(dataset, 64, False, int(item["seed"]))
        model = build_benchmark_model(item["model"], num_classes=6, dropout=0.2, head_dim=256).to(device)
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        expert_parts = []
        label_parts = []
        with torch.no_grad():
            for batch in loader:
                logits = model.forward_experts(batch["iq"].to(device, non_blocking=True))
                expert_parts.append(torch.sigmoid(logits).cpu().numpy())
                label_parts.append(batch["label"].numpy())
        dataset.close()
        expert_probability = np.concatenate(expert_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0).astype(np.uint8)
        pair_all = []
        pair_target = []
        for first, second in combinations(range(expert_probability.shape[1]), 2):
            pair_all.append(
                float(np.corrcoef(expert_probability[:, first].reshape(-1), expert_probability[:, second].reshape(-1))[0, 1])
            )
            pair_target.append(
                float(np.corrcoef(expert_probability[:, first, target_index], expert_probability[:, second, target_index])[0, 1])
            )
        expert_aupr = [
            float(average_precision_score(labels[:, target_index], expert_probability[:, index, target_index]))
            for index in range(expert_probability.shape[1])
        ]
        ensemble_probability = expert_probability.mean(axis=1)
        diversity_rows.append(
            {
                "method": item["method"],
                "seed": int(item["seed"]),
                "mean_pairwise_probability_correlation": float(np.mean(pair_all)),
                "mean_pairwise_vft_correlation": float(np.mean(pair_target)),
                "mean_probability_std_across_experts": float(expert_probability.std(axis=1).mean()),
                "mean_vft_probability_std_across_experts": float(expert_probability[:, :, target_index].std(axis=1).mean()),
                "expert_0_vft_aupr": expert_aupr[0],
                "expert_1_vft_aupr": expert_aupr[1],
                "expert_2_vft_aupr": expert_aupr[2],
                "ensemble_vft_aupr": float(average_precision_score(labels[:, target_index], ensemble_probability[:, target_index])),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(args.run_root / "expert_diversity.csv", diversity_rows)
    print(f"Wrote diagnostics for {len(summaries)} runs under {args.run_root.resolve()}")


if __name__ == "__main__":
    main()
