"""Evaluate a complete Route-B R1 development run on nominal and train-only power views.

The evaluator never discovers or opens a test MAT.  It writes separate
per-parameter, visibility, and paired-power tables and refuses to label an
incomplete B0--B4/five-seed run as a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.data import V73IQDataset, read_labels
from cj0724.models import build_benchmark_model
from cj0724.multitask import (
    PARAMETER_COLUMNS,
    build_multitask_model,
    masked_parameter_smooth_l1,
    parameter_metrics,
    reshape_selected_parameters,
)
from cj0724.multitask_benchmark import fit_normalizer
from cj0724.train import multilabel_metrics, get_device


VARIANTS = ("B0", "B1", "B2", "B3", "B4")
JOINT = {"B1", "B2", "B3", "B4"}
POWER_LEVELS = tuple(range(10, 51, 5))


def _float(value: object) -> float:
    return float(value) if value is not None else float("nan")


def evaluate_joint(model, loader, device, normalizer, threshold: float) -> dict[str, object]:
    model.eval()
    logits_parts, labels_parts, pred_parts, target_parts, mask_parts = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            raw_target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
            raw_mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
            logits, normalized_prediction = model(iq)
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels.float().cpu())
            pred_parts.append(normalizer.inverse_transform(normalized_prediction.float().cpu()))
            target_parts.append(raw_target.float().cpu())
            mask_parts.append(raw_mask.cpu())
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    prediction = torch.cat(pred_parts).numpy()
    target = torch.cat(target_parts).numpy()
    mask = torch.cat(mask_parts).numpy().astype(bool)
    labels_np = labels.numpy()
    probability = torch.sigmoid(logits).numpy()
    oracle_valid = mask & (labels_np[:, :, None] > 0.5)
    predicted_gate = mask & (probability[:, :, None] >= threshold)
    end_to_end_valid = predicted_gate & (labels_np[:, :, None] > 0.5)
    metrics = multilabel_metrics(logits, labels, threshold)
    metrics.update({
        "macro_aupr": float(average_precision_score(labels_np, probability, average="macro")),
        "macro_f1": float(metrics["macro_f1"]),
        "parameter": parameter_metrics(prediction, target, oracle_valid, normalizer),
        "prediction": prediction,
        "target": target,
        "mask": mask,
        "labels": labels_np,
        "probability": probability,
        "oracle_valid": oracle_valid,
        "predicted_gate": predicted_gate,
        "end_to_end_valid": end_to_end_valid,
    })
    return metrics


def fmt_or_nan(value: object) -> str:
    number = float(value)
    return "nan" if not math.isfinite(number) else f"{number:.10g}"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def power_path(root: Path, level: int) -> Path:
    return root / f"level_{(level - 5) // 5:02d}__jnr_{level}dB" / "closedset_6class_train.mat"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--power-root", type=Path, default=Path(r"E:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("docs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--skip-power", action="store_true", help="nominal-only diagnostic; does not produce paired-power evidence")
    parser.add_argument("--allow-incomplete", action="store_true", help="diagnostic only; status remains partial")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "per_seed_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    coverage = {variant: len({int(row["seed"]) for row in summary_rows if row["variant"] == variant}) for variant in VARIANTS}
    complete = all(coverage[variant] >= 5 for variant in args.variants) and set(args.variants) == set(VARIANTS)
    if not complete and not args.allow_incomplete:
        raise RuntimeError(f"Refusing evaluation claim for incomplete R1 run: {coverage}")

    data_file = Path(args.data_file).resolve()
    labels = read_labels(data_file)
    split = np.load(run_dir / "split_indices.npz")
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    val_indices = np.asarray(split["val_indices"], dtype=np.int64)
    normalizer = fit_normalizer(data_file, train_indices, labels)
    device = get_device(args.device)
    width = int(protocol["width"])
    head_dim = int(protocol["head_dim"])
    parameter_hidden_dim = int(protocol["parameter_hidden_dim"])
    train_joint = V73IQDataset(data_file, val_indices, PARAMETER_COLUMNS)
    loader = DataLoader(train_joint, batch_size=args.batch_size, shuffle=False)
    parameter_rows, visibility_rows, power_rows, ablation_rows = [], [], [], []

    for variant in args.variants:
        for seed in sorted({int(row["seed"]) for row in summary_rows if row["variant"] == variant}):
            checkpoint_path = run_dir / f"{variant}_seed{seed}" / "best.pt"
            if not checkpoint_path.exists():
                continue
            if variant == "B0":
                model = build_benchmark_model("legacy_mlcnn", num_classes=6, width=width, head_dim=head_dim).to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                continue
            model_name = {"B1": "legacy_mlcnn_global", "B2": "legacy_mlcnn_class_conditioned", "B3": "legacy_mlcnn_class_conditioned_detached", "B4": "legacy_mlcnn_class_conditioned"}[variant]
            model = build_multitask_model(model_name, width=width, head_dim=head_dim, parameter_hidden_dim=parameter_hidden_dim).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            nominal = evaluate_joint(model, loader, device, normalizer, args.threshold)
            pmetrics = nominal["parameter"]["per_target"]
            for target_index, row in enumerate(pmetrics):
                class_index = CLASS_NAMES.index(row["class_name"])
                parameter_index = 0 if row["name"] == "JNR" else 1
                oracle_count = int(nominal["oracle_valid"][:, class_index, parameter_index].sum())
                predicted_count = int(nominal["predicted_gate"][:, class_index, parameter_index].sum())
                e2e_valid = nominal["end_to_end_valid"][:, class_index, parameter_index]
                error = np.abs(nominal["prediction"][e2e_valid, class_index, parameter_index] - nominal["target"][e2e_valid, class_index, parameter_index])
                scale = float(normalizer.scale[class_index, parameter_index])
                parameter_rows.append({
                    "variant": variant, "seed": seed, "power_role": "nominal_train_val", "jnr_dB": 30,
                    "class_name": row["class_name"], "parameter": row["name"], "unit": row["unit"],
                    "oracle_valid_count": oracle_count, "oracle_coverage": oracle_count / max(len(val_indices), 1),
                    "predicted_gate_count": predicted_count, "predicted_gate_rate": predicted_count / max(len(val_indices), 1),
                    "oracle_nmae": row["normalized_mae"], "oracle_mae": row["mae"], "oracle_rmse": row["rmse"],
                    "end_to_end_count": int(e2e_valid.sum()), "end_to_end_nmae": float(error.mean() / scale) if error.size else float("nan"),
                    "evidence_status": "verified_development" if complete else "partial_development",
                })
            for class_index, class_name in enumerate(CLASS_NAMES):
                oracle = nominal["oracle_valid"][:, class_index].sum()
                predicted = nominal["predicted_gate"][:, class_index].sum()
                active = nominal["labels"][:, class_index].sum()
                class_predicted = nominal["predicted_gate"][:, class_index].any(axis=1)
                safe_stop = (~class_predicted & ~(nominal["labels"][:, class_index] > 0.5)).mean()
                visibility_rows.append({
                    "variant": variant, "seed": seed, "power_role": "nominal_train_val", "jnr_dB": 30,
                    "class_name": class_name, "active_label_count": int(active), "oracle_parameter_valid_count": int(oracle),
                    "predicted_parameter_gate_count": int(predicted), "oracle_coverage": float(oracle / max(len(val_indices) * 2, 1)),
                    "predicted_gate_rate": float(predicted / max(len(val_indices) * 2, 1)), "safe_stop_rate_on_negative": float(safe_stop),
                    "evidence_status": "verified_development" if complete else "partial_development",
                })
            ablation_rows.append({"variant": variant, "seed": seed, "macro_aupr": nominal["macro_aupr"], "macro_f1": nominal["macro_f1"], "parameter_macro_nmae": nominal["parameter"]["oracle_macro_target_nmae"], "power_role": "nominal_train_val", "evidence_status": "verified_development" if complete else "partial_development"})

            for level in (() if args.skip_power else POWER_LEVELS):
                path = power_path(args.power_root.resolve(), level)
                if not path.is_file():
                    continue
                power_dataset = V73IQDataset(path, val_indices, PARAMETER_COLUMNS)
                power_loader = DataLoader(power_dataset, batch_size=args.batch_size, shuffle=False)
                power_eval = evaluate_joint(model, power_loader, device, normalizer, args.threshold)
                power_metrics = power_eval["parameter"]
                power_rows.append({
                    "variant": variant, "seed": seed, "jnr_dB": level,
                    "macro_aupr": power_eval["macro_aupr"], "macro_f1": power_eval["macro_f1"],
                    "parameter_macro_nmae": power_metrics["oracle_macro_target_nmae"],
                    "parameter_micro_nmae": power_metrics["oracle_micro_nmae"],
                    "valid_value_count": power_metrics["oracle_valid_values"],
                    "evidence_status": "verified_development" if complete else "partial_development",
                })
                power_dataset.close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    train_joint.close()

    output_root = args.output_root.resolve()
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "receipts").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "results" / "route_b_per_parameter_metrics.csv", parameter_rows, list(parameter_rows[0]) if parameter_rows else ["variant"])
    write_csv(output_root / "results" / "route_b_visibility_audit.csv", visibility_rows, list(visibility_rows[0]) if visibility_rows else ["variant"])
    write_csv(output_root / "results" / "route_b_paired_power_consistency.csv", power_rows, list(power_rows[0]) if power_rows else ["variant"])
    write_csv(output_root / "results" / "route_b_r3_ablation.csv", ablation_rows, list(ablation_rows[0]) if ablation_rows else ["variant"])
    write_csv(output_root / "results" / "route_b_r2_conditional_head.csv", [row for row in ablation_rows if row.get("variant") in ("B2", "B3", "B4")], list(ablation_rows[0]) if ablation_rows else ["variant"])
    payload = {
        "stage": "R3",
        "status": "verified_development" if complete else "partial_development",
        "run_dir": str(run_dir), "coverage": coverage, "device": str(device),
        "test_mat_opened": False, "locked_test_used": False,
        "rows": {"per_parameter": len(parameter_rows), "visibility": len(visibility_rows), "paired_power": len(power_rows), "ablation": len(ablation_rows)},
        "performance_claim": "development_only; no locked-test claim" if complete else "not_validated_incomplete_run",
    }
    (output_root / "receipts" / "route_b_r3_result_receipt.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_root / "reports" / "route_b_r3_evaluation.md").write_text(
        "# Route B R3 evaluation\n\n" + f"- Status: **{payload['status']}**\n- Run: `{run_dir}`\n- Device: `{device}`\n- Test MAT opened: `false`\n- Locked test used: `false`\n- Rows: `{payload['rows']}`\n\nPer-parameter, visibility, and paired-power tables are development-role results only; no locked-test claim is made.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
