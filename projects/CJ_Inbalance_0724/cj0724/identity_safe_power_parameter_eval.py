"""Evaluate the frozen task router and promoted SawFM expert on paired power identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset, read_labels
from cj0724.dual_axis import POWER_LEVELS, audit_power_alignment, discover_power_train_files, read_identity_metadata
from cj0724.multitask import PARAMETER_COLUMNS, ParameterNormalizer, build_multitask_model
from cj0724.paper_core_evidence import gated_metrics, load_normalizer, macro_nmae
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.run_physics_parameter_specialists import _positive, _stratified_limit


def _identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _collect(model, loader, device, normalizer):
    logits, prediction, labels, targets, masks, indices = [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch_logits, batch_prediction = model(batch["iq"].to(device))
            logits.append(batch_logits.float().cpu())
            prediction.append(normalizer.inverse_transform(batch_prediction.float()).cpu())
            labels.append(batch["label"].cpu())
            targets.append(batch["parameter_values"].reshape(-1, 6, 2).cpu())
            masks.append(batch["parameter_mask"].reshape(-1, 6, 2).cpu())
            indices.append(batch["sample_index"].cpu())
    return {
        "logits": torch.cat(logits).numpy(), "prediction": torch.cat(prediction).numpy(),
        "labels": torch.cat(labels).numpy(), "target": torch.cat(targets).numpy(),
        "mask": torch.cat(masks).numpy(), "sample_indices": torch.cat(indices).numpy(),
    }


@torch.inference_mode()
def _saw_prediction(model, loader, device, mean: float, scale: float) -> np.ndarray:
    output = []
    model.eval()
    for batch in loader:
        normalized = model(batch["iq"].to(device))["prediction"]
        output.append((normalized * scale + mean).float().cpu().numpy())
    return np.concatenate(output)


def _weak_mask(labels: np.ndarray, level: int) -> np.ndarray:
    weak = np.zeros_like(labels, dtype=bool)
    if level < 30:
        eligible = labels[:, :3].any(axis=1)
        weak[eligible, 3:] = labels[eligible, 3:].astype(bool)
    elif level > 30:
        eligible = labels[:, 3:].any(axis=1)
        weak[eligible, :3] = labels[eligible, :3].astype(bool)
    return weak


def _level_metrics(record: dict[str, np.ndarray], weak: np.ndarray) -> dict[str, float]:
    metrics = gated_metrics(record)
    predicted = record["probabilities"] >= record["thresholds"][None, :]
    if weak.any():
        metrics["weak_component_recall"] = float(predicted[weak].mean())
        weak_valid = record["mask"].astype(bool) & weak[:, :, None]
        metrics["weak_oracle_parameter_nmae"] = macro_nmae(
            record["prediction"], record["target"], weak_valid, record["scale"]
        )
        weak_gated = weak_valid & predicted[:, :, None]
        metrics["weak_predicted_gate_parameter_coverage"] = float(weak_gated.sum() / max(weak_valid.sum(), 1))
        metrics["weak_predicted_gate_parameter_nmae"] = macro_nmae(
            record["prediction"], record["target"], weak_gated, record["scale"]
        )
    else:
        metrics.update({
            "weak_component_recall": float("nan"), "weak_oracle_parameter_nmae": float("nan"),
            "weak_predicted_gate_parameter_coverage": float("nan"),
            "weak_predicted_gate_parameter_nmae": float("nan"),
        })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/identity_safe_power_parameter_eval")
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root, specialist_root, output_root = Path(args.run_root), Path(args.specialist_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    roles = np.load(run_root / "role_indices.npz")
    confirm_indices = roles["confirm"].astype(np.int64)
    normalizer = load_normalizer(protocol["normalizer"])
    files = discover_power_train_files(args.power_root)
    alignment = audit_power_alignment(files)
    root_seeds, root_labels = read_identity_metadata(protocol["data_file"])
    power_seeds, power_labels = read_identity_metadata(files[30])
    if not np.array_equal(root_seeds, power_seeds) or not np.array_equal(root_labels, power_labels):
        raise RuntimeError("root train and paired power identities are not aligned")
    with h5py.File(protocol["data_file"], "r") as handle:
        parameter_values = np.asarray(handle["parameter_values"], dtype=np.float64)
        parameter_mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    selected_rows = [
        row for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"
    ]
    rows = []
    for row in selected_rows:
        seed = int(row["seed"])
        model = build_multitask_model(
            row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
            protocol["parameter_hidden_dim"], protocol["legacy_attention"],
        ).to(device)
        checkpoint = torch.load(row["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        saw.load_state_dict(torch.load(specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt", map_location=device, weights_only=True))
        train_positive = _positive(roles["train"], root_labels, parameter_mask, 1, 14)
        train_indices = _stratified_limit(train_positive, parameter_values[14], 2048, seed + 101, "scalar")
        saw_mean = float(np.mean(parameter_values[14, train_indices]))
        saw_scale = float(np.std(parameter_values[14, train_indices]))
        thresholds = np.asarray(json.loads(row["thresholds"]), dtype=np.float32)
        for level in POWER_LEVELS:
            dataset = V73IQDataset(files[level], confirm_indices, PARAMETER_COLUMNS)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
            base = _collect(model, loader, device, normalizer)
            probability = torch.sigmoid(torch.from_numpy(base["logits"])).numpy()
            common = {
                "sample_indices": base["sample_indices"], "labels": base["labels"].astype(np.uint8),
                "probabilities": probability, "target": base["target"], "mask": base["mask"].astype(np.uint8),
                "thresholds": thresholds, "scale": normalizer.scale.numpy(),
            }
            weak = _weak_mask(base["labels"].astype(bool), level)
            base_record = {**common, "prediction": base["prediction"]}
            base_metrics = _level_metrics(base_record, weak)
            saw_valid = base["mask"][:, 1, 1].astype(bool) & base["labels"][:, 1].astype(bool)
            base_metrics["sawfm_period_nmae"] = float((np.abs(base["prediction"][saw_valid, 1, 1] - base["target"][saw_valid, 1, 1]) / normalizer.scale[1, 1].item()).mean())
            if weak[:, 1].any():
                selected = saw_valid & weak[:, 1]
                base_metrics["weak_sawfm_period_nmae"] = float((np.abs(base["prediction"][selected, 1, 1] - base["target"][selected, 1, 1]) / normalizer.scale[1, 1].item()).mean())
            else:
                base_metrics["weak_sawfm_period_nmae"] = float("nan")
            rows.append({"method": "task_routed", "seed": seed, "power_level_db": level, **base_metrics})
            saw_values = _saw_prediction(saw, loader, device, saw_mean, saw_scale)
            fused_prediction = base["prediction"].copy()
            fused_prediction[:, 1, 1] = saw_values
            fused_record = {**common, "prediction": fused_prediction}
            fused_metrics = _level_metrics(fused_record, weak)
            saw_error = np.abs(fused_prediction[saw_valid, 1, 1] - base["target"][saw_valid, 1, 1]) / normalizer.scale[1, 1].item()
            fused_metrics["sawfm_period_nmae"] = float(saw_error.mean())
            if weak[:, 1].any():
                selected = saw_valid & weak[:, 1]
                fused_metrics["weak_sawfm_period_nmae"] = float((np.abs(fused_prediction[selected, 1, 1] - base["target"][selected, 1, 1]) / normalizer.scale[1, 1].item()).mean())
            else:
                fused_metrics["weak_sawfm_period_nmae"] = float("nan")
            rows.append({"method": "task_routed_plus_saw_period", "seed": seed, "power_level_db": level, **fused_metrics})
            dataset.close()
        del model, saw
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate = []
    for method in ("task_routed", "task_routed_plus_saw_period"):
        for level in POWER_LEVELS:
            selected = [row for row in rows if row["method"] == method and row["power_level_db"] == level]
            numeric_keys = [key for key, value in selected[0].items() if key not in ("method", "seed", "power_level_db") and isinstance(value, (int, float))]
            mean_values, std_values = {}, {}
            for key in numeric_keys:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                mean_values[f"{key}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                std_values[f"{key}_std"] = float(np.nanstd(values)) if np.isfinite(values).any() else float("nan")
            aggregate.append({
                "method": method, "power_level_db": level,
                **mean_values, **std_values,
            })
    payload = {
        "protocol": {
            "evaluation_role": "root-confirm identities only", "identity_key": "receiver_noise_seed",
            "power_views_per_identity": list(POWER_LEVELS), "specialist_attachment": "SawFM period output replacement only",
            "mft_and_vft_attachment": "rejected by frozen promotion gate", "locked_test_used": False,
        },
        "identity_audit": alignment,
        "root_identity_sha256": _identity_hash(root_seeds),
        "confirm_identity_sha256": _identity_hash(root_seeds[confirm_indices]),
        "per_seed_level": rows, "aggregate": aggregate,
    }
    (output_root / "identity_safe_power_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
