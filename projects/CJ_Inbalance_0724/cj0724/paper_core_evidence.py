"""Build paired-bootstrap and end-to-end gated parameter evidence from saved runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.multitask import PARAMETER_COLUMNS, ParameterNormalizer, build_multitask_model
from cj0724.tuned_multitask_sweep import _collect_outputs
from cj0724.train import get_device


def load_normalizer(state: dict[str, object]) -> ParameterNormalizer:
    normalizer = ParameterNormalizer()
    normalizer.mean = torch.tensor(state["mean"], dtype=torch.float32)
    normalizer.scale = torch.tensor(state["scale"], dtype=torch.float32)
    return normalizer


def macro_nmae(prediction, target, valid, scale) -> float:
    values = []
    for class_index in range(valid.shape[1]):
        for parameter_index in range(valid.shape[2]):
            selected = valid[:, class_index, parameter_index]
            if selected.any():
                error = np.abs(
                    prediction[selected, class_index, parameter_index]
                    - target[selected, class_index, parameter_index]
                ) / scale[class_index, parameter_index]
                values.append(float(error.mean()))
    return float(np.mean(values))


def gated_metrics(record: dict[str, np.ndarray]) -> dict[str, float]:
    labels = record["labels"].astype(bool)
    probabilities = record["probabilities"]
    predicted = probabilities >= record["thresholds"][None, :]
    oracle_valid = record["mask"].astype(bool) & labels[:, :, None]
    gated_valid = oracle_valid & predicted[:, :, None]
    scale = record["scale"]
    normalized_error = np.abs(record["prediction"] - record["target"]) / scale[None, :, :]
    coverage = float(gated_valid.sum() / max(oracle_valid.sum(), 1))
    gated_nmae = macro_nmae(record["prediction"], record["target"], gated_valid, scale)
    penalized = np.where(predicted[:, :, None], normalized_error, 1.0)
    penalty_prediction = record["target"] + penalized * scale[None, :, :]
    penalty_nmae = macro_nmae(penalty_prediction, record["target"], oracle_valid, scale)
    negatives = ~labels
    false_gate_rate = float(predicted[negatives].mean())
    exact_gate_accuracy = float(np.all(predicted == labels, axis=1).mean())
    return {
        "classification_macro_aupr": float(average_precision_score(labels, probabilities, average="macro")),
        "predicted_gate_parameter_coverage": coverage,
        "predicted_gate_parameter_nmae": gated_nmae,
        "miss_penalty_parameter_nmae": penalty_nmae,
        "false_positive_gate_rate": false_gate_rate,
        "exact_label_gate_accuracy": exact_gate_accuracy,
        "oracle_parameter_nmae": macro_nmae(
            record["prediction"], record["target"], oracle_valid, scale
        ),
    }


def bootstrap_delta(
    candidate: dict[int, dict[str, np.ndarray]],
    baseline: dict[int, dict[str, np.ndarray]],
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    seeds = sorted(set(candidate) & set(baseline))
    deltas = []
    observed = []
    for experiment_seed in seeds:
        cm = gated_metrics(candidate[experiment_seed])[metric]
        bm = gated_metrics(baseline[experiment_seed])[metric]
        observed.append(cm - bm)
    for _ in range(replicates):
        seed_deltas = []
        for experiment_seed in seeds:
            cand = candidate[experiment_seed]
            base = baseline[experiment_seed]
            if not np.array_equal(cand["sample_indices"], base["sample_indices"]):
                raise RuntimeError("paired bootstrap requires identical ordered sample indices")
            indices = rng.integers(0, len(cand["sample_indices"]), len(cand["sample_indices"]))
            cand_resampled = {
                key: value[indices] if isinstance(value, np.ndarray) and value.shape[:1] == (len(indices),) else value
                for key, value in cand.items()
            }
            base_resampled = {
                key: value[indices] if isinstance(value, np.ndarray) and value.shape[:1] == (len(indices),) else value
                for key, value in base.items()
            }
            seed_deltas.append(gated_metrics(cand_resampled)[metric] - gated_metrics(base_resampled)[metric])
        deltas.append(float(np.mean(seed_deltas)))
    values = np.asarray(deltas)
    return {
        "observed_mean_delta": float(np.mean(observed)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_probability_delta_positive": float((values > 0).mean()),
        "replicates": int(replicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--checkpoint-objective", default="parameter")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    device = get_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)
    seen = set()
    for root_text in args.run_roots:
        root = Path(root_text)
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        roles = np.load(root / "role_indices.npz")
        confirm_indices = roles["confirm"].astype(np.int64)
        normalizer = load_normalizer(protocol["normalizer"])
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        selected_rows = [
            row for row in summary
            if row["model"] in args.models
            and row.get("checkpoint_objective", "joint") == args.checkpoint_objective
        ]
        if not selected_rows:
            continue
        dataset = V73IQDataset(protocol["data_file"], confirm_indices, PARAMETER_COLUMNS)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        for row in selected_rows:
            key = (row["model"], int(row["seed"]))
            if key in seen:
                continue
            seen.add(key)
            model = build_multitask_model(
                row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
                protocol["parameter_hidden_dim"], protocol["legacy_attention"],
            ).to(device)
            checkpoint = torch.load(row["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            outputs = _collect_outputs(model, loader, device, normalizer, protocol["parameter_weight"])
            record = {
                "sample_indices": confirm_indices,
                "labels": outputs["labels"].numpy().astype(np.uint8),
                "probabilities": torch.sigmoid(outputs["logits"]).numpy(),
                "prediction": outputs["prediction"],
                "target": outputs["target"],
                "mask": outputs["mask"].astype(np.uint8),
                "thresholds": np.asarray(json.loads(row["thresholds"]), dtype=np.float32),
                "scale": normalizer.scale.numpy(),
            }
            records[row["model"]][int(row["seed"])] = record
            np.savez_compressed(output_root / f"{row['model']}__seed{row['seed']}.npz", **record)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset.close()
    metrics = {
        model: {str(seed): gated_metrics(record) for seed, record in by_seed.items()}
        for model, by_seed in records.items()
    }
    candidate = "mlcnn_iq_grouped_task_routed"
    comparisons = {}
    for baseline in ("iq_grouped", "legacy_mlcnn_global"):
        comparisons[f"{candidate}_minus_{baseline}"] = {
            metric: bootstrap_delta(
                records[candidate], records[baseline], metric, args.replicates,
                args.bootstrap_seed + offset,
            )
            for offset, metric in enumerate((
                "classification_macro_aupr", "oracle_parameter_nmae",
                "predicted_gate_parameter_coverage", "predicted_gate_parameter_nmae",
                "miss_penalty_parameter_nmae", "false_positive_gate_rate",
            ))
        }
    result = {
        "protocol": {
            "checkpoint_objective": args.checkpoint_objective,
            "bootstrap_unit": "paired confirm sample within seed; mean delta across three seeds",
            "miss_penalty": "normalized absolute error=1 for an oracle-valid parameter whose class gate is missed",
            "locked_test_used": False,
        },
        "metrics": metrics,
        "paired_bootstrap": comparisons,
    }
    (output_root / "paper_core_evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
