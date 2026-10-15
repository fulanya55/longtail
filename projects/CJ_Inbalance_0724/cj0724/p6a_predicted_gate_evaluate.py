"""Frozen-checkpoint predicted-gate audit for P6a-r1 development confirm roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES
from cj0724.data import V73IQDataset
from cj0724.multitask import PARAMETER_COLUMNS, ParameterNormalizer, build_multitask_model, reshape_selected_parameters
from cj0724.p1k_joint_level3_system import worst_combo_predicted_gate_nmae


def _normalizer(state: dict[str, object]) -> ParameterNormalizer:
    normalizer = ParameterNormalizer()
    normalizer.mean = torch.tensor(state["mean"], dtype=torch.float32)
    normalizer.scale = torch.tensor(state["scale"], dtype=torch.float32)
    return normalizer


@torch.inference_mode()
def _record(model: torch.nn.Module, dataset: V73IQDataset, normalizer: ParameterNormalizer, device: torch.device) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=63, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")}
    model.eval()
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        logits, normalized_prediction = model(iq)
        pieces["sample_indices"].append(batch["sample_index"].numpy())
        pieces["labels"].append(batch["label"].numpy())
        pieces["probabilities"].append(torch.sigmoid(logits).float().cpu().numpy())
        pieces["prediction"].append(normalizer.inverse_transform(normalized_prediction.float().cpu()).numpy())
        pieces["target"].append(reshape_selected_parameters(batch["parameter_values"]).numpy())
        pieces["mask"].append(reshape_selected_parameters(batch["parameter_mask"]).numpy().astype(bool))
    return {name: np.concatenate(values) for name, values in pieces.items()}


def _metrics(record: dict[str, np.ndarray], thresholds: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    valid = record["mask"].astype(bool) & record["labels"].astype(bool)[:, :, None]
    predicted = record["probabilities"] >= thresholds[None, :]
    gated = valid & predicted[:, :, None]
    error = np.abs(record["prediction"] - record["target"]) / scale[None, :, :]
    result = {
        "oracle_gate_parameter_nmae": float(error[valid].mean()),
        "predicted_gate_parameter_nmae": float(error[gated].mean()),
        "miss_penalty_parameter_nmae": float(np.where(predicted[:, :, None], error, 1.0)[valid].mean()),
        "predicted_positive_parameter_coverage": float(gated.sum() / valid.sum()),
        "valid_parameter_cells": int(valid.sum()),
        "gated_parameter_cells": int(gated.sum()),
    }
    complete = {**record, "labels": record["labels"].astype(bool), "thresholds": thresholds, "scale": scale}
    result["worst_combo_predicted_gate_nmae"] = worst_combo_predicted_gate_nmae(complete)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    resolve = lambda value: Path(value) if Path(value).is_absolute() else project / value
    run_root, data_file = resolve(args.run_root), resolve(args.data_file)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    indices = np.load(run_root / "role_indices.npz")["confirm"]
    normalizer = _normalizer(protocol["normalizer"])
    dataset = V73IQDataset(data_file, indices, PARAMETER_COLUMNS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = ("mlcnn_iq_grouped_task_routed", "mlcnn_iq_grouped_physical_film_task_routed")
    results: dict[str, dict[str, object]] = {}
    try:
        for model_name in names:
            results[model_name] = {}
            for seed in (16041, 16042, 16043):
                folder = run_root / f"{model_name}__cos_lr1e4_wd1e4__seed{seed}"
                checkpoint = torch.load(folder / "best_parameter.pt", map_location=device, weights_only=False)
                receipt = json.loads((folder / "receipt_parameter.json").read_text(encoding="utf-8"))
                thresholds = np.asarray([receipt["thresholds"][name] for name in CLASS_NAMES], dtype=np.float32)
                model = build_multitask_model(model_name, 24, 0.2, 256, 64).to(device)
                model.load_state_dict(checkpoint["model_state"])
                values = _metrics(_record(model, dataset, normalizer, device), thresholds, normalizer.scale.numpy())
                results[model_name][str(seed)] = {"thresholds": thresholds.tolist(), **values}
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        dataset.close()
    baseline, candidate = (results[name] for name in names)
    pairs = []
    for seed in (16041, 16042, 16043):
        b, c = baseline[str(seed)], candidate[str(seed)]
        pairs.append({
            "seed": seed,
            "predicted_gate_nmae_reduction": float(1.0 - c["predicted_gate_parameter_nmae"] / b["predicted_gate_parameter_nmae"]),
            "miss_penalty_delta": float(c["miss_penalty_parameter_nmae"] - b["miss_penalty_parameter_nmae"]),
            "coverage_delta": float(c["predicted_positive_parameter_coverage"] - b["predicted_positive_parameter_coverage"]),
            "worst_combo_delta": float(c["worst_combo_predicted_gate_nmae"] - b["worst_combo_predicted_gate_nmae"]),
        })
    result = {"role": "P6a-r1 frozen development confirm only", "results": results, "paired_delta": pairs,
              "test_generated": False, "test_opened": False, "locked_test_used": False}
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
