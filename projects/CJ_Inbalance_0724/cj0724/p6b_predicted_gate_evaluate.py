"""Evaluate frozen P6b checkpoints on their development confirm role only."""

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
def _collect(model, dataset, normalizer, device):
    loader = DataLoader(dataset, batch_size=63, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    output = {key: [] for key in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")}
    model.eval()
    for batch in loader:
        logits, predicted = model(batch["iq"].to(device, non_blocking=True))
        output["sample_indices"].append(batch["sample_index"].numpy())
        output["labels"].append(batch["label"].numpy().astype(bool))
        output["probabilities"].append(torch.sigmoid(logits).float().cpu().numpy())
        output["prediction"].append(normalizer.inverse_transform(predicted.float().cpu()).numpy())
        output["target"].append(reshape_selected_parameters(batch["parameter_values"]).numpy())
        output["mask"].append(reshape_selected_parameters(batch["parameter_mask"]).numpy().astype(bool))
    return {key: np.concatenate(values) for key, values in output.items()}


def _evaluate(record, threshold, scale):
    valid = record["mask"] & record["labels"][:, :, None]
    decision = record["probabilities"] >= threshold[None, :]
    gated = valid & decision[:, :, None]
    error = np.abs(record["prediction"] - record["target"]) / scale[None, :, :]
    full = {**record, "thresholds": threshold, "scale": scale}
    return {
        "oracle_gate_parameter_nmae": float(error[valid].mean()),
        "predicted_gate_parameter_nmae": float(error[gated].mean()),
        "miss_penalty_parameter_nmae": float(np.where(decision[:, :, None], error, 1.0)[valid].mean()),
        "coverage": float(gated.sum() / valid.sum()),
        "worst_combo_predicted_gate_nmae": worst_combo_predicted_gate_nmae(full),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    resolve = lambda value: Path(value) if Path(value).is_absolute() else project / value
    root, data_path = resolve(args.run_root), resolve(args.data_file)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    normalizer = _normalizer(protocol["normalizer"])
    dataset = V73IQDataset(data_path, np.load(root / "role_indices.npz")["confirm"], PARAMETER_COLUMNS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = ("mlcnn_iq_grouped_task_routed", "mlcnn_iq_grouped_temporal_query_task_routed", "mlcnn_iq_grouped_temporal_global_control_task_routed")
    results = {}
    try:
        for name in names:
            results[name] = {}
            for seed in (17041, 17042, 17043):
                folder = root / f"{name}__cos_lr1e4_wd1e4__seed{seed}"
                checkpoint = torch.load(folder / "best_parameter.pt", map_location=device, weights_only=False)
                receipt = json.loads((folder / "receipt_parameter.json").read_text(encoding="utf-8"))
                threshold = np.asarray([receipt["thresholds"][class_name] for class_name in CLASS_NAMES], dtype=np.float32)
                model = build_multitask_model(name, 24, 0.2, 256, 64).to(device)
                model.load_state_dict(checkpoint["model_state"])
                results[name][str(seed)] = {"thresholds": threshold.tolist(), **_evaluate(_collect(model, dataset, normalizer, device), threshold, normalizer.scale.numpy())}
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        dataset.close()
    c0, c1, c2 = (results[name] for name in names)
    paired = []
    for seed in (17041, 17042, 17043):
        baseline, query, control = (rows[str(seed)] for rows in (c0, c1, c2))
        paired.append({
            "seed": seed,
            "C1_vs_C0_predicted_gate_reduction": float(1.0 - query["predicted_gate_parameter_nmae"] / baseline["predicted_gate_parameter_nmae"]),
            "C1_vs_C2_predicted_gate_reduction": float(1.0 - query["predicted_gate_parameter_nmae"] / control["predicted_gate_parameter_nmae"]),
            "C1_vs_C0_miss_penalty_delta": float(query["miss_penalty_parameter_nmae"] - baseline["miss_penalty_parameter_nmae"]),
            "C1_vs_C0_worst_combo_delta": float(query["worst_combo_predicted_gate_nmae"] - baseline["worst_combo_predicted_gate_nmae"]),
        })
    result = {"role": "P6b frozen development confirm only", "results": results, "paired_delta": paired,
              "test_generated": False, "test_opened": False, "locked_test_used": False}
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
