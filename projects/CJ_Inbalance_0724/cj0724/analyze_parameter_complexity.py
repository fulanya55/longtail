"""Audit parameter checkpoints by label cardinality on the frozen confirm role."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.multitask import PARAMETER_COLUMNS, ParameterNormalizer, build_multitask_model, parameter_metrics
from cj0724.tuned_multitask_sweep import _collect_outputs
from cj0724.train import get_device


def _normalizer(state: dict[str, object]) -> ParameterNormalizer:
    result = ParameterNormalizer()
    result.mean = torch.tensor(state["mean"], dtype=torch.float32)
    result.scale = torch.tensor(state["scale"], dtype=torch.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--checkpoint-objective", default="parameter")
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = get_device(args.device)
    rows = []
    baseline_rows = []
    for root_text in args.run_roots:
        root = Path(root_text)
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        role_indices = np.load(root / "role_indices.npz")["confirm"]
        normalizer = _normalizer(protocol["normalizer"])
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        selected = [row for row in summary if row.get("checkpoint_objective", "joint") == args.checkpoint_objective]
        dataset = V73IQDataset(protocol["data_file"], role_indices, PARAMETER_COLUMNS)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        for summary_row in selected:
            model = build_multitask_model(
                summary_row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
                protocol["parameter_hidden_dim"], protocol["legacy_attention"],
            ).to(device)
            checkpoint = torch.load(summary_row["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            outputs = _collect_outputs(model, loader, device, normalizer, protocol["parameter_weight"])
            labels = outputs["labels"].numpy()
            valid = outputs["mask"] & (labels[:, :, None] > 0.5)
            cardinality = labels.sum(axis=1).astype(np.int64)
            for count in sorted(np.unique(cardinality)):
                selected_samples = cardinality == count
                metrics = parameter_metrics(
                    outputs["prediction"][selected_samples], outputs["target"][selected_samples],
                    valid[selected_samples], normalizer,
                )
                rows.append({
                    "model": summary_row["model"], "seed": summary_row["seed"],
                    "checkpoint_objective": args.checkpoint_objective,
                    "label_cardinality": int(count), "sample_count": int(selected_samples.sum()),
                    "valid_parameter_count": metrics["oracle_valid_values"],
                    "parameter_macro_nmae": metrics["oracle_macro_target_nmae"],
                    "parameter_macro_pearson": metrics["oracle_macro_pearson_r"],
                })
            constant = np.broadcast_to(normalizer.mean.numpy(), outputs["target"].shape)
            baseline = parameter_metrics(constant, outputs["target"], valid, normalizer)
            baseline_rows.append({
                "model": summary_row["model"], "seed": summary_row["seed"],
                "constant_macro_nmae": baseline["oracle_macro_target_nmae"],
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "constant_baselines": baseline_rows, "locked_test_used": False}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
