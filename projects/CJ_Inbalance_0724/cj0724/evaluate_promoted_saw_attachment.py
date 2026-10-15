"""Attach the promoted SawFM period specialist to saved task-router predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cj0724.paper_core_evidence import gated_metrics


def _target_metrics(prediction: np.ndarray, target: np.ndarray, scale: float) -> dict[str, float]:
    error = np.abs(prediction - target)
    correlation = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) > 0 else 0.0
    return {"mae": float(error.mean()), "nmae": float(error.mean() / scale), "pearson_r": correlation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", default="outputs/paper_core_evidence")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--output", default="outputs/physics_parameter_specialists/saw_attachment_results.json")
    args = parser.parse_args()
    rows = []
    for seed in (2041, 2042, 2043):
        core = dict(np.load(Path(args.core_root) / f"mlcnn_iq_grouped_task_routed__seed{seed}.npz"))
        specialist = dict(np.load(Path(args.specialist_root) / "sawfm_period_sequence" / f"seed_{seed}" / "predictions.npz"))
        lookup = {int(index): position for position, index in enumerate(core["sample_indices"])}
        positions = np.asarray([lookup[int(index)] for index in specialist["sample_index"]], dtype=np.int64)
        if not np.array_equal(core["labels"][positions, 1], np.ones(len(positions), dtype=np.uint8)):
            raise RuntimeError("specialist samples are not SawFM-positive in the task-router record")
        scale = float(core["scale"][1, 1])
        base_target = core["target"][positions, 1, 1]
        base_prediction = core["prediction"][positions, 1, 1]
        fused = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in core.items()}
        fused["prediction"][positions, 1, 1] = specialist["prediction"]
        base_target_metrics = _target_metrics(base_prediction, base_target, scale)
        specialist_target_metrics = _target_metrics(specialist["prediction"], specialist["target"], scale)
        base_end_to_end = gated_metrics(core)
        fused_end_to_end = gated_metrics(fused)
        rows.append({
            "seed": seed,
            "base_sawfm_period": base_target_metrics,
            "specialist_sawfm_period": specialist_target_metrics,
            "sawfm_period_nmae_relative_reduction": float(1.0 - specialist_target_metrics["nmae"] / base_target_metrics["nmae"]),
            "base_end_to_end": base_end_to_end,
            "attached_end_to_end": fused_end_to_end,
            "oracle_parameter_nmae_delta": float(fused_end_to_end["oracle_parameter_nmae"] - base_end_to_end["oracle_parameter_nmae"]),
            "predicted_gate_parameter_nmae_delta": float(fused_end_to_end["predicted_gate_parameter_nmae"] - base_end_to_end["predicted_gate_parameter_nmae"]),
            "miss_penalty_parameter_nmae_delta": float(fused_end_to_end["miss_penalty_parameter_nmae"] - base_end_to_end["miss_penalty_parameter_nmae"]),
        })
    summary = {
        "attachment": "replace only SawFM sweep-period output; classification and all other parameter outputs frozen",
        "three_seed_same_direction_vs_task_router": bool(all(row["sawfm_period_nmae_relative_reduction"] > 0 for row in rows)),
        "sawfm_period_nmae_relative_reduction_mean": float(np.mean([row["sawfm_period_nmae_relative_reduction"] for row in rows])),
        "sawfm_period_pearson_r_base_mean": float(np.mean([row["base_sawfm_period"]["pearson_r"] for row in rows])),
        "sawfm_period_pearson_r_specialist_mean": float(np.mean([row["specialist_sawfm_period"]["pearson_r"] for row in rows])),
        "oracle_parameter_nmae_delta_mean": float(np.mean([row["oracle_parameter_nmae_delta"] for row in rows])),
        "predicted_gate_parameter_nmae_delta_mean": float(np.mean([row["predicted_gate_parameter_nmae_delta"] for row in rows])),
        "miss_penalty_parameter_nmae_delta_mean": float(np.mean([row["miss_penalty_parameter_nmae_delta"] for row in rows])),
        "classification_unchanged": True,
        "locked_test_used": False,
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
