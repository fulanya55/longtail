"""Diagnose P2a from immutable saved predictions; never open waveform files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cj0724 import CLASS_NAMES
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.multitask import PARAMETER_SPECS
from cj0724.paper_core_evidence import load_normalizer


MODEL_SEEDS = (2041, 2042, 2043)


def per_parameter_nmae(
    labels: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    predicted = np.asarray(probabilities) >= np.asarray(thresholds)[None, :]
    normalized = np.abs(np.asarray(prediction) - np.asarray(target)) / np.asarray(scale)[None, :, :]
    output = np.full((6, 2), np.nan, dtype=np.float64)
    for class_index in range(6):
        for value_index in range(2):
            valid = (
                labels[:, class_index]
                & predicted[:, class_index]
                & mask[:, class_index, value_index]
                & np.isfinite(normalized[:, class_index, value_index])
            )
            if valid.any():
                output[class_index, value_index] = float(
                    normalized[valid, class_index, value_index].mean()
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--output-root", default="outputs/p2a_nine_power_generalization_20260731")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run_root, output_root = project / args.run_root, project / args.output_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    scale = load_normalizer(protocol["normalizer"]).scale.numpy()

    per_seed = []
    parameter_rows = []
    power_rows = []
    for seed in MODEL_SEEDS:
        with np.load(output_root / f"seed_{seed}" / "test_per_sample.npz") as saved:
            arrays = {name: saved[name] for name in saved.files}
        route_metric = per_parameter_nmae(
            arrays["labels"], arrays["mask"], arrays["routed_probability"],
            arrays["routed_thresholds"], arrays["base_prediction"], arrays["target"], scale,
        )
        joint_metric = per_parameter_nmae(
            arrays["labels"], arrays["mask"], arrays["routed_probability"],
            arrays["routed_thresholds"], arrays["joint_prediction"], arrays["target"], scale,
        )
        for flat_index, spec in enumerate(PARAMETER_SPECS):
            class_index, value_index = divmod(flat_index, 2)
            parameter_rows.append({
                "seed": seed, "class": spec.class_name, "parameter": spec.name,
                "safe_route_nmae": float(route_metric[class_index, value_index]),
                "joint_nmae": float(joint_metric[class_index, value_index]),
                "delta": float(joint_metric[class_index, value_index] - route_metric[class_index, value_index]),
            })
        seed_power = []
        for power in sorted(np.unique(arrays["power_db"]).astype(int)):
            selected = arrays["power_db"] == power
            route_power = per_parameter_nmae(
                arrays["labels"][selected], arrays["mask"][selected], arrays["routed_probability"][selected],
                arrays["routed_thresholds"], arrays["base_prediction"][selected], arrays["target"][selected], scale,
            )
            joint_power = per_parameter_nmae(
                arrays["labels"][selected], arrays["mask"][selected], arrays["routed_probability"][selected],
                arrays["routed_thresholds"], arrays["joint_prediction"][selected], arrays["target"][selected], scale,
            )
            route_macro, joint_macro = float(np.nanmean(route_power)), float(np.nanmean(joint_power))
            row = {
                "seed": seed, "power_db": int(power),
                "safe_route_macro_nmae": route_macro, "joint_macro_nmae": joint_macro,
                "relative_improvement": float(1.0 - joint_macro / route_macro),
                "sawfm_period_delta": float(joint_power[1, 1] - route_power[1, 1]),
                "vft_doppler_delta": float(joint_power[4, 1] - route_power[4, 1]),
            }
            power_rows.append(row)
            seed_power.append(row)
        per_seed.append({
            "seed": seed,
            "mean_power_relative_improvement": float(np.mean([row["relative_improvement"] for row in seed_power])),
            "minimum_power_relative_improvement": float(min(row["relative_improvement"] for row in seed_power)),
        })

    parameter_aggregate = []
    for spec in PARAMETER_SPECS:
        rows = [row for row in parameter_rows if row["class"] == spec.class_name and row["parameter"] == spec.name]
        parameter_aggregate.append({
            "class": spec.class_name, "parameter": spec.name,
            "mean_safe_route_nmae": float(np.mean([row["safe_route_nmae"] for row in rows])),
            "mean_joint_nmae": float(np.mean([row["joint_nmae"] for row in rows])),
            "mean_delta": float(np.mean([row["delta"] for row in rows])),
            "changed_by_p1n": bool(any(abs(row["delta"]) > 1e-10 for row in rows)),
        })
    power_aggregate = []
    for power in sorted({row["power_db"] for row in power_rows}):
        rows = [row for row in power_rows if row["power_db"] == power]
        power_aggregate.append({
            "power_db": power,
            "mean_relative_improvement": float(np.mean([row["relative_improvement"] for row in rows])),
            "mean_sawfm_period_delta": float(np.mean([row["sawfm_period_delta"] for row in rows])),
            "mean_vft_doppler_delta": float(np.mean([row["vft_doppler_delta"] for row in rows])),
        })

    output = {
        "experiment": "P2a failure diagnostics from immutable predictions",
        "statistics_source": "seed_*/test_per_sample.npz; no waveform access",
        "per_seed": per_seed,
        "per_parameter": parameter_aggregate,
        "per_power": power_aggregate,
        "diagnosis": {
            "changed_parameter_count": int(sum(row["changed_by_p1n"] for row in parameter_aggregate)),
            "unchanged_parameter_count": int(sum(not row["changed_by_p1n"] for row in parameter_aggregate)),
            "high_power_relative_gain_declines": bool(
                power_aggregate[-1]["mean_relative_improvement"]
                < power_aggregate[0]["mean_relative_improvement"]
            ),
        },
        "historical_test_reread": False,
        "waveform_reread": False,
    }
    (output_root / "failure_diagnostics.json").write_text(
        json.dumps(_jsonable(output), indent=2), encoding="utf-8",
    )
    with (output_root / "failure_diagnostics_by_power.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(power_aggregate[0]))
        writer.writeheader()
        writer.writerows(power_aggregate)
    print(json.dumps(_jsonable(output), indent=2))


if __name__ == "__main__":
    main()
