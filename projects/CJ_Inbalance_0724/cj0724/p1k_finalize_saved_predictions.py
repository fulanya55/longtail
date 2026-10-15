"""Finalize P1k statistics from immutable saved test predictions.

This script never opens waveform files.  It reconstructs the four frozen
variants from ``test_per_sample.npz`` and performs identity-grouped paired
bootstrap with vectorized parameter metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import ISRJ_GLOBAL, POWERS
from cj0724.p1k_joint_level3_system import (
    MODEL_SEEDS,
    VFT_CLASS_INDEX,
    evaluate_variant,
    variant_record,
)
from cj0724.paper_core_evidence import load_normalizer


def fast_metric(record: dict[str, np.ndarray], metric: str) -> float:
    labels = record["labels"].astype(bool)
    probabilities = record["probabilities"]
    predicted = probabilities >= record["thresholds"][None, :]
    if metric == "classification_macro_aupr":
        return float(average_precision_score(labels, probabilities, average="macro"))
    if metric == "isrj_recall":
        positive = labels[:, ISRJ_GLOBAL]
        return float(predicted[positive, ISRJ_GLOBAL].mean())
    oracle = record["mask"].astype(bool) & labels[:, :, None]
    normalized = np.abs(record["prediction"] - record["target"]) / record["scale"][None, :, :]
    if metric == "predicted_gate_parameter_nmae":
        valid = oracle & predicted[:, :, None]
        values = [float(normalized[:, c, p][valid[:, c, p]].mean())
                  for c in range(6) for p in range(2) if valid[:, c, p].any()]
        return float(np.mean(values))
    if metric == "miss_penalty_parameter_nmae":
        penalized = np.where(predicted[:, :, None], normalized, 1.0)
        values = [float(penalized[:, c, p][oracle[:, c, p]].mean())
                  for c in range(6) for p in range(2) if oracle[:, c, p].any()]
        return float(np.mean(values))
    raise KeyError(metric)


def grouped_bootstrap(
    rows: list[dict[str, object]], candidate: str, baseline: str,
    metrics: tuple[str, ...], replicates: int, seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    observed = {
        metric: float(np.mean([
            fast_metric(row["records"][candidate], metric)
            - fast_metric(row["records"][baseline], metric)
            for row in rows
        ]))
        for metric in metrics
    }
    samples = {metric: [] for metric in metrics}
    for _ in range(replicates):
        replicate = {metric: [] for metric in metrics}
        for row in rows:
            identities = np.asarray(row["identity"])
            unique, inverse = np.unique(identities, return_inverse=True)
            sampled = rng.integers(0, len(unique), size=len(unique))
            sampled_rows = np.concatenate([np.flatnonzero(inverse == value) for value in sampled])
            subset = {}
            for name in (candidate, baseline):
                source = row["records"][name]
                subset[name] = {
                    key: value[sampled_rows]
                    if isinstance(value, np.ndarray) and value.shape[:1] == (len(identities),)
                    else value
                    for key, value in source.items()
                }
            for metric in metrics:
                replicate[metric].append(
                    fast_metric(subset[candidate], metric) - fast_metric(subset[baseline], metric)
                )
        for metric in metrics:
            samples[metric].append(float(np.mean(replicate[metric])))
    output = {}
    for metric in metrics:
        values = np.asarray(samples[metric], dtype=np.float64)
        output[metric] = {
            "observed_mean_delta": observed[metric],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_delta_positive": float((values > 0).mean()),
            "replicates": int(replicates),
        }
    return output


def checkpoint_parameter_count(path: Path, nested: bool) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if nested:
        state = state["model_state"]
    return int(sum(value.numel() for value in state.values() if torch.is_tensor(value)))


def nine_power_recall_gates(per_power_gain: dict[str, float]) -> dict[str, bool]:
    values = np.asarray(list(per_power_gain.values()), dtype=np.float64)
    if len(values) != 9:
        return {}
    return {
        "isrj_at_least_eight_of_nine_powers_nonnegative": bool((values >= 0).sum() >= 8),
        "no_power_recall_decline_below_minus_0p02": bool(values.min() >= -0.02),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--output-root", default="outputs/p1k_joint_level3_system_20260801")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root, output_root = project / args.run_root, project / args.output_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    normalizer = load_normalizer(protocol["normalizer"])
    scale = normalizer.scale.numpy()
    freeze = json.loads((output_root / "method_freeze.json").read_text(encoding="utf-8"))
    powers = tuple(int(power) for power in freeze.get("powers_db", POWERS))
    freeze_by_seed = {int(row["seed"]): row for row in freeze["seed_freezes"]}
    results, bootstrap_rows = [], []
    for seed in MODEL_SEEDS:
        seed_freeze = freeze_by_seed[seed]
        payload_file = output_root / f"seed_{seed}" / "test_per_sample.npz"
        with np.load(payload_file) as saved:
            common = {name: saved[name] for name in (
                "sample_index", "receiver_noise_seed", "power_db", "labels", "target", "mask",
                "original_probability", "routed_probability", "base_prediction", "joint_prediction",
                "original_thresholds", "safe_thresholds", "routed_thresholds",
            )}
        payload = {
            "sample_indices": common["sample_index"], "power": common["power_db"],
            "labels": common["labels"], "target": common["target"], "mask": common["mask"],
        }
        variants_array = {
            "original_iq_grouped": (common["original_probability"], common["original_thresholds"], common["base_prediction"]),
            "safe_calibrated_iq_grouped": (common["original_probability"], common["safe_thresholds"], common["base_prediction"]),
            "safe_isrj_route": (common["routed_probability"], common["routed_thresholds"], common["base_prediction"]),
            "joint_physics_system": (common["routed_probability"], common["routed_thresholds"], common["joint_prediction"]),
        }
        variants = {
            name: evaluate_variant(payload, probability, thresholds, prediction, scale, powers)
            for name, (probability, thresholds, prediction) in variants_array.items()
        }
        records = {
            name: variant_record(payload, probability, thresholds, prediction, scale)
            for name, (probability, thresholds, prediction) in variants_array.items()
        }
        prior_result = json.loads((output_root / f"seed_{seed}" / "test_result.json").read_text(encoding="utf-8"))
        result = {
            "seed": seed, "variants": variants, "route_accept_coverage": prior_result["route_accept_coverage"],
            "timing": prior_result["timing"], "method_freeze_sha256": prior_result["method_freeze_sha256"],
            "identity_audit": prior_result["identity_audit"], "test_used_once": True,
            "test_fit_or_selection": False, "confirm_used": False, "locked_test_used": False,
        }
        results.append(result)
        bootstrap_rows.append({"identity": common["receiver_noise_seed"], "records": records})

    comparisons = {
        "safe_route_vs_safe_calibrated": grouped_bootstrap(
            bootstrap_rows, "safe_isrj_route", "safe_calibrated_iq_grouped",
            ("classification_macro_aupr", "isrj_recall"), args.bootstrap_replicates, 20260810,
        ),
        "joint_vs_original": grouped_bootstrap(
            bootstrap_rows, "joint_physics_system", "original_iq_grouped",
            ("classification_macro_aupr", "predicted_gate_parameter_nmae", "miss_penalty_parameter_nmae"),
            args.bootstrap_replicates, 20260820,
        ),
        "joint_vs_safe_route": grouped_bootstrap(
            bootstrap_rows, "joint_physics_system", "safe_isrj_route",
            ("predicted_gate_parameter_nmae", "miss_penalty_parameter_nmae"),
            args.bootstrap_replicates, 20260830,
        ),
    }

    def values(variant: str, key: str) -> np.ndarray:
        return np.asarray([row["variants"][variant]["aggregate"][key] for row in results], dtype=np.float64)

    original_nmae = values("original_iq_grouped", "predicted_gate_parameter_nmae")
    route_nmae = values("safe_isrj_route", "predicted_gate_parameter_nmae")
    joint_nmae = values("joint_physics_system", "predicted_gate_parameter_nmae")
    original_miss = values("original_iq_grouped", "miss_penalty_parameter_nmae")
    joint_miss = values("joint_physics_system", "miss_penalty_parameter_nmae")
    route_worst = values("safe_isrj_route", "worst_combo_predicted_gate_nmae")
    joint_worst = values("joint_physics_system", "worst_combo_predicted_gate_nmae")
    isrj_gain, macro_delta, weak_recall_delta = [], [], []
    maximum_isrj_fpr, maximum_vft_fpr = 0.0, 0.0
    for row in results:
        safe = row["variants"]["safe_calibrated_iq_grouped"]
        joint = row["variants"]["joint_physics_system"]
        gains = []
        for power in powers:
            base_class = safe["by_power"][str(power)]["classification"]
            joint_class = joint["by_power"][str(power)]["classification"]
            gains.append(joint_class["per_class"][ISRJ_GLOBAL]["recall"] - base_class["per_class"][ISRJ_GLOBAL]["recall"])
            maximum_isrj_fpr = max(maximum_isrj_fpr, joint_class["per_class"][ISRJ_GLOBAL]["fpr"])
            maximum_vft_fpr = max(maximum_vft_fpr, joint_class["per_class"][VFT_CLASS_INDEX]["fpr"])
            base_weak = safe["by_power"][str(power)]["weak_component_recall"]
            joint_weak = joint["by_power"][str(power)]["weak_component_recall"]
            if np.isfinite(base_weak) and np.isfinite(joint_weak):
                weak_recall_delta.append(float(joint_weak - base_weak))
        isrj_gain.append(float(np.mean(gains)))
        macro_delta.append(
            joint["aggregate"]["classification_macro_aupr"]
            - row["variants"]["original_iq_grouped"]["aggregate"]["classification_macro_aupr"]
        )
    nmae_improvement = 1.0 - joint_nmae / route_nmae
    parameter_ci = comparisons["joint_vs_safe_route"]["predicted_gate_parameter_nmae"]
    miss_ci = comparisons["joint_vs_original"]["miss_penalty_parameter_nmae"]
    per_power_gain = {
        str(power): float(np.mean([
            row["variants"]["joint_physics_system"]["by_power"][str(power)]["classification"]["per_class"][ISRJ_GLOBAL]["recall"]
            - row["variants"]["safe_calibrated_iq_grouped"]["by_power"][str(power)]["classification"]["per_class"][ISRJ_GLOBAL]["recall"]
            for row in results
        ]))
        for power in powers
    }
    gates = {
        "isrj_mean_recall_gain_at_least_0p05": bool(np.mean(isrj_gain) >= 0.05),
        "isrj_three_of_three_positive": bool(np.all(np.asarray(isrj_gain) > 0)),
        "maximum_isrj_and_vft_power_fpr_at_most_0p10": bool(max(maximum_isrj_fpr, maximum_vft_fpr) <= 0.10 + 1e-12),
        "macro_aupr_noninferior_vs_original": bool(np.mean(macro_delta) >= -0.005),
        "full_parameter_nmae_improvement_at_least_0p05": bool(np.mean(nmae_improvement) >= 0.05),
        "full_three_of_three_parameter_improvement": bool(np.all(nmae_improvement > 0)),
        "miss_penalty_nonworse_than_original": bool(np.mean(joint_miss - original_miss) <= 0),
        "worst_combo_nonworse_than_safe_route": bool(np.mean(joint_worst - route_worst) <= 0),
        "paired_bootstrap_parameter_and_miss_ci_high_below_zero": bool(
            parameter_ci["ci95_high"] < 0 and miss_ci["ci95_high"] < 0
        ),
    }
    if len(powers) == 9:
        gates.update(nine_power_recall_gates(per_power_gain))
        gates["recall_bootstrap_ci_low_above_zero"] = bool(
            comparisons["safe_route_vs_safe_calibrated"]["isrj_recall"]["ci95_low"] > 0
        )
        gates["all_seed_isrj_and_vft_calibrations_feasible"] = bool(all(
            seed_row["base_isrj_calibration"].get("feasible", False)
            and seed_row["vft_calibration"].get("feasible", False)
            for seed_row in freeze["seed_freezes"]
        ))
    timing = [row["timing"] for row in results]
    first = freeze_by_seed[MODEL_SEEDS[0]]
    parameters = {
        "main": checkpoint_parameter_count(Path(first["main_checkpoint"]), True),
        "unet": checkpoint_parameter_count(Path(first["unet_checkpoint"]), True),
        "saw": checkpoint_parameter_count(Path(first["saw_checkpoint"]), False),
        "vft_aligned_expert_trainable": 0,
    }
    parameters["joint_neural_total"] = parameters["main"] + parameters["unet"] + parameters["saw"]
    collection = {key: float(np.mean([row[key] for row in timing])) for key in (
        "main_seconds", "unet_physics_seconds", "vft_expert_seconds", "saw_expert_seconds", "joint_seconds",
    )}
    collection["joint_to_main_latency_multiplier"] = collection["joint_seconds"] / collection["main_seconds"]
    lazy_rule = freeze.get("lazy_expert_rule")
    if lazy_rule:
        for key in ("isrj_candidate_coverage", "saw_candidate_coverage", "vft_candidate_coverage"):
            collection[key] = float(np.mean([row[key] for row in timing]))
        gates["lazy_joint_to_main_latency_multiplier_at_most_1p5"] = bool(
            collection["joint_to_main_latency_multiplier"] <= float(lazy_rule["latency_multiplier_gate"])
        )
    output = {
        "experiment": freeze["experiment"],
        "recognition": {
            "mean_isrj_recall_gain_vs_safe_calibrated": float(np.mean(isrj_gain)),
            "per_seed_isrj_recall_gain": isrj_gain,
            "maximum_isrj_power_fpr": float(maximum_isrj_fpr), "maximum_vft_power_fpr": float(maximum_vft_fpr),
            "mean_macro_aupr_delta_vs_original": float(np.mean(macro_delta)),
            "per_power_isrj_recall_gain": per_power_gain,
            "nonnegative_isrj_power_count": int(sum(value >= 0 for value in per_power_gain.values())),
            "minimum_isrj_power_recall_gain": float(min(per_power_gain.values())),
            "mean_weak_component_recall_delta_vs_safe_calibrated": float(np.mean(weak_recall_delta)),
        },
        "parameters": {
            "mean_full_predicted_gate_nmae_improvement_vs_safe_route": float(np.mean(nmae_improvement)),
            "per_seed_improvement": nmae_improvement,
            "mean_joint_nmae": float(np.mean(joint_nmae)), "mean_safe_route_nmae": float(np.mean(route_nmae)),
            "mean_original_nmae": float(np.mean(original_nmae)),
            "mean_miss_penalty_delta_vs_original": float(np.mean(joint_miss - original_miss)),
            "mean_worst_combo_delta_vs_safe_route": float(np.mean(joint_worst - route_worst)),
        },
        "paired_bootstrap": comparisons,
        "efficiency": {"parameters": parameters, "measured_test_collection": collection},
        "gates": gates, "level3_validated": bool(all(gates.values())),
        "level4_phase_a_validated": bool(len(powers) == 9 and all(gates.values())),
        "level4_lazy_validated": bool(len(powers) == 9 and lazy_rule and all(gates.values())),
        "method_freeze_sha256": results[0]["method_freeze_sha256"],
        "identity_audit": results[0]["identity_audit"],
        "statistics_source": "immutable saved P1k test predictions; no waveform re-access",
        "statistics_elapsed_seconds": float(time.perf_counter() - started),
        "test_used_once": True, "test_fit_or_selection": False,
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(output), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(output), indent=2))


if __name__ == "__main__":
    main()
