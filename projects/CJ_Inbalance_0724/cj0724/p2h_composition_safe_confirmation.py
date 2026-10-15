"""Independent confirmation of the frozen composition-safe P2h method."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cj0724.p1k_joint_level3_system import MODEL_SEEDS
from cj0724.p2e_stage_b_weak_generalization import CONDITIONS, sha256_file
from cj0724.p2f_composition_safe_development import (
    TARGETS,
    class_miss_penalty_nmae,
    gated_prediction,
    parameter_nmae,
    target_recall_fpr,
    worst_combination_parameter_nmae,
)
from cj0724.paper_core_evidence import load_normalizer


def variant_metrics(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    prediction: np.ndarray, scale: np.ndarray, target_name: str, rows: np.ndarray,
) -> dict[str, object]:
    class_index, condition_ids, parameter_index = TARGETS[target_name]
    recall, maximum_fpr, per_condition_fpr = target_recall_fpr(
        saved, probability, thresholds, class_index, condition_ids, rows,
    )
    output: dict[str, object] = {
        "recall": recall,
        "maximum_condition_fpr": maximum_fpr,
        "per_condition_fpr": per_condition_fpr,
        "miss_penalty_nmae": class_miss_penalty_nmae(
            saved, probability, thresholds, prediction, scale,
            class_index, condition_ids, rows,
        ),
    }
    if parameter_index is not None:
        output.update({
            "expert_parameter_oracle_nmae": parameter_nmae(
                saved, np.ones_like(probability), np.zeros_like(thresholds), prediction, scale,
                class_index, parameter_index, condition_ids, rows,
            ),
            "expert_parameter_predicted_gate_nmae": parameter_nmae(
                saved, probability, thresholds, prediction, scale,
                class_index, parameter_index, condition_ids, rows,
            ),
            "expert_parameter_coverage": float((
                rows & np.isin(saved["condition_id"], condition_ids)
                & saved["labels"][:, class_index].astype(bool)
                & saved["mask"][:, class_index, parameter_index].astype(bool)
                & (probability[:, class_index] >= thresholds[class_index])
            ).sum() / max((
                rows & np.isin(saved["condition_id"], condition_ids)
                & saved["labels"][:, class_index].astype(bool)
                & saved["mask"][:, class_index, parameter_index].astype(bool)
            ).sum(), 1)),
            "worst_combination_predicted_gate_nmae": worst_combination_parameter_nmae(
                saved, probability, thresholds, prediction, scale,
                class_index, parameter_index, condition_ids, rows,
            ),
        })
    return output


def stratified_metrics(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    prediction: np.ndarray, scale: np.ndarray, target_name: str,
) -> dict[str, object]:
    class_index, condition_ids, _ = TARGETS[target_name]
    output = {"aggregate": variant_metrics(
        saved, probability, thresholds, prediction, scale, target_name,
        np.ones(len(saved["labels"]), dtype=bool),
    )}
    by_condition = {}
    for condition_id in condition_ids:
        rows = saved["condition_id"] == condition_id
        by_condition[CONDITIONS[condition_id][0]] = variant_metrics(
            saved, probability, thresholds, prediction, scale, target_name, rows,
        )
    output["by_condition"] = by_condition
    by_size = {}
    sizes = saved["labels"].sum(axis=1)
    target_rows = np.isin(saved["condition_id"], condition_ids) & saved["labels"][:, class_index].astype(bool)
    for size in range(1, 7):
        rows = target_rows & (sizes == size)
        if rows.any():
            by_size[str(size)] = variant_metrics(
                saved, probability, thresholds, prediction, scale, target_name, rows,
            )
    output["by_combination_size"] = by_size
    return output


def metric_value(
    payload: dict[str, object], target_name: str, metric: str, rows: np.ndarray,
) -> float:
    saved = payload["saved"]
    variant = payload["variant"]
    class_index, condition_ids, parameter_index = TARGETS[target_name]
    if metric == "recall":
        return target_recall_fpr(
            saved, variant["probability"], variant["thresholds"], class_index, condition_ids, rows,
        )[0]
    if metric == "parameter_nmae":
        assert parameter_index is not None
        return parameter_nmae(
            saved, variant["probability"], variant["thresholds"], variant["prediction"],
            payload["scale"], class_index, parameter_index, condition_ids, rows,
        )
    if metric == "miss_penalty_nmae":
        return class_miss_penalty_nmae(
            saved, variant["probability"], variant["thresholds"], variant["prediction"],
            payload["scale"], class_index, condition_ids, rows,
        )
    raise KeyError(metric)


def paired_identity_bootstrap(
    seed_payloads: list[dict[str, object]], target_name: str, metric: str,
    candidate: str, baseline: str, replicates: int, seed: int,
) -> dict[str, float]:
    def subset_payload(payload: dict[str, object], selected_rows: np.ndarray) -> dict[str, object]:
        sample_count = len(payload["saved"]["labels"])
        saved = {
            key: value[selected_rows]
            if isinstance(value, np.ndarray) and value.shape[:1] == (sample_count,)
            else value
            for key, value in payload["saved"].items()
        }
        variant = {
            key: value[selected_rows]
            if isinstance(value, np.ndarray) and value.shape[:1] == (sample_count,)
            else value
            for key, value in payload["variant"].items()
        }
        return {"saved": saved, "variant": variant, "scale": payload["scale"]}

    observed = []
    prepared = []
    for seed_payload in seed_payloads:
        identities = seed_payload["identity"]
        unique, inverse = np.unique(identities, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique))
        if not np.all(counts == counts[0]):
            raise RuntimeError("Confirmation identity groups must have equal paired-view counts")
        group_rows = np.argsort(inverse, kind="stable").reshape(len(unique), int(counts[0]))
        all_rows = np.ones(len(identities), dtype=bool)
        observed.append(
            metric_value(seed_payload[candidate], target_name, metric, all_rows)
            - metric_value(seed_payload[baseline], target_name, metric, all_rows)
        )
        prepared.append((seed_payload, group_rows))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        deltas = []
        for seed_payload, group_rows in prepared:
            sampled = rng.integers(0, len(group_rows), size=len(group_rows))
            sampled_rows = group_rows[sampled].reshape(-1)
            candidate_subset = subset_payload(seed_payload[candidate], sampled_rows)
            baseline_subset = subset_payload(seed_payload[baseline], sampled_rows)
            subset_rows = np.ones(len(sampled_rows), dtype=bool)
            deltas.append(
                metric_value(candidate_subset, target_name, metric, subset_rows)
                - metric_value(baseline_subset, target_name, metric, subset_rows)
            )
        samples.append(float(np.mean(deltas)))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "observed_mean_delta": float(np.mean(observed)),
        "ci95_low": float(np.nanquantile(values, 0.025)),
        "ci95_high": float(np.nanquantile(values, 0.975)),
        "probability_delta_positive": float(np.nanmean(values > 0)),
        "replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", default="outputs/p2h_stage_b_confirmation_predictions_20260801")
    parser.add_argument("--development-root", default="outputs/p2h_composition_safe_development_v3_20260801")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--output-root", default="outputs/p2h_composition_safe_confirmation_20260801")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    prediction_root = project / args.prediction_root
    development_root = project / args.development_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    method_path = development_root / "method_freeze.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    development_summary = json.loads((development_root / "summary.json").read_text(encoding="utf-8"))
    if not development_summary["development_promoted"]:
        raise RuntimeError("Confirmation refused: frozen development method did not pass promotion gates")
    method_hash = sha256_file(method_path)
    (output_root / "confirmation_access_receipt.json").write_text(json.dumps({
        "method_freeze": str(method_path.resolve()),
        "method_freeze_sha256": method_hash,
        "access_started_unix": time.time(),
        "method_changed_after_access": False,
        "locked_test_used": False,
    }, indent=2), encoding="utf-8")
    protocol = json.loads((project / args.run_root / "protocol.json").read_text(encoding="utf-8"))
    scale = load_normalizer(protocol["normalizer"]).scale.numpy()
    freeze_by_seed = {int(row["seed"]): row for row in method["seed_freezes"]}
    seed_results, bootstrap_payloads = [], []
    for seed in MODEL_SEEDS:
        path = prediction_root / f"seed_{seed}" / "test_per_sample.npz"
        with np.load(path, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        freeze = freeze_by_seed[seed]
        candidate_thresholds = np.asarray(freeze["candidate_thresholds"], dtype=np.float64)
        main_thresholds = np.asarray(freeze["fpr_matched_main_thresholds"], dtype=np.float64)
        candidate_probability = saved["routed_probability"]
        main_probability = saved["original_probability"]
        candidate_prediction, call_coverage = gated_prediction(
            saved, candidate_probability, candidate_thresholds,
            int(freeze["sawfm_max_predicted_complexity"]),
            int(freeze["vft_max_predicted_complexity"]),
            float(freeze["sawfm_max_disagreement"]),
            float(freeze["vft_max_disagreement"]),
            freeze["sawfm_allowed_predicted_patterns"],
            freeze["vft_allowed_predicted_patterns"],
        )
        variants = {
            "fpr_matched_main": {
                "probability": main_probability,
                "thresholds": main_thresholds,
                "prediction": saved["base_prediction"],
            },
            "composition_safe_joint": {
                "probability": candidate_probability,
                "thresholds": candidate_thresholds,
                "prediction": candidate_prediction,
            },
        }
        metrics = {
            variant_name: {
                target_name: stratified_metrics(
                    saved, variant["probability"], variant["thresholds"],
                    variant["prediction"], scale, target_name,
                )
                for target_name in TARGETS
            }
            for variant_name, variant in variants.items()
        }
        seed_result = {
            "seed": seed,
            "metrics": metrics,
            "expert_call_coverage": call_coverage,
            "prediction_sha256": sha256_file(path),
            "method_freeze_sha256": method_hash,
            "locked_test_used": False,
        }
        seed_results.append(seed_result)
        base_payload = {"saved": saved, "variant": variants["fpr_matched_main"], "scale": scale}
        candidate_payload = {"saved": saved, "variant": variants["composition_safe_joint"], "scale": scale}
        bootstrap_payloads.append({
            "identity": saved["receiver_noise_seed"],
            "fpr_matched_main": base_payload,
            "composition_safe_joint": candidate_payload,
        })

    bootstrap = {
        "isrj_recall": paired_identity_bootstrap(
            bootstrap_payloads, "ISRJ", "recall", "composition_safe_joint", "fpr_matched_main",
            args.bootstrap_replicates, 20261001,
        ),
        "sawfm_parameter": paired_identity_bootstrap(
            bootstrap_payloads, "SawFM", "parameter_nmae", "composition_safe_joint", "fpr_matched_main",
            args.bootstrap_replicates, 20261002,
        ),
        "vft_parameter": paired_identity_bootstrap(
            bootstrap_payloads, "VFT", "parameter_nmae", "composition_safe_joint", "fpr_matched_main",
            args.bootstrap_replicates, 20261003,
        ),
    }
    for index, target_name in enumerate(TARGETS):
        bootstrap[f"{target_name.lower()}_miss_penalty"] = paired_identity_bootstrap(
            bootstrap_payloads, target_name, "miss_penalty_nmae",
            "composition_safe_joint", "fpr_matched_main",
            args.bootstrap_replicates, 20261010 + index,
        )

    def aggregate_value(target: str, key: str, variant: str) -> list[float]:
        return [
            float(row["metrics"][variant][target]["aggregate"][key])
            for row in seed_results
        ]

    maximum_fpr = float(max(max(values) if isinstance(values, list) else values for values in [
        aggregate_value(target, "maximum_condition_fpr", "composition_safe_joint") for target in TARGETS
    ]))
    isrj_gain = np.asarray(aggregate_value("ISRJ", "recall", "composition_safe_joint")) - np.asarray(
        aggregate_value("ISRJ", "recall", "fpr_matched_main")
    )
    parameter_improvements = {}
    worst_deltas = {}
    for target in ("SawFM", "VFT"):
        candidate = np.asarray(aggregate_value(target, "expert_parameter_predicted_gate_nmae", "composition_safe_joint"))
        baseline = np.asarray(aggregate_value(target, "expert_parameter_predicted_gate_nmae", "fpr_matched_main"))
        parameter_improvements[target] = ((baseline - candidate) / baseline).tolist()
        worst_deltas[target] = (
            np.asarray(aggregate_value(target, "worst_combination_predicted_gate_nmae", "composition_safe_joint"))
            - np.asarray(aggregate_value(target, "worst_combination_predicted_gate_nmae", "fpr_matched_main"))
        ).tolist()
    miss_deltas = {
        target: (
            np.asarray(aggregate_value(target, "miss_penalty_nmae", "composition_safe_joint"))
            - np.asarray(aggregate_value(target, "miss_penalty_nmae", "fpr_matched_main"))
        ).tolist()
        for target in TARGETS
    }
    gates = {
        "maximum_target_fpr_at_most_0p10": bool(maximum_fpr <= 0.10 + 1e-12),
        "isrj_mean_recall_gain_at_least_0p05": bool(isrj_gain.mean() >= 0.05),
        "isrj_three_of_three_positive": bool(np.all(isrj_gain > 0)),
        "sawfm_mean_parameter_improvement_at_least_0p05": bool(np.mean(parameter_improvements["SawFM"]) >= 0.05),
        "sawfm_three_of_three_positive": bool(np.all(np.asarray(parameter_improvements["SawFM"]) > 0)),
        "vft_mean_parameter_improvement_at_least_0p05": bool(np.mean(parameter_improvements["VFT"]) >= 0.05),
        "vft_three_of_three_positive": bool(np.all(np.asarray(parameter_improvements["VFT"]) > 0)),
        "miss_penalty_nonworse_vs_fpr_matched_main": bool(
            all(np.mean(values) <= 0 for values in miss_deltas.values())
        ),
        "worst_combination_nonworse": bool(
            all(np.mean(values) <= 0 for values in worst_deltas.values())
        ),
        "paired_bootstrap_directional_ci": bool(
            bootstrap["isrj_recall"]["ci95_low"] > 0
            and bootstrap["sawfm_parameter"]["ci95_high"] < 0
            and bootstrap["vft_parameter"]["ci95_high"] < 0
            and all(bootstrap[f"{target.lower()}_miss_penalty"]["ci95_high"] < 0 for target in TARGETS)
        ),
    }
    collector_summary = json.loads((prediction_root / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "experiment": method["experiment"] + " independent confirmation",
        "aggregate": {
            "maximum_target_fpr": maximum_fpr,
            "isrj_recall_gain_per_seed": isrj_gain.tolist(),
            "mean_isrj_recall_gain": float(isrj_gain.mean()),
            "parameter_improvement_per_seed": parameter_improvements,
            "mean_parameter_improvement": {
                target: float(np.mean(values)) for target, values in parameter_improvements.items()
            },
            "miss_penalty_deltas": miss_deltas,
            "worst_combination_deltas": worst_deltas,
        },
        "paired_bootstrap": bootstrap,
        "gates": gates,
        "stage_b_validated": bool(all(gates.values())),
        "identity_audit": collector_summary["identity_audit"],
        "method_freeze_sha256": method_hash,
        "confirmation_fit_or_selection": False,
        "locked_test_used": False,
    }
    (output_root / "seed_results.json").write_text(json.dumps(seed_results, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"] | {"gates": gates, "stage_b_validated": summary["stage_b_validated"]}, indent=2))


if __name__ == "__main__":
    main()
