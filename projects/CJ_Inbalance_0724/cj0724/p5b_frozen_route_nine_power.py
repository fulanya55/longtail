"""Evaluate frozen P4c global and routed models on paired fresh nine-power views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724 import CLASS_NAMES
from cj0724.data import V73IQDataset
from cj0724.dual_axis import POWER_LEVELS, discover_power_train_files, read_identity_metadata
from cj0724.multitask import PARAMETER_COLUMNS, PARAMETER_SPECS, build_multitask_model
from cj0724.paper_core_evidence import load_normalizer
from cj0724.tuned_multitask_sweep import _collect_outputs, _metrics


GLOBAL = "legacy_mlcnn_global"
ROUTED = "mlcnn_iq_grouped_task_routed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _power_files(root: Path) -> dict[int, Path]:
    if (root / "power_sweep_manifest.csv").is_file():
        return discover_power_train_files(root)
    files = {
        int(level): root / f"level_{index:02d}__jnr_{int(level)}dB" / "closedset_6class_train.mat"
        for index, level in enumerate(POWER_LEVELS, start=1)
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing power files: {missing}")
    return files


def _weak_mask(labels: np.ndarray, level: int) -> np.ndarray:
    weak = np.zeros_like(labels, dtype=bool)
    if level < 30:
        eligible = labels[:, :3].any(axis=1)
        weak[eligible, 3:] = labels[eligible, 3:].astype(bool)
    elif level > 30:
        eligible = labels[:, 3:].any(axis=1)
        weak[eligible, :3] = labels[eligible, :3].astype(bool)
    return weak


def _per_class(probability: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> list[dict[str, float]]:
    predicted = probability >= thresholds[None, :]
    expected = labels.astype(bool)
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        truth = expected[:, index]
        guess = predicted[:, index]
        tp = np.logical_and(guess, truth).sum()
        fp = np.logical_and(guess, ~truth).sum()
        fn = np.logical_and(~guess, truth).sum()
        tn = np.logical_and(~guess, ~truth).sum()
        rows.append({
            "class_name": name,
            "recall": float(tp / max(tp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "fpr": float(fp / max(fp + tn, 1)),
        })
    return rows


def _sample_nmae(outputs: dict[str, object], scale: np.ndarray) -> np.ndarray:
    labels = outputs["labels"].numpy() > 0.5
    valid = outputs["mask"] & labels[:, :, None]
    normalized_error = np.abs(outputs["prediction"] - outputs["target"]) / scale[None, :, :]
    numerator = np.where(valid, normalized_error, 0.0).sum(axis=(1, 2))
    denominator = valid.sum(axis=(1, 2)).clip(min=1)
    return numerator / denominator


def _weak_metrics(
    outputs: dict[str, object], probability: np.ndarray, thresholds: np.ndarray,
    weak: np.ndarray, scale: np.ndarray,
) -> dict[str, float]:
    if not weak.any():
        return {
            "weak_component_recall": float("nan"),
            "weak_oracle_parameter_nmae": float("nan"),
            "weak_predicted_gate_parameter_coverage": float("nan"),
            "weak_predicted_gate_parameter_nmae": float("nan"),
        }
    predicted = probability >= thresholds[None, :]
    valid = outputs["mask"] & weak[:, :, None]
    normalized_error = np.abs(outputs["prediction"] - outputs["target"]) / scale[None, :, :]
    gated = valid & predicted[:, :, None]
    return {
        "weak_component_recall": float(predicted[weak].mean()),
        "weak_oracle_parameter_nmae": float(normalized_error[valid].mean()),
        "weak_predicted_gate_parameter_coverage": float(gated.sum() / max(valid.sum(), 1)),
        "weak_predicted_gate_parameter_nmae": (
            float(normalized_error[gated].mean()) if gated.any() else float("nan")
        ),
    }


def _target_map(metrics: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        f"{row['class_name']}.{row['name']}": row
        for row in metrics["parameter"]["per_target"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--power-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project = Path.cwd()
    preregister_path = project / args.preregister
    preregister = json.loads(preregister_path.read_text(encoding="utf-8"))
    data_audit = json.loads((project / args.data_audit).read_text(encoding="utf-8"))
    if not data_audit["valid_for_frozen_inference"]:
        raise RuntimeError("P5b data audit did not authorize frozen inference")
    output_root = project / args.output_root
    prediction_root = output_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    run_root = project / preregister["frozen_method"]["source_run"]
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    normalizer = load_normalizer(protocol["normalizer"])
    scale = normalizer.scale.numpy()
    files = _power_files(project / args.power_root)
    identities, reference_labels = read_identity_metadata(files[30])
    indices = np.arange(identities.size, dtype=np.int64)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    selected_rows = [
        row for row in summary
        if row["model"] in (GLOBAL, ROUTED)
        and row.get("checkpoint_objective") == "parameter"
        and int(row["seed"]) in preregister["frozen_method"]["model_seeds"]
    ]
    if len(selected_rows) != 6:
        raise RuntimeError(f"Expected six frozen rows, found {len(selected_rows)}")

    evaluations = []
    sample_nmae: dict[tuple[str, int, int], np.ndarray] = {}
    for row in selected_rows:
        model_name = str(row["model"])
        model_seed = int(row["seed"])
        model = build_multitask_model(
            model_name, protocol["width"], protocol["dropout"], protocol["head_dim"],
            protocol["parameter_hidden_dim"], protocol["legacy_attention"],
        ).to(device)
        checkpoint_path = Path(row["checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = project / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        thresholds = np.asarray(json.loads(row["thresholds"]), dtype=np.float32)
        saved = {
            "identities": identities.astype(np.int64),
            "powers_db": np.asarray(POWER_LEVELS, dtype=np.int16),
            "thresholds": thresholds,
        }
        probabilities, predictions, targets, masks = [], [], [], []
        for level in POWER_LEVELS:
            level_identities, labels = read_identity_metadata(files[level])
            if not np.array_equal(level_identities, identities) or not np.array_equal(labels, reference_labels):
                raise RuntimeError(f"Power pairing changed before inference at {level} dB")
            dataset = V73IQDataset(files[level], indices, PARAMETER_COLUMNS)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                pin_memory=device.type == "cuda",
            )
            outputs = _collect_outputs(
                model, loader, device, normalizer, float(protocol["parameter_weight"])
            )
            metrics = _metrics(outputs, thresholds, normalizer)
            probability = torch.sigmoid(outputs["logits"]).numpy()
            weak = _weak_mask(reference_labels.astype(bool), int(level))
            per_class = _per_class(probability, reference_labels, thresholds)
            weak_result = _weak_metrics(outputs, probability, thresholds, weak, scale)
            sample_nmae[(model_name, model_seed, int(level))] = _sample_nmae(outputs, scale)
            evaluations.append({
                "model": model_name,
                "model_seed": model_seed,
                "power_db": int(level),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "thresholds": thresholds.tolist(),
                "classification": {
                    "macro_aupr": float(metrics["macro_aupr"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "micro_f1": float(metrics["micro_f1"]),
                    "subset_accuracy": float(metrics["subset_acc"]),
                    "per_class": per_class,
                },
                "parameter": {
                    "oracle_macro_nmae": float(metrics["parameter"]["oracle_macro_target_nmae"]),
                    "oracle_macro_pearson": float(metrics["parameter"]["oracle_macro_pearson_r"]),
                    "per_target": metrics["parameter"]["per_target"],
                },
                "weak": weak_result,
            })
            probabilities.append(probability.astype(np.float32))
            predictions.append(outputs["prediction"].astype(np.float32))
            targets.append(outputs["target"].astype(np.float32))
            masks.append(outputs["mask"].astype(np.uint8))
            dataset.close()
        np.savez_compressed(
            prediction_root / f"{model_name}__seed{model_seed}.npz",
            **saved,
            probabilities=np.stack(probabilities),
            predictions=np.stack(predictions),
            targets=np.stack(targets),
            masks=np.stack(masks),
            labels=reference_labels.astype(np.uint8),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    indexed = {
        (row["model"], int(row["model_seed"]), int(row["power_db"])): row
        for row in evaluations
    }
    seeds = preregister["frozen_method"]["model_seeds"]
    pairs = []
    for seed in seeds:
        for level in POWER_LEVELS:
            global_row = indexed[(GLOBAL, int(seed), int(level))]
            routed_row = indexed[(ROUTED, int(seed), int(level))]
            global_nmae = global_row["parameter"]["oracle_macro_nmae"]
            routed_nmae = routed_row["parameter"]["oracle_macro_nmae"]
            global_targets = _target_map({"parameter": {"per_target": global_row["parameter"]["per_target"]}})
            routed_targets = _target_map({"parameter": {"per_target": routed_row["parameter"]["per_target"]}})
            pairs.append({
                "model_seed": int(seed),
                "power_db": int(level),
                "global_macro_nmae": global_nmae,
                "routed_macro_nmae": routed_nmae,
                "relative_macro_nmae_reduction": (global_nmae - routed_nmae) / global_nmae,
                "parameter_macro_pearson_delta": (
                    routed_row["parameter"]["oracle_macro_pearson"]
                    - global_row["parameter"]["oracle_macro_pearson"]
                ),
                "macro_aupr_delta": (
                    routed_row["classification"]["macro_aupr"]
                    - global_row["classification"]["macro_aupr"]
                ),
                "subset_accuracy_delta": (
                    routed_row["classification"]["subset_accuracy"]
                    - global_row["classification"]["subset_accuracy"]
                ),
                "weak_component_recall_delta": (
                    routed_row["weak"]["weak_component_recall"]
                    - global_row["weak"]["weak_component_recall"]
                ),
                "weak_oracle_parameter_nmae_delta": (
                    routed_row["weak"]["weak_oracle_parameter_nmae"]
                    - global_row["weak"]["weak_oracle_parameter_nmae"]
                ),
                "per_target_nmae_delta": {
                    key: float(routed_targets[key]["normalized_mae"])
                    - float(global_targets[key]["normalized_mae"])
                    for key in global_targets
                },
            })

    seed_summary = []
    for seed in seeds:
        selected = [row for row in pairs if row["model_seed"] == seed]
        reductions = np.asarray([row["relative_macro_nmae_reduction"] for row in selected])
        seed_summary.append({
            "model_seed": int(seed),
            "mean_relative_macro_nmae_reduction": float(reductions.mean()),
            "nonnegative_reduction_power_count": int((reductions >= 0.0).sum()),
            "mean_parameter_macro_pearson_delta": float(np.nanmean([
                row["parameter_macro_pearson_delta"] for row in selected
            ])),
            "mean_macro_aupr_delta": float(np.mean([
                row["macro_aupr_delta"] for row in selected
            ])),
            "worst_subset_accuracy_delta": float(np.min([
                row["subset_accuracy_delta"] for row in selected
            ])),
        })
    power_summary = []
    systematic = []
    for level in POWER_LEVELS:
        selected = [row for row in pairs if row["power_db"] == level]
        reductions = [row["relative_macro_nmae_reduction"] for row in selected]
        power_summary.append({
            "power_db": int(level),
            "mean_relative_macro_nmae_reduction": float(np.mean(reductions)),
            "minimum_relative_macro_nmae_reduction": float(np.min(reductions)),
            "mean_macro_aupr_delta": float(np.mean([row["macro_aupr_delta"] for row in selected])),
            "mean_subset_accuracy_delta": float(np.mean([row["subset_accuracy_delta"] for row in selected])),
            "mean_weak_component_recall_delta": float(np.nanmean([
                row["weak_component_recall_delta"] for row in selected
            ])) if level != 30 else float("nan"),
            "mean_weak_oracle_parameter_nmae_delta": float(np.nanmean([
                row["weak_oracle_parameter_nmae_delta"] for row in selected
            ])) if level != 30 else float("nan"),
        })
        for target in [f"{spec.class_name}.{spec.name}" for spec in PARAMETER_SPECS]:
            deltas = [row["per_target_nmae_delta"][target] for row in selected]
            limit = preregister["robust_generalization_gates"][
                "no_target_worse_at_any_power_in_all_three_model_seeds_by_more_than_absolute_nmae"
            ]
            if all(delta > limit for delta in deltas):
                systematic.append({
                    "power_db": int(level), "target": target,
                    "deltas": deltas, "threshold": float(limit),
                })

    cluster_gains = []
    for seed in seeds:
        per_power_gain = np.stack([
            sample_nmae[(GLOBAL, int(seed), int(level))]
            - sample_nmae[(ROUTED, int(seed), int(level))]
            for level in POWER_LEVELS
        ])
        cluster_gains.append(per_power_gain.mean(axis=0))
    cluster_gain = np.stack(cluster_gains).mean(axis=0)
    rng = np.random.default_rng(preregister["statistics"]["bootstrap_seed"])
    bootstrap = np.empty(preregister["statistics"]["identity_clustered_bootstrap_replicates"])
    for index in range(bootstrap.size):
        sampled = rng.integers(0, identities.size, size=identities.size)
        bootstrap[index] = cluster_gain[sampled].mean()
    bootstrap_result = {
        "mean_absolute_nmae_gain": float(cluster_gain.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "replicates": int(bootstrap.size),
        "cluster": "receiver_noise_seed with all nine power views kept together",
    }

    across_power_variance = []
    for model_name in (GLOBAL, ROUTED):
        for seed in seeds:
            matrix = np.stack([
                sample_nmae[(model_name, int(seed), int(level))] for level in POWER_LEVELS
            ])
            across_power_variance.append({
                "model": model_name,
                "model_seed": int(seed),
                "mean_identity_nmae_variance_across_powers": float(matrix.var(axis=0).mean()),
            })

    gates = preregister["robust_generalization_gates"]
    gate_results = {
        "each_seed_mean_nmae_reduction": all(
            row["mean_relative_macro_nmae_reduction"]
            >= gates["each_model_seed_mean_nine_power_relative_macro_nmae_reduction_at_least"]
            for row in seed_summary
        ),
        "each_seed_nonnegative_power_count": all(
            row["nonnegative_reduction_power_count"]
            >= gates["each_model_seed_nonnegative_nmae_reduction_power_levels_at_least"]
            for row in seed_summary
        ),
        "each_power_mean_nmae_reduction": all(
            row["mean_relative_macro_nmae_reduction"]
            >= gates["each_power_mean_relative_macro_nmae_reduction_at_least"]
            for row in power_summary
        ),
        "each_seed_mean_pearson_nonlower": all(
            row["mean_parameter_macro_pearson_delta"]
            >= gates["each_model_seed_mean_nine_power_parameter_macro_pearson_delta_at_least"]
            for row in seed_summary
        ),
        "each_seed_mean_macro_aupr": all(
            row["mean_macro_aupr_delta"]
            >= gates["each_model_seed_mean_nine_power_macro_aupr_delta_at_least"]
            for row in seed_summary
        ),
        "each_seed_worst_subset": all(
            row["worst_subset_accuracy_delta"]
            >= gates["each_model_seed_worst_power_subset_accuracy_delta_at_least"]
            for row in seed_summary
        ),
        "no_systematic_target_failure": not systematic,
        "identity_clustered_bootstrap": bootstrap_result["ci95_low"]
        > gates["identity_clustered_bootstrap_mean_nmae_gain_ci95_low_greater_than"],
    }
    payload = {
        "experiment": preregister["experiment"],
        "preregister": str(preregister_path),
        "preregister_sha256": sha256_file(preregister_path),
        "source_run": str(run_root),
        "data_audit": data_audit,
        "evaluation_contract": {
            "training_on_p5b_data": False,
            "normalizer_refit": False,
            "threshold_recalibration": False,
            "model_selection_on_p5b_data": False,
            "test_generated": False,
            "test_opened": False,
            "locked_test_used": False,
        },
        "evaluations": evaluations,
        "paired_route_vs_global": pairs,
        "seed_summary": seed_summary,
        "power_summary": power_summary,
        "systematic_target_failures": systematic,
        "sawfm_sweep_period_by_power": [
            {
                "power_db": int(level),
                "deltas": [
                    row["per_target_nmae_delta"]["SawFM.sweep_period"]
                    for row in pairs if row["power_db"] == level
                ],
            }
            for level in POWER_LEVELS
        ],
        "identity_clustered_bootstrap": bootstrap_result,
        "same_identity_across_power_variance": across_power_variance,
        "gate_results": gate_results,
        "robust_generalization_passed": bool(all(gate_results.values())),
        "physical_sufficiency_screen_required": bool(systematic),
        "predictions_root": str(prediction_root),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "result.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "seed_summary": seed_summary,
        "power_summary": power_summary,
        "systematic_target_failures": systematic,
        "identity_clustered_bootstrap": bootstrap_result,
        "gate_results": gate_results,
        "robust_generalization_passed": payload["robust_generalization_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
