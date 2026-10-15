"""P2e Stage B: frozen weak-component and 1--6 component validation.

The P2c models, thresholds, routes, experts and lazy rules are reused without
fitting.  Nine target-specific counterfactual views weaken SawFM, VFT or ISRJ
by 5/10/15 dB while preserving all 63 non-empty class combinations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from cj0724.dual_axis import read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import (
    _jsonable, _load_main, _load_unet,
)
from cj0724.p1f_isrj_confidence_bound_router import (
    ISRJ_GLOBAL, apply_route, ensemble_scores,
)
from cj0724.p1k_joint_level3_system import (
    MODEL_SEEDS, VFT_CLASS_INDEX, collect_joint_records, evaluate_variant,
    fused_prediction, role_payload,
)
from cj0724.paper_core_evidence import load_normalizer
from cj0724.physics_parameter_specialists import build_physics_specialist


CONDITIONS = (
    ("sawfm_weak_05dB", 1, 5), ("sawfm_weak_10dB", 1, 10), ("sawfm_weak_15dB", 1, 15),
    ("vft_weak_05dB", 4, 5), ("vft_weak_10dB", 4, 10), ("vft_weak_15dB", 4, 15),
    ("isrj_weak_05dB", 5, 5), ("isrj_weak_10dB", 5, 10), ("isrj_weak_15dB", 5, 15),
)
TARGET_NAMES = {1: "SawFM", 4: "VFT", 5: "ISRJ"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_record(saved: dict[str, np.ndarray], variant: str, scale: np.ndarray) -> dict[str, np.ndarray]:
    if variant == "original_iq_grouped":
        probability, thresholds, prediction = saved["original_probability"], saved["original_thresholds"], saved["base_prediction"]
    elif variant == "safe_calibrated_iq_grouped":
        probability, thresholds, prediction = saved["original_probability"], saved["safe_thresholds"], saved["base_prediction"]
    elif variant == "safe_isrj_route":
        probability, thresholds, prediction = saved["routed_probability"], saved["routed_thresholds"], saved["base_prediction"]
    elif variant == "joint_physics_system":
        probability, thresholds, prediction = saved["routed_probability"], saved["routed_thresholds"], saved["joint_prediction"]
    else:
        raise KeyError(variant)
    return {
        "probability": probability,
        "thresholds": thresholds,
        "prediction": prediction,
        "labels": saved["labels"].astype(bool),
        "target": saved["target"],
        "mask": saved["mask"].astype(bool),
        "condition_id": saved["condition_id"],
        "scale": scale,
    }


def target_metric(record: dict[str, np.ndarray], class_index: int, condition_ids: tuple[int, ...], metric: str) -> float:
    rows = np.isin(record["condition_id"], condition_ids)
    labels = record["labels"][rows]
    probability = record["probability"][rows]
    predicted = probability >= record["thresholds"][None, :]
    positive = labels[:, class_index]
    if metric == "recall":
        return float(predicted[positive, class_index].mean()) if positive.any() else float("nan")
    if metric == "fpr":
        return float(predicted[~positive, class_index].mean()) if (~positive).any() else float("nan")
    target = record["target"][rows]
    mask = record["mask"][rows]
    prediction = record["prediction"][rows]
    normalized = np.abs(prediction - target) / record["scale"][None, :, :]
    oracle = mask[:, class_index, :] & positive[:, None]
    gated = oracle & predicted[:, class_index, None]
    values = []
    for parameter in range(2):
        valid = oracle[:, parameter]
        if not valid.any():
            continue
        if metric == "oracle_nmae":
            values.append(float(normalized[:, class_index, parameter][valid].mean()))
        elif metric == "predicted_gate_nmae":
            selected = gated[:, parameter]
            values.append(float(normalized[:, class_index, parameter][selected].mean()) if selected.any() else float("nan"))
        elif metric == "miss_penalty_nmae":
            penalized = np.where(predicted[:, class_index], normalized[:, class_index, parameter], 1.0)
            values.append(float(penalized[valid].mean()))
        elif metric == "coverage":
            values.append(float(gated[:, parameter].sum() / valid.sum()))
        else:
            raise KeyError(metric)
    return float(np.mean(values))


def condition_metrics(record: dict[str, np.ndarray], condition_id: int, class_index: int) -> dict[str, object]:
    output = {
        metric: target_metric(record, class_index, (condition_id,), metric)
        for metric in ("recall", "fpr", "coverage", "oracle_nmae", "predicted_gate_nmae", "miss_penalty_nmae")
    }
    rows = record["condition_id"] == condition_id
    labels = record["labels"][rows]
    combo_size = labels.sum(axis=1)
    def subset_record(keep: np.ndarray) -> dict[str, np.ndarray]:
        return {
            key: value[rows][keep]
            if isinstance(value, np.ndarray) and value.shape[:1] == (len(rows),)
            else value
            for key, value in record.items()
        }
    output["by_combination_size"] = {
        str(size): {
            metric: target_metric(
                subset_record(combo_size == size),
                class_index, (condition_id,), metric,
            )
            for metric in ("recall", "coverage", "predicted_gate_nmae", "miss_penalty_nmae")
        }
        for size in range(1, 7)
        if np.any((combo_size == size) & labels[:, class_index])
    }
    combinations = np.asarray(["".join(str(int(value)) for value in row) for row in labels])
    output["by_combination"] = {
        combination: {
            metric: target_metric(subset_record(combinations == combination), class_index, (condition_id,), metric)
            for metric in ("recall", "coverage", "predicted_gate_nmae", "miss_penalty_nmae")
        }
        for combination in sorted(set(combinations[labels[:, class_index]].tolist()))
    }
    finite_worst = [
        row["predicted_gate_nmae"] for row in output["by_combination"].values()
        if np.isfinite(row["predicted_gate_nmae"])
    ]
    output["worst_combination_predicted_gate_nmae"] = float(max(finite_worst)) if finite_worst else float("nan")
    return output


def grouped_bootstrap(
    rows: list[dict[str, object]], class_index: int, condition_ids: tuple[int, ...],
    metric: str, candidate: str, baseline: str, replicates: int, seed: int,
) -> dict[str, float]:
    observed = float(np.mean([
        target_metric(row["records"][candidate], class_index, condition_ids, metric)
        - target_metric(row["records"][baseline], class_index, condition_ids, metric)
        for row in rows
    ]))
    rng = np.random.default_rng(seed)
    prepared_rows = []
    for row in rows:
        identities = np.asarray(row["identity"])
        unique, inverse = np.unique(identities, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique))
        order = np.argsort(inverse, kind="stable")
        if np.all(counts == counts[0]):
            identity_rows: np.ndarray | list[np.ndarray] = order.reshape(len(unique), int(counts[0]))
        else:
            offsets = np.concatenate(([0], np.cumsum(counts)))
            identity_rows = [order[offsets[index]:offsets[index + 1]] for index in range(len(unique))]
        prepared_rows.append((row, unique, identity_rows))
    samples = []
    for _ in range(replicates):
        seed_deltas = []
        for row, unique, identity_rows in prepared_rows:
            sampled = rng.integers(0, len(unique), size=len(unique))
            if isinstance(identity_rows, np.ndarray):
                sampled_rows = identity_rows[sampled].reshape(-1)
            else:
                sampled_rows = np.concatenate([identity_rows[value] for value in sampled])
            subset = {}
            for name in (candidate, baseline):
                source = row["records"][name]
                subset[name] = {
                    key: value[sampled_rows]
                    if isinstance(value, np.ndarray) and value.shape[:1] == (len(identities),)
                    else value
                    for key, value in source.items()
                }
            seed_deltas.append(
                target_metric(subset[candidate], class_index, condition_ids, metric)
                - target_metric(subset[baseline], class_index, condition_ids, metric)
            )
        samples.append(float(np.mean(seed_deltas)))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "observed_mean_delta": observed,
        "ci95_low": float(np.nanquantile(values, 0.025)),
        "ci95_high": float(np.nanquantile(values, 0.975)),
        "probability_delta_positive": float(np.nanmean(values > 0)),
        "replicates": int(replicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="generated_data/p2e_stage_b_weak_components_seed1500000000_rechunked")
    parser.add_argument("--source-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--output-root", default="outputs/p2e_stage_b_weak_generalization_20260801")
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--evidence-role", choices=("independent_test", "development"), default="independent_test")
    parser.add_argument(
        "--resume-existing-predictions", action="store_true",
        help="Resume bootstrap/reporting from immutable saved predictions after a computational interruption.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data_root, source_root = project / args.data_root, project / args.source_root
    run_root, output_root = project / args.run_root, project / args.output_root
    if args.resume_existing_predictions:
        if not output_root.is_dir():
            raise FileNotFoundError(f"Missing output directory to resume: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    selected_rows = {
        int(row["seed"]): row for row in run_summary
        if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"
    }
    source_freeze_path = source_root / "method_freeze.json"
    source_freeze = json.loads(source_freeze_path.read_text(encoding="utf-8"))
    source_by_seed = {int(row["seed"]): row for row in source_freeze["seed_freezes"]}
    method_freeze = {
        "experiment": "P2e Stage-B frozen weak-component and composition validation",
        "evidence_role": args.evidence_role,
        "source_method_freeze": str(source_freeze_path.resolve()),
        "source_method_freeze_sha256": sha256_file(source_freeze_path),
        "conditions": [
            {"id": index, "name": name, "target": TARGET_NAMES[class_index], "deficit_dB": deficit}
            for index, (name, class_index, deficit) in enumerate(CONDITIONS)
        ],
        "method_changes": "none; frozen P2c model, thresholds, routes, specialists and lazy rule",
        "promotion_gates": {
            "maximum_target_fpr": 0.10,
            "isrj_mean_recall_gain_vs_safe_min": 0.05,
            "sawfm_and_vft_parameter_nmae_improvement_min": 0.05,
            "three_of_three_seed_direction": True,
            "no_target_deficit_recall_decline_below": -0.02,
            "miss_penalty_and_worst_combination_nonworse": True,
            "identity_grouped_bootstrap_directional_ci": True,
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "test_used_at_freeze": False,
        "locked_test_used": False,
    }
    freeze_path = output_root / "method_freeze.json"
    if args.resume_existing_predictions:
        frozen_method = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen_method != _jsonable(method_freeze):
            raise RuntimeError("Resume refused: current arguments do not reproduce the frozen Stage-B method")
    else:
        freeze_path.write_text(json.dumps(_jsonable(method_freeze), indent=2), encoding="utf-8")
        (output_root / "test_access_started.json").write_text(json.dumps({
            "method_freeze_sha256": sha256_file(freeze_path), "started_unix": time.time(),
            "purpose": f"frozen Stage-B {args.evidence_role} weak-component evaluation",
        }, indent=2), encoding="utf-8")
    freeze_hash = sha256_file(freeze_path)

    files = {
        index: data_root / name / "compound_dataset_imbalanced.mat"
        for index, (name, _, _) in enumerate(CONDITIONS)
    }
    identities, labels = None, None
    condition_audit = {}
    for index, path in files.items():
        current_identities, current_labels = read_identity_metadata(path)
        if identities is None:
            identities, labels = current_identities, current_labels
        elif not np.array_equal(identities, current_identities) or not np.array_equal(labels, current_labels):
            raise RuntimeError(f"Stage-B paired identity/label alignment failed for {CONDITIONS[index][0]}")
        condition_audit[CONDITIONS[index][0]] = {
            "path": str(path.resolve()), "samples": int(len(current_identities)),
            "identity_sha256": hashlib.sha256(current_identities.astype("<i8").tobytes()).hexdigest(),
            "file_sha256": sha256_file(path),
        }
    assert identities is not None and labels is not None
    excluded_overlaps = []
    for raw_path in args.excluded_identity_files:
        excluded, _ = read_identity_metadata(project / raw_path)
        excluded_overlaps.append(int(np.intersect1d(identities, excluded).size))
    original_identities, _ = read_identity_metadata(protocol["data_file"])
    identity_audit = {
        "identity_count": int(len(identities)),
        "condition_count": len(CONDITIONS),
        "conditions_aligned": True,
        "original_overlap": int(np.intersect1d(identities, original_identities).size),
        "excluded_file_count": len(excluded_overlaps),
        "excluded_overlaps": excluded_overlaps,
        "conditions": condition_audit,
    }
    if identity_audit["original_overlap"] or any(excluded_overlaps):
        raise RuntimeError("Stage-B identity isolation failed")

    normalizer = load_normalizer(protocol["normalizer"])
    scale = normalizer.scale.numpy()
    indices = np.arange(len(labels), dtype=np.int64)
    results, bootstrap_rows = [], []
    variant_names = (
        "original_iq_grouped", "safe_calibrated_iq_grouped",
        "safe_isrj_route", "joint_physics_system",
    )
    if args.resume_existing_predictions:
        prediction_receipt = []
        for seed in MODEL_SEEDS:
            seed_dir = output_root / f"seed_{seed}"
            prediction_path = seed_dir / "test_per_sample.npz"
            result_path = seed_dir / "test_result.json"
            with np.load(prediction_path, allow_pickle=False) as archive:
                saved = {key: archive[key] for key in archive.files}
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["method_freeze_sha256"] != freeze_hash:
                raise RuntimeError(f"Resume refused: seed {seed} prediction freeze hash mismatch")
            records_by_variant = {name: load_record(saved, name, scale) for name in variant_names}
            results.append(result)
            bootstrap_rows.append({"identity": saved["receiver_noise_seed"], "records": records_by_variant})
            prediction_receipt.append({
                "seed": seed,
                "prediction_sha256": sha256_file(prediction_path),
                "result_sha256": sha256_file(result_path),
            })
        (output_root / "bootstrap_resume_receipt.json").write_text(json.dumps({
            "reason": "original bootstrap implementation repeatedly rescanned all rows per sampled identity",
            "predictions_recomputed": False,
            "models_thresholds_routes_or_experts_changed": False,
            "method_freeze_sha256": freeze_hash,
            "prediction_artifacts": prediction_receipt,
            "resume_unix": time.time(),
        }, indent=2), encoding="utf-8")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in MODEL_SEEDS:
            seed_dir = output_root / f"seed_{seed}"
            seed_dir.mkdir()
            seed_freeze = source_by_seed[seed]
            main_model, _ = _load_main(selected_rows[seed], protocol, project, device)
            unet = _load_unet(Path(seed_freeze["unet_checkpoint"]), device)
            saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
            saw.load_state_dict(torch.load(seed_freeze["saw_checkpoint"], map_location=device, weights_only=True))
            saw.eval()
            frozen = joblib.load(source_root / f"seed_{seed}" / "frozen_models.joblib")
            routed_thresholds = np.asarray(seed_freeze["routed_thresholds"], dtype=np.float32)
            records, timing = collect_joint_records(
                main_model, unet, frozen["experts"], saw,
                float(seed_freeze["saw_mean"]), float(seed_freeze["saw_scale"]),
                files, labels, indices, args.batch_size, seed + 200000,
                device, normalizer, tuple(files), routed_thresholds,
            )
            payload = role_payload(records, indices)
            original_thresholds = np.asarray(seed_freeze["original_thresholds"], dtype=np.float32)
            safe_thresholds = np.asarray(seed_freeze["safe_thresholds"], dtype=np.float32)
            scores = ensemble_scores(frozen["ensemble"], payload, float(original_thresholds[ISRJ_GLOBAL]))
            routed_probability, route_accept = apply_route(
                payload, scores["lower"], seed_freeze["isrj_selection"], 1.0,
            )
            joint_prediction = fused_prediction(payload)
            variant_arrays = {
                "original_iq_grouped": (payload["main"], original_thresholds, payload["prediction"]),
                "safe_calibrated_iq_grouped": (payload["main"], safe_thresholds, payload["prediction"]),
                "safe_isrj_route": (routed_probability, routed_thresholds, payload["prediction"]),
                "joint_physics_system": (routed_probability, routed_thresholds, joint_prediction),
            }
            standard = {
                name: evaluate_variant(payload, probability, thresholds, prediction, scale, tuple(files))
                for name, (probability, thresholds, prediction) in variant_arrays.items()
            }
            saved = {
                "sample_index": payload["sample_indices"],
                "receiver_noise_seed": identities[payload["sample_indices"]],
                "condition_id": payload["power"], "labels": payload["labels"],
                "target": payload["target"], "mask": payload["mask"],
                "original_probability": payload["main"], "routed_probability": routed_probability,
                "base_prediction": payload["prediction"], "joint_prediction": joint_prediction,
                "original_thresholds": original_thresholds, "safe_thresholds": safe_thresholds,
                "routed_thresholds": routed_thresholds,
            }
            np.savez_compressed(seed_dir / "test_per_sample.npz", **saved)
            records_by_variant = {name: load_record(saved, name, scale) for name in variant_arrays}
            custom = {
                name: {
                    CONDITIONS[condition_id][0]: condition_metrics(record, condition_id, CONDITIONS[condition_id][1])
                    for condition_id in files
                }
                for name, record in records_by_variant.items()
            }
            result = {
                "seed": seed, "standard_metrics": standard, "target_condition_metrics": custom,
                "route_accept_coverage": float(route_accept.mean()), "timing": timing,
                "method_freeze_sha256": freeze_hash, "locked_test_used": False,
            }
            (seed_dir / "test_result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
            results.append(result)
            bootstrap_rows.append({"identity": saved["receiver_noise_seed"], "records": records_by_variant})
            del main_model, unet, saw
            if device.type == "cuda":
                torch.cuda.empty_cache()

    grouped_ids = {
        "SawFM": (1, (0, 1, 2)), "VFT": (4, (3, 4, 5)), "ISRJ": (5, (6, 7, 8)),
    }
    comparisons = {
        "isrj_recall_joint_vs_safe": grouped_bootstrap(
            bootstrap_rows, 5, (6, 7, 8), "recall", "joint_physics_system",
            "safe_calibrated_iq_grouped", args.bootstrap_replicates, 20260841,
        ),
        "sawfm_parameter_joint_vs_safe_route": grouped_bootstrap(
            bootstrap_rows, 1, (0, 1, 2), "predicted_gate_nmae", "joint_physics_system",
            "safe_isrj_route", args.bootstrap_replicates, 20260842,
        ),
        "vft_parameter_joint_vs_safe_route": grouped_bootstrap(
            bootstrap_rows, 4, (3, 4, 5), "predicted_gate_nmae", "joint_physics_system",
            "safe_isrj_route", args.bootstrap_replicates, 20260843,
        ),
    }
    for target_name, (class_index, condition_ids) in grouped_ids.items():
        comparisons[f"{target_name.lower()}_miss_joint_vs_original"] = grouped_bootstrap(
            bootstrap_rows, class_index, condition_ids, "miss_penalty_nmae", "joint_physics_system",
            "original_iq_grouped", args.bootstrap_replicates, 20260900 + class_index,
        )

    def per_seed_delta(class_index, condition_ids, metric, candidate, baseline):
        return [
            target_metric(row["records"][candidate], class_index, condition_ids, metric)
            - target_metric(row["records"][baseline], class_index, condition_ids, metric)
            for row in bootstrap_rows
        ]

    isrj_recall = per_seed_delta(5, (6, 7, 8), "recall", "joint_physics_system", "safe_calibrated_iq_grouped")
    saw_delta = per_seed_delta(1, (0, 1, 2), "predicted_gate_nmae", "joint_physics_system", "safe_isrj_route")
    vft_delta = per_seed_delta(4, (3, 4, 5), "predicted_gate_nmae", "joint_physics_system", "safe_isrj_route")
    saw_base = [target_metric(row["records"]["safe_isrj_route"], 1, (0, 1, 2), "predicted_gate_nmae") for row in bootstrap_rows]
    vft_base = [target_metric(row["records"]["safe_isrj_route"], 4, (3, 4, 5), "predicted_gate_nmae") for row in bootstrap_rows]
    saw_improvement = [-delta / base for delta, base in zip(saw_delta, saw_base)]
    vft_improvement = [-delta / base for delta, base in zip(vft_delta, vft_base)]
    maximum_fpr = max(
        result["target_condition_metrics"]["joint_physics_system"][name]["fpr"]
        for result in results for name, _, _ in CONDITIONS
    )
    per_deficit_recall_delta = {
        name: float(np.mean([
            result["target_condition_metrics"]["joint_physics_system"][name]["recall"]
            - result["target_condition_metrics"]["safe_calibrated_iq_grouped"][name]["recall"]
            for result in results
        ]))
        for name, _, _ in CONDITIONS
    }
    miss_nonworse = all(
        comparisons[f"{target.lower()}_miss_joint_vs_original"]["observed_mean_delta"] <= 0
        for target in grouped_ids
    )
    worst_nonworse = all(
        np.mean([
            result["target_condition_metrics"]["joint_physics_system"][name]["worst_combination_predicted_gate_nmae"]
            - result["target_condition_metrics"]["safe_isrj_route"][name]["worst_combination_predicted_gate_nmae"]
            for result in results for name, class_index, _ in CONDITIONS if class_index == target_class
        ]) <= 0
        for target_class, _ in grouped_ids.values()
    )
    gates = {
        "maximum_target_fpr_at_most_0p10": bool(maximum_fpr <= 0.10 + 1e-12),
        "isrj_mean_recall_gain_at_least_0p05": bool(np.mean(isrj_recall) >= 0.05),
        "isrj_three_of_three_positive": bool(np.all(np.asarray(isrj_recall) > 0)),
        "sawfm_parameter_improvement_at_least_0p05": bool(np.mean(saw_improvement) >= 0.05),
        "sawfm_three_of_three_parameter_improvement": bool(np.all(np.asarray(saw_improvement) > 0)),
        "vft_parameter_improvement_at_least_0p05": bool(np.mean(vft_improvement) >= 0.05),
        "vft_three_of_three_parameter_improvement": bool(np.all(np.asarray(vft_improvement) > 0)),
        "no_target_deficit_recall_decline_below_minus_0p02": bool(min(per_deficit_recall_delta.values()) >= -0.02),
        "target_miss_penalty_nonworse": bool(miss_nonworse),
        "target_worst_combination_nonworse": bool(worst_nonworse),
        "bootstrap_directional_ci": bool(
            comparisons["isrj_recall_joint_vs_safe"]["ci95_low"] > 0
            and comparisons["sawfm_parameter_joint_vs_safe_route"]["ci95_high"] < 0
            and comparisons["vft_parameter_joint_vs_safe_route"]["ci95_high"] < 0
            and all(comparisons[f"{target.lower()}_miss_joint_vs_original"]["ci95_high"] < 0 for target in grouped_ids)
        ),
    }
    summary = {
        "experiment": method_freeze["experiment"],
        "recognition": {
            "mean_isrj_recall_gain": float(np.mean(isrj_recall)),
            "per_seed_isrj_recall_gain": isrj_recall,
            "maximum_target_fpr": float(maximum_fpr),
            "per_condition_recall_delta": per_deficit_recall_delta,
        },
        "parameters": {
            "mean_sawfm_predicted_gate_nmae_improvement": float(np.mean(saw_improvement)),
            "per_seed_sawfm_improvement": saw_improvement,
            "mean_vft_predicted_gate_nmae_improvement": float(np.mean(vft_improvement)),
            "per_seed_vft_improvement": vft_improvement,
        },
        "paired_bootstrap": comparisons,
        "gates": gates,
        "stage_b_validated": bool(all(gates.values())),
        "identity_audit": identity_audit,
        "method_freeze_sha256": freeze_hash,
        "source_method_freeze_sha256": method_freeze["source_method_freeze_sha256"],
        "test_used_once": bool(args.evidence_role == "independent_test"),
        "development_used": bool(args.evidence_role == "development"),
        "test_fit_or_selection": False,
        "bootstrap_resumed_from_saved_predictions": bool(args.resume_existing_predictions),
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
