"""Post-hoc power-robust calibration of the frozen task router.

The deployable candidates use only frozen model outputs.  An oracle-level
variant is retained strictly as a diagnostic upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import POWER_LEVELS, audit_power_alignment, discover_power_train_files, read_identity_metadata
from cj0724.identity_safe_power_parameter_eval import _collect, _level_metrics, _saw_prediction, _weak_mask
from cj0724.multitask import PARAMETER_COLUMNS, build_multitask_model
from cj0724.paper_core_evidence import load_normalizer
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.run_physics_parameter_specialists import _positive, _stratified_limit


def _sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _concat_role(by_level: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = next(iter(by_level.values())).keys()
    output = {key: np.concatenate([by_level[level][key] for level in POWER_LEVELS]) for key in keys}
    output["level"] = np.concatenate([
        np.full(len(by_level[level]["labels"]), level, dtype=np.int64) for level in POWER_LEVELS
    ])
    return output


def _visibility_features(record: dict[str, np.ndarray], target_index: int, variant: str) -> np.ndarray:
    class_index = target_index // 2
    prediction_z = record["prediction_z"][:, target_index : target_index + 1]
    if variant == "global":
        return prediction_z
    if variant == "visibility":
        logits = record["logits"]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        summaries = np.stack(
            [
                probabilities[:, :3].mean(axis=1),
                probabilities[:, 3:].mean(axis=1),
                probabilities[:, :3].max(axis=1),
                probabilities[:, 3:].max(axis=1),
                probabilities[:, class_index],
            ],
            axis=1,
        )
        return np.concatenate([prediction_z, logits, summaries], axis=1)
    if variant == "oracle_level":
        level_code = np.stack([(record["level"] == level).astype(np.float32) for level in POWER_LEVELS], axis=1)
        return np.concatenate([prediction_z, level_code], axis=1)
    raise ValueError(f"unknown calibrator variant {variant!r}")


def fit_parameter_calibrator(
    record: dict[str, np.ndarray], variant: str, alpha: float
) -> list[object]:
    models = []
    labels = record["labels"].astype(bool)
    mask = record["mask"].astype(bool).reshape(len(labels), 12)
    target_z = record["target_z"]
    for target_index in range(12):
        class_index = target_index // 2
        valid = labels[:, class_index] & mask[:, target_index]
        feature = _visibility_features(record, target_index, variant)[valid]
        residual = target_z[valid, target_index] - record["prediction_z"][valid, target_index]
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(feature, residual)
        models.append(model)
    return models


def apply_parameter_calibrator(
    record: dict[str, np.ndarray], models: list[object], variant: str, gamma: float
) -> np.ndarray:
    output = record["prediction_z"].copy()
    for target_index, model in enumerate(models):
        correction = np.asarray(model.predict(_visibility_features(record, target_index, variant)))
        output[:, target_index] += float(gamma) * np.clip(correction, -4.0, 4.0)
    return output


def gate_features(logits: np.ndarray) -> np.ndarray:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    summaries = np.stack(
        [
            probabilities[:, :3].mean(axis=1),
            probabilities[:, 3:].mean(axis=1),
            probabilities[:, :3].max(axis=1),
            probabilities[:, 3:].max(axis=1),
        ],
        axis=1,
    )
    return np.concatenate([logits, summaries], axis=1)


def fit_logistic_gate(record: dict[str, np.ndarray], c_value: float) -> list[object]:
    feature = gate_features(record["logits"])
    models = []
    for class_index in range(6):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(c_value), class_weight="balanced", max_iter=1000, random_state=20260725),
        )
        model.fit(feature, record["labels"][:, class_index])
        models.append(model)
    return models


def apply_logistic_gate(record: dict[str, np.ndarray], models: list[object]) -> np.ndarray:
    feature = gate_features(record["logits"])
    return np.stack([model.predict_proba(feature)[:, 1] for model in models], axis=1)


def fbeta_thresholds(labels: np.ndarray, probability: np.ndarray, beta: float) -> np.ndarray:
    beta2 = float(beta) ** 2
    thresholds = np.full(6, 0.5, dtype=np.float32)
    grid = np.arange(0.02, 0.951, 0.01)
    for class_index in range(6):
        truth = labels[:, class_index].astype(bool)
        best = (-1.0, -1.0, -1.0)
        for threshold in grid:
            predicted = probability[:, class_index] >= threshold
            tp = float(np.logical_and(predicted, truth).sum())
            fp = float(np.logical_and(predicted, ~truth).sum())
            fn = float(np.logical_and(~predicted, truth).sum())
            score = (1.0 + beta2) * tp / max((1.0 + beta2) * tp + beta2 * fn + fp, 1.0)
            candidate = (score, -abs(float(threshold) - 0.5), float(threshold))
            if candidate > best:
                best = candidate
        thresholds[class_index] = best[2]
    return thresholds


def side_thresholds(record: dict[str, np.ndarray], probability: np.ndarray, beta: float) -> dict[str, np.ndarray]:
    output = {}
    for name, selected in (
        ("low", record["level"] < 30),
        ("mid", record["level"] == 30),
        ("high", record["level"] > 30),
    ):
        output[name] = fbeta_thresholds(record["labels"][selected], probability[selected], beta)
    return output


def thresholds_for_level(value: object, level: int) -> np.ndarray:
    if isinstance(value, dict):
        return value["low" if level < 30 else "mid" if level == 30 else "high"]
    return np.asarray(value, dtype=np.float32)


def _physical_prediction(prediction_z: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (prediction_z * scale.reshape(1, 12) + mean.reshape(1, 12)).reshape(-1, 6, 2)


def evaluate_method(
    by_level: dict[int, dict[str, np.ndarray]],
    prediction_z_by_level: dict[int, np.ndarray],
    probabilities_by_level: dict[int, np.ndarray],
    thresholds: object,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    correlations = []
    for level in POWER_LEVELS:
        source = by_level[level]
        prediction = _physical_prediction(prediction_z_by_level[level], mean, scale)
        probability = probabilities_by_level[level]
        threshold = thresholds_for_level(thresholds, level)
        record = {
            "sample_indices": source["sample_indices"], "labels": source["labels"].astype(np.uint8),
            "probabilities": probability, "prediction": prediction, "target": source["target"],
            "mask": source["mask"].astype(np.uint8), "thresholds": threshold,
            "scale": scale.reshape(6, 2),
        }
        level_metrics = _level_metrics(record, _weak_mask(source["labels"].astype(bool), level))
        valid = source["mask"].astype(bool) & source["labels"].astype(bool)[:, :, None]
        target_correlations = []
        for class_index in range(6):
            for parameter_index in range(2):
                selected = valid[:, class_index, parameter_index]
                pred = prediction[selected, class_index, parameter_index]
                truth = source["target"][selected, class_index, parameter_index]
                if len(pred) >= 2 and np.std(pred) > 0 and np.std(truth) > 0:
                    target_correlations.append(float(np.corrcoef(pred, truth)[0, 1]))
        level_metrics["oracle_parameter_macro_pearson"] = float(np.mean(target_correlations)) if target_correlations else 0.0
        correlations.append(level_metrics["oracle_parameter_macro_pearson"])
        rows.append({"power_level_db": level, **level_metrics})
    mean_miss = float(np.mean([row["miss_penalty_parameter_nmae"] for row in rows]))
    mean_gate = float(np.mean([row["predicted_gate_parameter_nmae"] for row in rows]))
    mean_oracle = float(np.mean([row["oracle_parameter_nmae"] for row in rows]))
    recall10 = float(next(row for row in rows if row["power_level_db"] == 10)["weak_component_recall"])
    recall50 = float(next(row for row in rows if row["power_level_db"] == 50)["weak_component_recall"])
    miss10 = float(next(row for row in rows if row["power_level_db"] == 10)["miss_penalty_parameter_nmae"])
    miss50 = float(next(row for row in rows if row["power_level_db"] == 50)["miss_penalty_parameter_nmae"])
    aupr30 = float(next(row for row in rows if row["power_level_db"] == 30)["classification_macro_aupr"])
    fpr = float(np.mean([row["false_positive_gate_rate"] for row in rows]))
    summary = {
        "nine_level_mean_oracle_parameter_nmae": mean_oracle,
        "nine_level_mean_predicted_gate_parameter_nmae": mean_gate,
        "nine_level_mean_miss_penalty_parameter_nmae": mean_miss,
        "weak_recall_10db": recall10, "weak_recall_50db": recall50,
        "miss_penalty_nmae_10db": miss10, "miss_penalty_nmae_50db": miss50,
        "classification_macro_aupr_30db": aupr30, "nine_level_mean_false_positive_gate_rate": fpr,
        "nine_level_mean_oracle_parameter_macro_pearson": float(np.mean(correlations)),
    }
    return rows, summary


def selection_score(summary: dict[str, float]) -> float:
    recall_penalty = 2.0 * max(0.0, 0.8 - summary["weak_recall_10db"]) + 2.0 * max(0.0, 0.8 - summary["weak_recall_50db"])
    extreme_penalty = max(0.0, summary["miss_penalty_nmae_10db"] - 1.0) + max(0.0, summary["miss_penalty_nmae_50db"] - 1.0)
    return summary["nine_level_mean_miss_penalty_parameter_nmae"] + recall_penalty + extreme_penalty + 0.2 * summary["nine_level_mean_false_positive_gate_rate"]


def prepare_record(record: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, level: int) -> dict[str, np.ndarray]:
    output = dict(record)
    output["prediction_z"] = ((record["prediction"].reshape(-1, 12) - mean.reshape(1, 12)) / scale.reshape(1, 12)).astype(np.float64)
    output["target_z"] = ((record["target"].reshape(-1, 12) - mean.reshape(1, 12)) / scale.reshape(1, 12)).astype(np.float64)
    output["level"] = np.full(len(record["labels"]), int(level), dtype=np.int64)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/power_robustness_calibration_20260725")
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root, specialist_root, output_root = Path(args.run_root), Path(args.specialist_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    roles = np.load(run_root / "role_indices.npz")
    files = discover_power_train_files(args.power_root)
    alignment = audit_power_alignment(files)
    root_seeds, root_labels = read_identity_metadata(protocol["data_file"])
    power_seeds, power_labels = read_identity_metadata(files[30])
    if not np.array_equal(root_seeds, power_seeds) or not np.array_equal(root_labels, power_labels):
        raise RuntimeError("root and power identities differ")
    normalizer = load_normalizer(protocol["normalizer"])
    mean = normalizer.mean.numpy().reshape(12).astype(np.float64)
    scale = normalizer.scale.numpy().reshape(12).astype(np.float64)
    with h5py.File(protocol["data_file"], "r") as handle:
        root_parameter_values = np.asarray(handle["parameter_values"], dtype=np.float64)
        root_parameter_mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)

    selected_rows = [row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"]
    all_seed_results = []
    for selected_row in selected_rows:
        seed = int(selected_row["seed"])
        model = build_multitask_model(
            selected_row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
            protocol["parameter_hidden_dim"], protocol["legacy_attention"],
        ).to(device)
        model.load_state_dict(torch.load(selected_row["checkpoint"], map_location=device, weights_only=False)["model_state"])
        saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        saw.load_state_dict(torch.load(specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt", map_location=device, weights_only=True))
        train_positive = _positive(roles["train"], root_labels, root_parameter_mask, 1, 14)
        saw_train = _stratified_limit(train_positive, root_parameter_values[14], 2048, seed + 101, "scalar")
        saw_mean, saw_scale = float(np.mean(root_parameter_values[14, saw_train])), float(np.std(root_parameter_values[14, saw_train]))

        role_records: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        for role in ("early", "calibration", "tune", "confirm"):
            by_level = {}
            for level in POWER_LEVELS:
                dataset = V73IQDataset(files[level], roles[role], PARAMETER_COLUMNS)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
                record = _collect(model, loader, device, normalizer)
                record["prediction"][:, 1, 1] = _saw_prediction(saw, loader, device, saw_mean, saw_scale)
                by_level[level] = prepare_record(record, mean, scale, level)
                dataset.close()
            role_records[role] = by_level
        combined = {role: _concat_role(by_level) for role, by_level in role_records.items()}

        parameter_candidates = {}
        raw_predictions = {level: role_records["tune"][level]["prediction_z"] for level in POWER_LEVELS}
        raw_probabilities = {level: 1.0 / (1.0 + np.exp(-np.clip(role_records["tune"][level]["logits"], -30, 30))) for level in POWER_LEVELS}
        base_thresholds = np.asarray(json.loads(selected_row["thresholds"]), dtype=np.float32)
        _, raw_tune_summary = evaluate_method(role_records["tune"], raw_predictions, raw_probabilities, base_thresholds, mean, scale)
        parameter_candidates["raw"] = {"models": None, "variant": "raw", "alpha": None, "gamma": 0.0, "tune": raw_tune_summary}
        for variant in ("global", "visibility", "oracle_level"):
            best = None
            for alpha in (0.1, 1.0, 10.0, 100.0):
                models = fit_parameter_calibrator(combined["early"], variant, alpha)
                for gamma in (0.25, 0.5, 0.75, 1.0):
                    tune_predictions = {level: apply_parameter_calibrator(role_records["tune"][level], models, variant, gamma) for level in POWER_LEVELS}
                    _, tune_summary = evaluate_method(role_records["tune"], tune_predictions, raw_probabilities, base_thresholds, mean, scale)
                    candidate = (tune_summary["nine_level_mean_oracle_parameter_nmae"], alpha, gamma, models, tune_summary)
                    if best is None or candidate[:3] < best[:3]:
                        best = candidate
            parameter_candidates[variant] = {"models": best[3], "variant": variant, "alpha": best[1], "gamma": best[2], "tune": best[4]}

        calibration_base_probability = 1.0 / (1.0 + np.exp(-np.clip(combined["calibration"]["logits"], -30, 30)))
        gate_candidates = {"base": {"models": None, "thresholds": base_thresholds, "c": None, "beta": None}}
        for beta in (1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0):
            threshold = fbeta_thresholds(combined["calibration"]["labels"], calibration_base_probability, beta)
            name = f"global_f{beta:g}"
            gate_candidates[name] = {"models": None, "thresholds": threshold, "c": None, "beta": beta}
            gate_candidates[f"oracle_side_f{beta:g}"] = {"models": None, "thresholds": side_thresholds(combined["calibration"], calibration_base_probability, beta), "c": None, "beta": beta}
        for c_value in (0.1, 1.0, 10.0):
            gate_models = fit_logistic_gate(combined["early"], c_value)
            calibration_probability = apply_logistic_gate(combined["calibration"], gate_models)
            for beta in (1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0):
                gate_candidates[f"logistic_c{c_value:g}_f{beta:g}"] = {
                    "models": gate_models,
                    "thresholds": fbeta_thresholds(combined["calibration"]["labels"], calibration_probability, beta),
                    "c": c_value, "beta": beta,
                }

        for name, candidate in gate_candidates.items():
            tune_probability = {
                level: (
                    raw_probabilities[level]
                    if candidate["models"] is None
                    else apply_logistic_gate(role_records["tune"][level], candidate["models"])
                ) for level in POWER_LEVELS
            }
            _, tune_summary = evaluate_method(role_records["tune"], raw_predictions, tune_probability, candidate["thresholds"], mean, scale)
            candidate["tune"] = tune_summary
            candidate["selection_score"] = selection_score(tune_summary)
        deployable_gate_names = [name for name in gate_candidates if not name.startswith("oracle_side")]
        selected_gate_name = min(deployable_gate_names, key=lambda name: gate_candidates[name]["selection_score"])
        guarded_fpr_cap = gate_candidates["base"]["tune"]["nine_level_mean_false_positive_gate_rate"] + 0.05
        guarded_gate_names = [
            name for name in deployable_gate_names
            if gate_candidates[name]["tune"]["nine_level_mean_false_positive_gate_rate"] <= guarded_fpr_cap
        ]
        selected_guarded_gate_name = min(guarded_gate_names, key=lambda name: gate_candidates[name]["selection_score"])
        selected_oracle_gate_name = min(
            [name for name in gate_candidates if name.startswith("oracle_side")],
            key=lambda name: gate_candidates[name]["selection_score"],
        )

        methods = {
            "constant_train_mean": ({"models": None, "variant": "constant", "alpha": None, "gamma": 0.0}, gate_candidates["base"], False),
            "baseline_saw": (parameter_candidates["raw"], gate_candidates["base"], False),
            "global_parameter": (parameter_candidates["global"], gate_candidates["base"], False),
            "visibility_parameter": (parameter_candidates["visibility"], gate_candidates["base"], False),
            "selected_gate": (parameter_candidates["raw"], gate_candidates[selected_gate_name], False),
            "joint_deployable": (parameter_candidates["visibility"], gate_candidates[selected_gate_name], False),
            "guarded_gate": (parameter_candidates["raw"], gate_candidates[selected_guarded_gate_name], False),
            "joint_guarded": (parameter_candidates["visibility"], gate_candidates[selected_guarded_gate_name], False),
            "oracle_power_upper": (parameter_candidates["oracle_level"], gate_candidates[selected_oracle_gate_name], True),
        }
        seed_methods = []
        for method_name, (parameter_candidate, gate_candidate, oracle_power_used) in methods.items():
            confirm_prediction = {
                level: (
                    np.zeros_like(role_records["confirm"][level]["prediction_z"])
                    if parameter_candidate["variant"] == "constant"
                    else role_records["confirm"][level]["prediction_z"]
                    if parameter_candidate["models"] is None
                    else apply_parameter_calibrator(
                        role_records["confirm"][level], parameter_candidate["models"],
                        parameter_candidate["variant"], parameter_candidate["gamma"],
                    )
                ) for level in POWER_LEVELS
            }
            confirm_probability = {
                level: (
                    1.0 / (1.0 + np.exp(-np.clip(role_records["confirm"][level]["logits"], -30, 30)))
                    if gate_candidate["models"] is None
                    else apply_logistic_gate(role_records["confirm"][level], gate_candidate["models"])
                ) for level in POWER_LEVELS
            }
            level_rows, method_summary = evaluate_method(
                role_records["confirm"], confirm_prediction, confirm_probability,
                gate_candidate["thresholds"], mean, scale,
            )
            seed_methods.append({
                "method": method_name, "seed": seed, "oracle_power_used": oracle_power_used,
                "parameter_variant": parameter_candidate["variant"], "parameter_alpha": parameter_candidate["alpha"],
                "parameter_gamma": parameter_candidate["gamma"], "gate_name": selected_gate_name if method_name in ("selected_gate", "joint_deployable") else selected_guarded_gate_name if method_name in ("guarded_gate", "joint_guarded") else selected_oracle_gate_name if oracle_power_used else "base",
                "gate_c": gate_candidate["c"], "gate_beta": gate_candidate["beta"],
                **method_summary, "per_level": level_rows,
            })
        all_seed_results.extend(seed_methods)
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "selection.json").write_text(json.dumps({
            "parameter_candidates": {name: {key: value for key, value in candidate.items() if key != "models"} for name, candidate in parameter_candidates.items()},
            "gate_candidates": {name: {key: value.tolist() if isinstance(value, np.ndarray) else {k: v.tolist() for k, v in value.items()} if isinstance(value, dict) and key == "thresholds" else value for key, value in candidate.items() if key != "models"} for name, candidate in gate_candidates.items()},
            "selected_deployable_gate": selected_gate_name, "selected_guarded_gate": selected_guarded_gate_name,
            "guarded_fpr_cap": guarded_fpr_cap, "selected_oracle_gate": selected_oracle_gate_name,
        }, indent=2), encoding="utf-8")
        del model, saw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregates = []
    for method_name in sorted({row["method"] for row in all_seed_results}):
        rows = [row for row in all_seed_results if row["method"] == method_name]
        keys = [
            "nine_level_mean_oracle_parameter_nmae", "nine_level_mean_predicted_gate_parameter_nmae",
            "nine_level_mean_miss_penalty_parameter_nmae", "weak_recall_10db", "weak_recall_50db",
            "miss_penalty_nmae_10db", "miss_penalty_nmae_50db", "classification_macro_aupr_30db",
            "nine_level_mean_false_positive_gate_rate",
            "nine_level_mean_oracle_parameter_macro_pearson",
        ]
        entry = {"method": method_name, "seed_count": len(rows), "oracle_power_used": bool(rows[0]["oracle_power_used"])}
        for key in keys:
            values = [row[key] for row in rows]
            entry[f"{key}_mean"] = float(np.mean(values))
            entry[f"{key}_std"] = float(np.std(values))
        entry["three_seed_oracle_nmae_improvement"] = None
        aggregates.append(entry)
    baseline = next(row for row in aggregates if row["method"] == "baseline_saw")
    for entry in aggregates:
        entry["relative_oracle_nmae_reduction_vs_baseline"] = float(
            1.0 - entry["nine_level_mean_oracle_parameter_nmae_mean"] / baseline["nine_level_mean_oracle_parameter_nmae_mean"]
        )
        entry["relative_gate_nmae_reduction_vs_baseline"] = float(
            1.0 - entry["nine_level_mean_predicted_gate_parameter_nmae_mean"] / baseline["nine_level_mean_predicted_gate_parameter_nmae_mean"]
        )
        entry["relative_miss_penalty_reduction_vs_baseline"] = float(
            1.0 - entry["nine_level_mean_miss_penalty_parameter_nmae_mean"] / baseline["nine_level_mean_miss_penalty_parameter_nmae_mean"]
        )
        method_rows = [row for row in all_seed_results if row["method"] == entry["method"]]
        base_by_seed = {row["seed"]: row for row in all_seed_results if row["method"] == "baseline_saw"}
        entry["three_seed_same_direction_miss_penalty"] = bool(all(
            row["nine_level_mean_miss_penalty_parameter_nmae"] < base_by_seed[row["seed"]]["nine_level_mean_miss_penalty_parameter_nmae"]
            for row in method_rows
        )) if entry["method"] != "baseline_saw" else True
        entry["locked_test_gate_passed"] = bool(
            not entry["oracle_power_used"]
            and entry["relative_gate_nmae_reduction_vs_baseline"] >= 0.10
            and entry["three_seed_same_direction_miss_penalty"]
            and entry["weak_recall_10db_mean"] >= 0.80
            and entry["weak_recall_50db_mean"] >= 0.80
            and entry["miss_penalty_nmae_10db_mean"] <= 1.0
            and entry["miss_penalty_nmae_50db_mean"] <= 1.0
            and entry["classification_macro_aupr_30db_mean"] >= baseline["classification_macro_aupr_30db_mean"] - 0.005
        )
        entry["posthoc_deployment_safety_passed"] = bool(
            entry["locked_test_gate_passed"]
            and entry["nine_level_mean_false_positive_gate_rate_mean"]
            <= baseline["nine_level_mean_false_positive_gate_rate_mean"] + 0.05
        )
    payload = {
        "protocol": {
            "frozen_base": "MLCNN classification + IQ-Grouped parameter + SawFM period replacement",
            "fit_role": "early", "threshold_role": "calibration", "selection_role": "tune", "final_role": "confirm",
            "identity_key": "receiver_noise_seed", "levels_db": list(POWER_LEVELS),
            "deployable_visibility_features": "frozen logits and prediction only",
            "oracle_power_upper_bound": "uses true power level/side and is not a deployment candidate",
            "posthoc_safety_audit": "false-positive gate rate must not exceed the frozen baseline by more than 0.05; discovered after the first confirm run and not claimed as preregistered",
            "locked_test_used": False,
        },
        "role_hashes": {role: _sha(roles[role]) for role in ("early", "calibration", "tune", "confirm")},
        "power_alignment": alignment, "per_seed": all_seed_results, "aggregate": aggregates,
    }
    (output_root / "power_robustness_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "aggregate": aggregates}, indent=2))


if __name__ == "__main__":
    main()
