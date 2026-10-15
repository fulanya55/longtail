"""P1e: transition-region physical feature sufficiency screen.

The experiment freezes the P1c/P1d main network, U-Net and three experts.  It
uses only the independent P1d *train* identities at 20 and 25 dB, creates a
new combination-stratified fit/calibration/probe split, and asks whether
target-specific physical features improve oracle benefit/risk prediction over
the existing probability-only router features.  The consumed independent test
and all locked tests are deliberately inaccessible from this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.ensemble import HistGradientBoostingClassifier

from cj0724.dual_axis import discover_power_train_files, read_identity_metadata
from cj0724.hierarchical_np_risk_router import build_risk_benefit_targets
from cj0724.independent_worst_power_calibration import discover_split_files
from cj0724.integrate_swept_experts_with_six_class import (
    TARGET_INDICES,
    TARGET_NAMES,
    _jsonable,
    _load_main,
    _load_unet,
    align_payloads,
    collect_main,
    expert_probabilities,
    extract_unet_role,
    fit_frozen_experts,
)
from cj0724.oracle_swept_isolation_experiment import set_seed
from cj0724.paper_core_evidence import load_normalizer
from cj0724.three_state_negative_protection_router import three_state_features


TRANSITION_POWERS = (20, 25)
FEATURE_VARIANTS = ("probability_only", "physical_only", "probability_plus_physical")


class ConstantBinaryModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        value = np.full(len(features), self.probability, dtype=np.float64)
        return np.column_stack((1.0 - value, value))


def _hash_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def stratified_identity_split(labels: np.ndarray, seed: int = 20260731) -> dict[str, np.ndarray]:
    """Split every label combination 10/5/5 into fit/calibration/probe."""
    labels = np.asarray(labels, dtype=np.uint8)
    _, inverse = np.unique(labels, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    roles = {"fit": [], "calibration": [], "probe": []}
    for group in range(int(inverse.max()) + 1):
        indices = np.flatnonzero(inverse == group)
        if len(indices) != 20:
            raise ValueError(f"Expected 20 identities per combination, got {len(indices)}")
        indices = rng.permutation(indices)
        roles["fit"].extend(indices[:10])
        roles["calibration"].extend(indices[10:15])
        roles["probe"].extend(indices[15:20])
    return {name: np.sort(np.asarray(values, dtype=np.int64)) for name, values in roles.items()}


def _fit_binary(features: np.ndarray, target: np.ndarray, seed: int):
    target = np.asarray(target, dtype=np.uint8)
    if np.unique(target).size < 2:
        return ConstantBinaryModel(float(target[0]))
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=7,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(np.asarray(features, dtype=np.float32), target)
    return model


def binary_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    prevalence = float(target.mean())
    if np.unique(target).size < 2:
        return {"prevalence": prevalence, "auroc": float("nan"), "aupr": float("nan"), "aupr_lift": float("nan")}
    aupr = float(sk_metrics.average_precision_score(target, score))
    return {
        "prevalence": prevalence,
        "auroc": float(sk_metrics.roc_auc_score(target, score)),
        "aupr": aupr,
        "aupr_lift": float(aupr / max(prevalence, 1e-12)),
    }


def select_veto_threshold(target: np.ndarray, score: np.ndarray, minimum_recall: float = 0.90) -> dict[str, float | bool]:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    if not target.any():
        return {"threshold": 1.1, "recall": float("nan"), "precision": float("nan"), "veto_coverage": 0.0, "feasible": False}
    candidates = np.unique(np.concatenate(([1.1], score, [-0.1])))
    rows = []
    for threshold in candidates:
        veto = score >= threshold
        recall = float(veto[target].mean())
        if recall + 1e-12 < minimum_recall:
            continue
        precision = float(target[veto].mean()) if veto.any() else 0.0
        rows.append((float(veto.mean()), -precision, float(threshold), recall, precision))
    if not rows:
        return {"threshold": -0.1, "recall": 1.0, "precision": float(target.mean()), "veto_coverage": 1.0, "feasible": False}
    coverage, _, threshold, recall, precision = min(rows)
    return {"threshold": threshold, "recall": recall, "precision": precision, "veto_coverage": coverage, "feasible": True}


def route_metrics(
    labels: np.ndarray,
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    accept: np.ndarray,
    local_target: int,
    threshold: float,
    alpha: float = 0.75,
) -> dict[str, float]:
    global_target = int(TARGET_INDICES[local_target])
    truth = labels[:, global_target].astype(bool)
    base = main_probability[:, global_target]
    expert = expert_probability[:, local_target]
    final = np.clip(base + alpha * np.asarray(accept, dtype=np.float32) * (expert - base), 0.0, 1.0)
    base_prediction = base >= threshold
    final_prediction = final >= threshold
    return {
        "recall_gain": float(final_prediction[truth].mean() - base_prediction[truth].mean()),
        "fpr_delta": float(final_prediction[~truth].mean() - base_prediction[~truth].mean()),
        "aupr_delta": float(sk_metrics.average_precision_score(truth, final) - sk_metrics.average_precision_score(truth, base)),
        "accept_coverage": float(np.asarray(accept, dtype=bool).mean()),
    }


def select_benefit_cutoff(
    labels: np.ndarray,
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    benefit_score: np.ndarray,
    risk_score: np.ndarray,
    risk_threshold: float,
    local_target: int,
    threshold: float,
) -> dict[str, object]:
    rows = []
    for cutoff in np.linspace(0.25, 0.85, 13):
        accept = (benefit_score >= cutoff) & (risk_score < risk_threshold)
        metrics = route_metrics(labels, main_probability, expert_probability, accept, local_target, threshold)
        row = {"benefit_cutoff": float(cutoff), **metrics}
        row["eligible"] = bool(metrics["fpr_delta"] <= 0.005 + 1e-12 and metrics["aupr_delta"] >= -0.005)
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return {
            "benefit_cutoff": 1.1, "recall_gain": 0.0, "fpr_delta": 0.0,
            "aupr_delta": 0.0, "accept_coverage": 0.0, "eligible": False,
            "candidate_rows": rows,
        }
    selected = max(eligible, key=lambda row: (row["recall_gain"], row["aupr_delta"], -row["accept_coverage"]))
    return {**selected, "candidate_rows": rows}


def _physical_features(payload: dict[str, np.ndarray], local_target: int) -> np.ndarray:
    return payload[("mft", "trajectory", "isrj")[local_target]]


def feature_matrix(
    payload: dict[str, np.ndarray], local_target: int, threshold: float, variant: str,
) -> np.ndarray:
    probability = three_state_features(payload["main"], payload["expert"], local_target, threshold)
    physical = _physical_features(payload, local_target)
    if variant == "probability_only":
        return probability
    if variant == "physical_only":
        return physical.astype(np.float32)
    if variant == "probability_plus_physical":
        return np.concatenate((probability, physical), axis=1).astype(np.float32)
    raise KeyError(variant)


def _role_payload(records: list[dict[str, np.ndarray]], indices: np.ndarray) -> dict[str, np.ndarray]:
    output = {name: [] for name in ("sample_indices", "power", "labels", "main", "expert", "mft", "trajectory", "isrj")}
    selected = set(np.asarray(indices, dtype=np.int64).tolist())
    for record in records:
        mask = np.asarray([int(value) in selected for value in record["sample_indices"]], dtype=bool)
        for name in output:
            output[name].append(record[name][mask])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _targets(payload: dict[str, np.ndarray], local_target: int, threshold: float):
    risk, benefit, _, audit = build_risk_benefit_targets(
        payload["main"], payload["expert"], payload["labels"], local_target, threshold
    )
    return risk.astype(np.uint8), benefit.astype(np.uint8), audit


def _evaluate_variant(
    fit: dict[str, np.ndarray], calibration: dict[str, np.ndarray], probe: dict[str, np.ndarray],
    local_target: int, threshold: float, variant: str, seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fit_risk, fit_benefit, fit_audit = _targets(fit, local_target, threshold)
    cal_risk, cal_benefit, cal_audit = _targets(calibration, local_target, threshold)
    probe_risk, probe_benefit, probe_audit = _targets(probe, local_target, threshold)
    fit_features = feature_matrix(fit, local_target, threshold, variant)
    cal_features = feature_matrix(calibration, local_target, threshold, variant)
    probe_features = feature_matrix(probe, local_target, threshold, variant)
    risk_model = _fit_binary(fit_features, fit_risk, seed + local_target * 101)
    benefit_model = _fit_binary(fit_features, fit_benefit, seed + local_target * 101 + 1)
    cal_risk_score = risk_model.predict_proba(cal_features)[:, 1]
    cal_benefit_score = benefit_model.predict_proba(cal_features)[:, 1]
    probe_risk_score = risk_model.predict_proba(probe_features)[:, 1]
    probe_benefit_score = benefit_model.predict_proba(probe_features)[:, 1]
    veto = select_veto_threshold(cal_risk, cal_risk_score, 0.90)
    selection = select_benefit_cutoff(
        calibration["labels"], calibration["main"], calibration["expert"],
        cal_benefit_score, cal_risk_score, float(veto["threshold"]), local_target, threshold,
    )
    probe_accept = (
        (probe_benefit_score >= float(selection["benefit_cutoff"]))
        & (probe_risk_score < float(veto["threshold"]))
    )
    probe_route = route_metrics(
        probe["labels"], probe["main"], probe["expert"], probe_accept, local_target, threshold
    )
    veto_mask = probe_risk_score >= float(veto["threshold"])
    probe_veto = {
        "risk_recall": float(veto_mask[probe_risk.astype(bool)].mean()) if probe_risk.any() else float("nan"),
        "risk_precision": float(probe_risk[veto_mask].mean()) if veto_mask.any() else float("nan"),
        "veto_coverage": float(veto_mask.mean()),
        "retained_benefit": float((~veto_mask)[probe_benefit.astype(bool)].mean()) if probe_benefit.any() else float("nan"),
    }
    by_power = {}
    for power in TRANSITION_POWERS:
        mask = probe["power"] == power
        by_power[str(power)] = {
            "risk": binary_metrics(probe_risk[mask], probe_risk_score[mask]),
            "benefit": binary_metrics(probe_benefit[mask], probe_benefit_score[mask]),
            "route": route_metrics(
                probe["labels"][mask], probe["main"][mask], probe["expert"][mask],
                probe_accept[mask], local_target, threshold,
            ),
        }
    result = {
        "feature_variant": variant,
        "feature_dimension": int(fit_features.shape[1]),
        "fit_target_audit": fit_audit,
        "calibration_target_audit": cal_audit,
        "probe_target_audit": probe_audit,
        "probe_risk": binary_metrics(probe_risk, probe_risk_score),
        "probe_benefit": binary_metrics(probe_benefit, probe_benefit_score),
        "calibrated_veto": veto,
        "probe_veto": probe_veto,
        "calibration_router_selection": selection,
        "probe_route": probe_route,
        "probe_by_power": by_power,
    }
    arrays = {
        "risk_score": probe_risk_score.astype(np.float32),
        "benefit_score": probe_benefit_score.astype(np.float32),
        "accept": probe_accept.astype(np.uint8),
        "risk_target": probe_risk,
        "benefit_target": probe_benefit,
    }
    return result, arrays


def _oracle_metrics(payload: dict[str, np.ndarray], local_target: int, threshold: float) -> dict[str, float]:
    risk, benefit, _ = _targets(payload, local_target, threshold)
    accept = benefit.astype(bool) & ~risk.astype(bool)
    return route_metrics(payload["labels"], payload["main"], payload["expert"], accept, local_target, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--original-power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument(
        "--new-power-root",
        default="generated_data/p1d_independent_power_sweep_seed100000000_rechunked",
        help="Float32 sample-chunked cache of the P1d train data; values are unchanged.",
    )
    parser.add_argument("--output-root", default="outputs/p1e_transition_physical_sufficiency_20260731")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root = project / args.run_root
    separator_root = project / args.separator_root
    new_root = project / args.new_power_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    role_file = np.load(run_root / "role_indices.npz")
    original_roles = {name: role_file[name].astype(np.int64) for name in ("train", "early")}
    original_files = discover_power_train_files(args.original_power_root)
    _, original_labels = read_identity_metadata(protocol["data_file"])
    new_files = discover_split_files(new_root, "train")
    new_identities, new_labels = read_identity_metadata(new_files[30])
    role_indices = stratified_identity_split(new_labels)
    for first, second in (("fit", "calibration"), ("fit", "probe"), ("calibration", "probe")):
        if np.intersect1d(role_indices[first], role_indices[second]).size:
            raise RuntimeError(f"identity overlap: {first}/{second}")
    normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {
        int(row["seed"]): row for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"
    }
    frozen_protocol = {
        "experiment": "P1e transition-region physical sufficiency screen",
        "source_data": str(new_root),
        "source_split": "train_only",
        "transition_powers_db": list(TRANSITION_POWERS),
        "class_mapping": {"local_0": "MFT", "local_1": "VFT", "local_2": "ISRJ"},
        "identity_split": {name: {"count": int(len(values)), "index_hash": _hash_int64(values), "identity_hash": _hash_int64(new_identities[values])} for name, values in role_indices.items()},
        "feature_variants": list(FEATURE_VARIANTS),
        "frozen_model_rule": "same main, U-Net and experts as P1c/P1d; no retraining of neural networks",
        "router_model": "HistGradientBoosting max_iter=150 lr=0.05 leaf=7 l2=1 balanced",
        "oracle_margin_nats": 0.05,
        "veto_calibration_target_recall": 0.90,
        "route_alpha": 0.75,
        "route_safety": {"fpr_delta_max": 0.005, "aupr_delta_min": -0.005},
        "promotion_gate_per_seed": {
            "combined_risk_auroc_min": 0.70,
            "combined_benefit_auroc_min": 0.65,
            "risk_auroc_gain_over_probability_min": 0.03,
            "probe_risk_recall_min": 0.80,
            "retained_benefit_min": 0.50,
            "probe_route_recall_gain_over_probability_min": 0.02,
            "probe_route_fpr_delta_max": 0.005,
        },
        "target_promotion": "all gates pass in 3/3 seeds",
        "independent_test_used": False,
        "confirm_used": False,
        "locked_test_used": False,
    }
    (output_root / "protocol.json").write_text(json.dumps(frozen_protocol, indent=2), encoding="utf-8")

    results = []
    for seed in args.seeds:
        set_seed(seed)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        row = selected_rows[seed]
        raw_thresholds = row["thresholds"]
        thresholds = np.asarray(json.loads(raw_thresholds) if isinstance(raw_thresholds, str) else raw_thresholds, dtype=np.float32)
        main_model, main_checkpoint = _load_main(row, protocol, project, device)
        unet_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(unet_checkpoint, device)

        expert_train = extract_unet_role(
            unet, original_files[10], original_files[50], original_labels,
            original_roles["train"], args.batch_size, seed, device,
        )
        expert_early = extract_unet_role(
            unet, original_files[10], original_files[50], original_labels,
            original_roles["early"], args.batch_size, seed + 1, device,
        )
        experts = fit_frozen_experts(expert_train, expert_early, seed)

        records = []
        all_indices = np.arange(len(new_labels), dtype=np.int64)
        for power in TRANSITION_POWERS:
            main_record = collect_main(main_model, new_files[power], all_indices, args.batch_size, device, normalizer)
            unet_record = extract_unet_role(
                unet, new_files[power], new_files[50], new_labels,
                all_indices, args.batch_size, seed + power, device,
            )
            align_payloads(main_record, unet_record)
            records.append({
                "sample_indices": main_record["sample_indices"],
                "power": np.full(len(all_indices), power, dtype=np.int16),
                "labels": main_record["labels"],
                "main": main_record["probabilities"],
                "expert": expert_probabilities(unet_record, experts),
                "mft": unet_record["mft"],
                "trajectory": unet_record["trajectory"],
                "isrj": unet_record["isrj"],
            })
        roles = {name: _role_payload(records, indices) for name, indices in role_indices.items()}

        targets = {}
        saved_arrays = {
            "sample_index": roles["probe"]["sample_indices"],
            "receiver_noise_seed": new_identities[roles["probe"]["sample_indices"]],
            "power_db": roles["probe"]["power"],
            "labels": roles["probe"]["labels"].astype(np.uint8),
            "main_probability": roles["probe"]["main"],
            "expert_probability": roles["probe"]["expert"],
        }
        for local, (name, global_target) in enumerate(zip(TARGET_NAMES, TARGET_INDICES)):
            variants = {}
            variant_arrays = {}
            for variant_index, variant in enumerate(FEATURE_VARIANTS):
                variant_result, arrays = _evaluate_variant(
                    roles["fit"], roles["calibration"], roles["probe"],
                    local, float(thresholds[global_target]), variant,
                    seed + variant_index * 1000,
                )
                variants[variant] = variant_result
                variant_arrays[variant] = arrays
                for key, value in arrays.items():
                    saved_arrays[f"{name}_{variant}_{key}"] = value
            probability = variants["probability_only"]
            combined = variants["probability_plus_physical"]
            gates = {
                "combined_risk_auroc_min": bool(combined["probe_risk"]["auroc"] >= 0.70),
                "combined_benefit_auroc_min": bool(combined["probe_benefit"]["auroc"] >= 0.65),
                "risk_auroc_gain_over_probability_min": bool(
                    combined["probe_risk"]["auroc"] - probability["probe_risk"]["auroc"] >= 0.03
                ),
                "probe_risk_recall_min": bool(combined["probe_veto"]["risk_recall"] >= 0.80),
                "retained_benefit_min": bool(combined["probe_veto"]["retained_benefit"] >= 0.50),
                "probe_route_recall_gain_over_probability_min": bool(
                    combined["probe_route"]["recall_gain"] - probability["probe_route"]["recall_gain"] >= 0.02
                ),
                "probe_route_fpr_delta_max": bool(combined["probe_route"]["fpr_delta"] <= 0.005 + 1e-12),
            }
            targets[name] = {
                "oracle_probe_route": _oracle_metrics(roles["probe"], local, float(thresholds[global_target])),
                "variants": variants,
                "gates": gates,
                "seed_passed": bool(all(gates.values())),
            }
        np.savez_compressed(seed_dir / "probe_per_sample.npz", **saved_arrays)
        result = {
            "seed": seed,
            "frozen_main_checkpoint": str(main_checkpoint),
            "frozen_unet_checkpoint": str(unet_checkpoint),
            "targets": targets,
            "identity_audit": frozen_protocol["identity_split"],
            "independent_test_used": False,
            "confirm_used": False,
            "locked_test_used": False,
        }
        (seed_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)
        del main_model, unet, experts
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {}
    for target in TARGET_NAMES:
        seed_rows = [row["targets"][target] for row in results]
        aggregate[target] = {
            "seed_pass_count": int(sum(row["seed_passed"] for row in seed_rows)),
            "promoted": bool(all(row["seed_passed"] for row in seed_rows)),
            "mean_oracle_recall_gain": float(np.mean([row["oracle_probe_route"]["recall_gain"] for row in seed_rows])),
            "mean_probability_risk_auroc": float(np.mean([row["variants"]["probability_only"]["probe_risk"]["auroc"] for row in seed_rows])),
            "mean_combined_risk_auroc": float(np.mean([row["variants"]["probability_plus_physical"]["probe_risk"]["auroc"] for row in seed_rows])),
            "mean_risk_auroc_gain": float(np.mean([
                row["variants"]["probability_plus_physical"]["probe_risk"]["auroc"]
                - row["variants"]["probability_only"]["probe_risk"]["auroc"] for row in seed_rows
            ])),
            "mean_probability_route_recall_gain": float(np.mean([row["variants"]["probability_only"]["probe_route"]["recall_gain"] for row in seed_rows])),
            "mean_combined_route_recall_gain": float(np.mean([row["variants"]["probability_plus_physical"]["probe_route"]["recall_gain"] for row in seed_rows])),
            "maximum_combined_route_fpr_delta": float(max(row["variants"]["probability_plus_physical"]["probe_route"]["fpr_delta"] for row in seed_rows)),
        }
    summary_payload = {
        "experiment": "P1e transition-region physical sufficiency screen",
        "aggregate": aggregate,
        "any_target_promoted": bool(any(row["promoted"] for row in aggregate.values())),
        "decision": "allow target-specific physical risk router study" if any(row["promoted"] for row in aggregate.values()) else "stop increasing router capacity; physical features are insufficient",
        "elapsed_seconds": float(time.perf_counter() - started),
        "independent_test_used": False,
        "confirm_used": False,
        "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary_payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary_payload), indent=2))


if __name__ == "__main__":
    main()
