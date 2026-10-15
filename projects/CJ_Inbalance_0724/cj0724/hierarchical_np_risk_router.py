"""P1c hierarchical false-positive-risk and expert-benefit selective router."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cj0724.dual_axis import POWER_LEVELS, audit_power_alignment, discover_power_train_files, read_identity_metadata
from cj0724.identity_safe_power_parameter_eval import _level_metrics, _weak_mask
from cj0724.integrate_swept_experts_with_six_class import (
    TARGET_INDICES,
    TARGET_NAMES,
    _hash,
    _identity_split,
    _jsonable,
    _load_main,
    _load_unet,
    align_payloads,
    classification_metrics,
    collect_main,
    expert_probabilities,
    extract_unet_role,
    fit_frozen_experts,
)
from cj0724.isrj_separator_physics_head import clipped_logit, threshold_at_fpr
from cj0724.oracle_swept_isolation_experiment import set_seed
from cj0724.paper_core_evidence import load_normalizer
from cj0724.three_state_negative_protection_router import _matching_features, three_state_features


class ConstantBinaryModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        value = np.full(len(features), self.probability, dtype=np.float64)
        return np.column_stack((1.0 - value, value))


def _fit_binary(features, target, seed, sample_weight=None):
    target = np.asarray(target, dtype=np.uint8)
    if np.unique(target).size < 2:
        return ConstantBinaryModel(float(target[0]))
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        l2_regularization=1.0, class_weight="balanced", random_state=seed,
    )
    model.fit(features, target, sample_weight=sample_weight)
    return model


def build_risk_benefit_targets(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    labels: np.ndarray,
    local_target: int,
    threshold: float,
):
    global_target = int(TARGET_INDICES[local_target])
    truth = labels[:, global_target].astype(np.float64)
    base = np.clip(main_probability[:, global_target], 1e-6, 1.0 - 1e-6)
    expert = np.clip(expert_probability[:, local_target], 1e-6, 1.0 - 1e-6)
    base_loss = -(truth * np.log(base) + (1.0 - truth) * np.log(1.0 - base))
    expert_loss = -(truth * np.log(expert) + (1.0 - truth) * np.log(1.0 - expert))
    benefit = (expert_loss + 0.05 < base_loss)
    false_positive_risk = (truth == 0) & (expert > base) & (expert >= float(threshold))

    matched = np.zeros(len(labels), dtype=bool)
    source = np.flatnonzero((truth == 1) & benefit)
    candidates = np.flatnonzero(false_positive_risk)
    if len(source) and len(candidates):
        matching = _matching_features(main_probability, expert_probability, local_target)
        scaler = StandardScaler().fit(np.concatenate((matching[source], matching[candidates])))
        neighbor = NearestNeighbors(n_neighbors=1).fit(scaler.transform(matching[candidates]))
        nearest = neighbor.kneighbors(scaler.transform(matching[source]), return_distance=False).ravel()
        matched[candidates[np.unique(nearest)]] = True
    risk_weight = np.ones(len(labels), dtype=np.float64)
    risk_weight[false_positive_risk] = 2.0
    risk_weight[matched] = 4.0
    audit = {
        "benefit": int(benefit.sum()),
        "false_positive_risk": int(false_positive_risk.sum()),
        "matched_false_positive_risk": int(matched.sum()),
        "positive_benefit": int(((truth == 1) & benefit).sum()),
        "negative_benefit": int(((truth == 0) & benefit).sum()),
    }
    return false_positive_risk.astype(np.uint8), benefit.astype(np.uint8), risk_weight, audit


def fit_hierarchical_models(main_probability, expert_probability, labels, thresholds, seed):
    models, audits = [], []
    for local, global_target in enumerate(TARGET_INDICES):
        features = three_state_features(
            main_probability, expert_probability, local, thresholds[global_target]
        )
        risk, benefit, risk_weight, audit = build_risk_benefit_targets(
            main_probability, expert_probability, labels, local, thresholds[global_target]
        )
        models.append({
            "risk": _fit_binary(features, risk, seed + local * 17, risk_weight),
            "benefit": _fit_binary(features, benefit, seed + local * 17 + 1),
        })
        audits.append(audit)
    return models, audits


def predict_hierarchical_scores(models, main_probability, expert_probability, thresholds):
    risk, benefit, score = [], [], []
    for local, (model, global_target) in enumerate(zip(models, TARGET_INDICES)):
        features = three_state_features(main_probability, expert_probability, local, thresholds[global_target])
        risk_probability = model["risk"].predict_proba(features)[:, 1]
        benefit_probability = model["benefit"].predict_proba(features)[:, 1]
        risk.append(risk_probability)
        benefit.append(benefit_probability)
        score.append(clipped_logit(benefit_probability) - clipped_logit(risk_probability))
    return (
        np.column_stack(risk).astype(np.float32),
        np.column_stack(benefit).astype(np.float32),
        np.column_stack(score).astype(np.float32),
    )


def selective_blend(main_probability, expert_probability, route_score, selections):
    final = np.asarray(main_probability, dtype=np.float32).copy()
    accept = np.zeros((len(final), 3), dtype=bool)
    for local, (selection, global_target) in enumerate(zip(selections, TARGET_INDICES)):
        alpha = float(selection["alpha"])
        if alpha <= 0.0:
            continue
        accept[:, local] = route_score[:, local] >= float(selection["score_cutoff"])
        base = final[:, global_target]
        final[:, global_target] = np.clip(
            base + alpha * accept[:, local] * (expert_probability[:, local] - base), 0.0, 1.0
        )
    return final, accept


def _target_candidate_metrics(records, thresholds, local, alpha, cutoff):
    global_target = int(TARGET_INDICES[local])
    recall_delta, fpr_delta, aupr_delta, coverage = [], [], [], []
    for record in records:
        base = record["probabilities"][:, global_target]
        accept = record["route_score"][:, local] >= cutoff
        final = np.clip(base + alpha * accept * (record["expert"][:, local] - base), 0.0, 1.0)
        truth = record["labels"][:, global_target].astype(bool)
        negative = ~truth
        base_pred = base >= thresholds[global_target]
        final_pred = final >= thresholds[global_target]
        if int(record["power"][0]) in (10, 50):
            recall_delta.append(float(final_pred[truth].mean() - base_pred[truth].mean()))
        fpr_delta.append(float(final_pred[negative].mean() - base_pred[negative].mean()))
        aupr_delta.append(float(
            sk_metrics.average_precision_score(truth, final)
            - sk_metrics.average_precision_score(truth, base)
        ))
        coverage.append(float(accept.mean()))
    return {
        "extreme_recall_gain": float(np.mean(recall_delta)),
        "mean_fpr_delta": float(np.mean(fpr_delta)),
        "mean_aupr_delta": float(np.mean(aupr_delta)),
        "mean_accept_coverage": float(np.mean(coverage)),
    }


def select_per_target(records, thresholds):
    stacked_score = np.concatenate([record["route_score"] for record in records])
    all_rows, selected = {}, []
    for local, name in enumerate(TARGET_NAMES):
        rows = []
        for alpha in (0.25, 0.5, 0.75):
            for coverage in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20):
                cutoff = float(np.quantile(stacked_score[:, local], 1.0 - coverage))
                row = _target_candidate_metrics(records, thresholds, local, alpha, cutoff)
                row.update({"alpha": alpha, "coverage_budget": coverage, "score_cutoff": cutoff})
                row["eligible"] = bool(row["mean_fpr_delta"] <= 0.005 and row["mean_aupr_delta"] >= -0.005)
                rows.append(row)
        eligible = [row for row in rows if row["eligible"]]
        if eligible:
            choice = max(eligible, key=lambda row: (row["extreme_recall_gain"], row["mean_aupr_delta"], -row["mean_fpr_delta"]))
        else:
            choice = {
                "alpha": 0.0, "coverage_budget": 0.0, "score_cutoff": float("inf"),
                "extreme_recall_gain": 0.0, "mean_fpr_delta": 0.0,
                "mean_aupr_delta": 0.0, "mean_accept_coverage": 0.0,
                "eligible": False, "fallback_reason": "no safe target candidate",
            }
        selected.append(choice)
        all_rows[name] = rows
    combined_rows = []
    for record in records:
        final, accept = selective_blend(record["probabilities"], record["expert"], record["route_score"], selected)
        combined_rows.append({
            "power": int(record["power"][0]),
            "base": classification_metrics(record["labels"], record["probabilities"], thresholds),
            "final": classification_metrics(record["labels"], final, thresholds),
            "accept": accept.mean(axis=0),
        })
    combined_macro_delta = float(np.mean([
        row["final"]["macro_aupr"] - row["base"]["macro_aupr"] for row in combined_rows
    ]))
    if combined_macro_delta < -0.005:
        selected = [{
            "alpha": 0.0, "coverage_budget": 0.0, "score_cutoff": float("inf"),
            "extreme_recall_gain": 0.0, "mean_fpr_delta": 0.0,
            "mean_aupr_delta": 0.0, "mean_accept_coverage": 0.0,
            "eligible": False, "fallback_reason": "combined macro-AUPR safety failure",
        } for _ in TARGET_NAMES]
    return selected, all_rows, combined_macro_delta


def calibrate_thresholds(records, selections, main_thresholds, base):
    thresholds = np.asarray(main_thresholds, dtype=np.float32).copy()
    labels, probabilities = [], []
    for record in records:
        if base:
            final = record["probabilities"]
        else:
            final, _ = selective_blend(record["probabilities"], record["expert"], record["route_score"], selections)
        labels.append(record["labels"])
        probabilities.append(final)
    labels, probabilities = np.concatenate(labels), np.concatenate(probabilities)
    for index in TARGET_INDICES:
        thresholds[index] = threshold_at_fpr(labels[:, index], probabilities[:, index], 0.10)
    return thresholds


def evaluate(records, selections, main_thresholds, base_recalibrated, final_recalibrated, scale):
    rows = []
    for record in records:
        final, accept = selective_blend(
            record["probabilities"], record["expert"], record["route_score"], selections
        )
        level = int(record["power"][0])
        weak = _weak_mask(record["labels"].astype(bool), level)
        common = {
            "sample_indices": record["sample_indices"], "labels": record["labels"],
            "prediction": record["prediction"], "target": record["target"],
            "mask": record["mask"], "scale": scale,
        }
        rows.append({
            "power_level_db": level,
            "base_fixed": classification_metrics(record["labels"], record["probabilities"], main_thresholds),
            "final_fixed": classification_metrics(record["labels"], final, main_thresholds),
            "base_recalibrated": classification_metrics(record["labels"], record["probabilities"], base_recalibrated),
            "final_recalibrated": classification_metrics(record["labels"], final, final_recalibrated),
            "base_fixed_parameter_gate": _level_metrics({**common, "probabilities": record["probabilities"], "thresholds": main_thresholds}, weak),
            "final_fixed_parameter_gate": _level_metrics({**common, "probabilities": final, "thresholds": main_thresholds}, weak),
            "base_recalibrated_parameter_gate": _level_metrics({**common, "probabilities": record["probabilities"], "thresholds": base_recalibrated}, weak),
            "final_recalibrated_parameter_gate": _level_metrics({**common, "probabilities": final, "thresholds": final_recalibrated}, weak),
            "accept_coverage": dict(zip(TARGET_NAMES, accept.mean(axis=0))),
        })

    def aggregate(base_name, final_name):
        gain = float(np.mean([
            row[final_name]["per_class"][index]["recall"] - row[base_name]["per_class"][index]["recall"]
            for row in rows if row["power_level_db"] in (10, 50) for index in TARGET_INDICES
        ]))
        macro = float(np.mean([row[final_name]["macro_aupr"] - row[base_name]["macro_aupr"] for row in rows]))
        fpr = {TARGET_NAMES[local]: float(np.mean([row[final_name]["per_class"][index]["fpr"] for row in rows])) for local, index in enumerate(TARGET_INDICES)}
        fpr_delta = {TARGET_NAMES[local]: float(np.mean([
            row[final_name]["per_class"][index]["fpr"] - row[base_name]["per_class"][index]["fpr"] for row in rows
        ])) for local, index in enumerate(TARGET_INDICES)}
        gate_base, gate_final = base_name + "_parameter_gate", final_name + "_parameter_gate"
        return {
            "extreme_three_class_recall_gain": gain,
            "mean_nine_power_macro_aupr_delta": macro,
            "mean_target_fpr": fpr,
            "mean_target_fpr_delta": fpr_delta,
            "mean_parameter_coverage_delta": float(np.mean([
                row[gate_final]["predicted_gate_parameter_coverage"] - row[gate_base]["predicted_gate_parameter_coverage"] for row in rows
            ])),
            "mean_miss_penalty_parameter_nmae_delta": float(np.mean([
                row[gate_final]["miss_penalty_parameter_nmae"] - row[gate_base]["miss_penalty_parameter_nmae"] for row in rows
            ])),
        }
    return rows, {
        "fixed_threshold": aggregate("base_fixed", "final_fixed"),
        "recalibrated": aggregate("base_recalibrated", "final_recalibrated"),
    }


def _merge(records):
    names = ("sample_indices", "labels", "probabilities", "prediction", "target", "mask", "expert", "risk_probability", "benefit_probability", "route_score", "power")
    return {name: np.concatenate([record[name] for record in records]) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/p1c_hierarchical_np_router_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root, separator_root, output_root = project / args.run_root, project / args.separator_root, project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    role_file = np.load(run_root / "role_indices.npz")
    roles = {name: role_file[name].astype(np.int64) for name in ("train", "early", "calibration", "tune")}
    files = discover_power_train_files(args.power_root)
    alignment = audit_power_alignment(files)
    identities, labels = read_identity_metadata(protocol["data_file"])
    power_identities, power_labels = read_identity_metadata(files[30])
    if not np.array_equal(identities, power_identities) or not np.array_equal(labels, power_labels):
        raise RuntimeError("root and power identities differ")
    normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {int(row["seed"]): row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}
    results = []

    for seed in args.seeds:
        set_seed(seed)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        main_model, main_checkpoint = _load_main(selected_rows[seed], protocol, project, device)
        unet_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(unet_checkpoint, device)
        raw_thresholds = selected_rows[seed]["thresholds"]
        thresholds = np.asarray(json.loads(raw_thresholds) if isinstance(raw_thresholds, str) else raw_thresholds, dtype=np.float32)
        train_unet = extract_unet_role(unet, files[10], files[50], labels, roles["train"], args.batch_size, seed, device)
        early_unet = extract_unet_role(unet, files[10], files[50], labels, roles["early"], args.batch_size, seed, device)
        experts = fit_frozen_experts(train_unet, early_unet, seed)
        fit_indices, safety_indices = _identity_split(roles["calibration"], identities, seed + 7000)
        if np.intersect1d(identities[fit_indices], identities[safety_indices]).size:
            raise RuntimeError("fit/safety identity overlap")
        records = {"fit": [], "safety": [], "tune": []}
        for role_name, indices in (("fit", fit_indices), ("safety", safety_indices), ("tune", roles["tune"])):
            for level in POWER_LEVELS:
                main_record = collect_main(main_model, files[level], indices, args.batch_size, device, normalizer)
                unet_record = extract_unet_role(unet, files[level], files[50], labels, indices, args.batch_size, seed + level, device)
                align_payloads(main_record, unet_record)
                records[role_name].append({
                    **{name: main_record[name] for name in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")},
                    "expert": expert_probabilities(unet_record, experts),
                    "power": np.full(len(indices), level, dtype=np.int16),
                })
        fit = {name: np.concatenate([record[name] for record in records["fit"]]) for name in ("labels", "probabilities", "expert")}
        models, training_audit = fit_hierarchical_models(fit["probabilities"], fit["expert"], fit["labels"], thresholds, seed)
        for role_name in records:
            for record in records[role_name]:
                risk, benefit, score = predict_hierarchical_scores(models, record["probabilities"], record["expert"], thresholds)
                record["risk_probability"], record["benefit_probability"], record["route_score"] = risk, benefit, score
        selections, candidates, safety_macro_delta = select_per_target(records["safety"], thresholds)
        base_recalibrated = calibrate_thresholds(records["safety"], selections, thresholds, True)
        final_recalibrated = calibrate_thresholds(records["safety"], selections, thresholds, False)
        by_power, aggregate = evaluate(records["tune"], selections, thresholds, base_recalibrated, final_recalibrated, normalizer.scale.numpy())
        tune = _merge(records["tune"])
        final_parts, accept_parts = [], []
        for record in records["tune"]:
            final, accept = selective_blend(record["probabilities"], record["expert"], record["route_score"], selections)
            final_parts.append(final)
            accept_parts.append(accept)
        np.savez_compressed(
            seed_dir / "tune_per_sample.npz",
            sample_index=tune["sample_indices"], receiver_noise_seed=identities[tune["sample_indices"]],
            power_db=tune["power"], labels=tune["labels"].astype(np.uint8),
            main_probability=tune["probabilities"], expert_probability=tune["expert"],
            risk_probability=tune["risk_probability"], benefit_probability=tune["benefit_probability"],
            route_score=tune["route_score"], accept_mask=np.concatenate(accept_parts).astype(np.uint8),
            final_probability=np.concatenate(final_parts), main_thresholds=thresholds,
            base_recalibrated_thresholds=base_recalibrated, final_recalibrated_thresholds=final_recalibrated,
            parameter_prediction=tune["prediction"], parameter_target=tune["target"], parameter_mask=tune["mask"].astype(np.uint8),
        )
        result = {
            "seed": seed, "frozen_main_checkpoint": str(main_checkpoint), "frozen_unet_checkpoint": str(unet_checkpoint),
            "training_audit": dict(zip(TARGET_NAMES, training_audit)),
            "selected_per_target": dict(zip(TARGET_NAMES, selections)), "safety_candidates": candidates,
            "safety_combined_macro_aupr_delta": safety_macro_delta,
            "thresholds": {"main": thresholds, "base_recalibrated": base_recalibrated, "final_recalibrated": final_recalibrated},
            "tune_by_power": by_power, "tune_aggregate": aggregate,
            "identity_audit": {
                "fit_hash": _hash(fit_indices), "safety_hash": _hash(safety_indices), "fit_safety_overlap": 0,
                "tune_fit_overlap": int(np.intersect1d(roles["tune"], fit_indices).size),
                "tune_safety_overlap": int(np.intersect1d(roles["tune"], safety_indices).size),
            },
            "confirm_used": False, "locked_test_used": False,
        }
        (seed_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)
        del main_model, unet, experts, models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fixed_gain = [row["tune_aggregate"]["fixed_threshold"]["extreme_three_class_recall_gain"] for row in results]
    fixed_macro = [row["tune_aggregate"]["fixed_threshold"]["mean_nine_power_macro_aupr_delta"] for row in results]
    fixed_fpr_delta = [max(row["tune_aggregate"]["fixed_threshold"]["mean_target_fpr_delta"].values()) for row in results]
    recal_gain = [row["tune_aggregate"]["recalibrated"]["extreme_three_class_recall_gain"] for row in results]
    recal_macro = [row["tune_aggregate"]["recalibrated"]["mean_nine_power_macro_aupr_delta"] for row in results]
    recal_fpr = [max(row["tune_aggregate"]["recalibrated"]["mean_target_fpr"].values()) for row in results]
    mechanism_pass = bool(np.mean(fixed_gain) >= 0.10 and min(fixed_gain) > 0 and np.mean(fixed_macro) >= -0.005 and max(fixed_fpr_delta) <= 0.005)
    deployment_pass = bool(np.mean(recal_gain) >= 0.10 and min(recal_gain) > 0 and np.mean(recal_macro) >= -0.005 and max(recal_fpr) <= 0.10 + 1e-12)
    payload = {
        "experiment": "P1c hierarchical NP risk-benefit router",
        "fixed_threshold": {
            "mean_extreme_recall_gain": float(np.mean(fixed_gain)), "per_seed_gain": fixed_gain,
            "same_direction_seeds": int(sum(value > 0 for value in fixed_gain)),
            "mean_macro_aupr_delta": float(np.mean(fixed_macro)), "maximum_seed_target_fpr_delta": float(max(fixed_fpr_delta)),
            "mechanism_passed": mechanism_pass,
        },
        "recalibrated": {
            "mean_extreme_recall_gain": float(np.mean(recal_gain)), "per_seed_gain": recal_gain,
            "same_direction_seeds": int(sum(value > 0 for value in recal_gain)),
            "mean_macro_aupr_delta": float(np.mean(recal_macro)), "maximum_seed_target_mean_fpr": float(max(recal_fpr)),
            "deployment_passed": deployment_pass,
        },
        "overall_promotion_passed": bool(mechanism_pass and deployment_pass),
        "next_gate": "enter P2 parameter-gate evaluation" if mechanism_pass and deployment_pass else "do not enter P2",
        "wall_seconds": float(time.perf_counter() - started),
        "identity_alignment_audit": alignment, "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
