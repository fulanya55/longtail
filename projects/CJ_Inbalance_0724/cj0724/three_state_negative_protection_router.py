"""P1b matched-hard-negative three-state routing for frozen swept experts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
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
from cj0724.isrj_separator_physics_head import threshold_at_fpr
from cj0724.oracle_swept_isolation_experiment import set_seed
from cj0724.paper_core_evidence import load_normalizer


KEEP_MAIN, ACCEPT_EXPERT, ABSTAIN = 0, 1, 2
STATE_NAMES = ("keep_main", "accept_expert", "abstain")


def three_state_features(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    local_target: int,
    threshold: float,
) -> np.ndarray:
    global_target = int(TARGET_INDICES[local_target])
    base_target = main_probability[:, global_target]
    expert_target = expert_probability[:, local_target]
    other = np.delete(main_probability, global_target, axis=1)
    return np.column_stack((
        main_probability,
        expert_probability,
        base_target,
        expert_target,
        expert_target - base_target,
        np.abs(expert_target - base_target),
        base_target - float(threshold),
        expert_target - float(threshold),
        other.sum(axis=1),
        other.max(axis=1),
    )).astype(np.float32)


def _matching_features(
    main_probability: np.ndarray, expert_probability: np.ndarray, local_target: int
) -> np.ndarray:
    global_target = int(TARGET_INDICES[local_target])
    base = main_probability[:, global_target]
    expert = expert_probability[:, local_target]
    other_sum = np.delete(main_probability, global_target, axis=1).sum(axis=1)
    return np.column_stack((base, expert, np.abs(expert - base), other_sum)).astype(np.float32)


def build_three_state_targets(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    labels: np.ndarray,
    local_target: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    global_target = int(TARGET_INDICES[local_target])
    truth = labels[:, global_target].astype(np.float64)
    base = np.clip(main_probability[:, global_target], 1e-6, 1.0 - 1e-6)
    expert = np.clip(expert_probability[:, local_target], 1e-6, 1.0 - 1e-6)
    base_loss = -(truth * np.log(base) + (1.0 - truth) * np.log(1.0 - base))
    expert_loss = -(truth * np.log(expert) + (1.0 - truth) * np.log(1.0 - expert))
    advantage = expert_loss + 0.05 < base_loss
    negative = truth == 0
    positive = ~negative
    hard_negative = negative & ((base >= threshold) | (expert >= threshold)) & ~advantage
    ambiguous_positive = positive & ~advantage & (
        (base < threshold) | (expert < threshold) | (np.abs(expert - base) >= 0.15)
    )
    state = np.full(len(labels), KEEP_MAIN, dtype=np.uint8)
    state[advantage] = ACCEPT_EXPERT
    state[hard_negative | ambiguous_positive] = ABSTAIN

    matched = np.zeros(len(labels), dtype=bool)
    accept_positive = np.flatnonzero(advantage & positive)
    difficult_negative = np.flatnonzero(hard_negative)
    if len(accept_positive) and len(difficult_negative):
        features = _matching_features(main_probability, expert_probability, local_target)
        scaler = StandardScaler().fit(np.concatenate((features[accept_positive], features[difficult_negative])))
        negative_features = scaler.transform(features[difficult_negative])
        positive_features = scaler.transform(features[accept_positive])
        neighbors = NearestNeighbors(n_neighbors=1).fit(negative_features)
        nearest = neighbors.kneighbors(positive_features, return_distance=False).ravel()
        matched[difficult_negative[np.unique(nearest)]] = True
    weight = np.ones(len(labels), dtype=np.float64)
    weight[state == ABSTAIN] = 2.0
    weight[matched] = 4.0
    audit = {
        "keep_main": int((state == KEEP_MAIN).sum()),
        "accept_expert": int((state == ACCEPT_EXPERT).sum()),
        "abstain": int((state == ABSTAIN).sum()),
        "hard_negative": int(hard_negative.sum()),
        "ambiguous_positive": int(ambiguous_positive.sum()),
        "matched_difficult_negative": int(matched.sum()),
    }
    return state, weight, audit


class AlignedThreeStateRouter:
    def __init__(self, model, classes: np.ndarray):
        self.model = model
        self.classes = np.asarray(classes, dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        partial = self.model.predict_proba(features)
        output = np.zeros((len(features), 3), dtype=np.float64)
        output[:, self.classes] = partial
        return output


class ConstantThreeStateRouter:
    def __init__(self, state: int):
        self.state = int(state)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        output = np.zeros((len(features), 3), dtype=np.float64)
        output[:, self.state] = 1.0
        return output


def fit_three_state_routers(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    seed: int,
):
    routers, audits = [], []
    for local, global_target in enumerate(TARGET_INDICES):
        state, weight, audit = build_three_state_targets(
            main_probability, expert_probability, labels, local, thresholds[global_target]
        )
        classes = np.unique(state)
        if len(classes) == 1:
            router = ConstantThreeStateRouter(int(classes[0]))
        else:
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2500, random_state=seed),
            )
            pipeline.fit(
                three_state_features(main_probability, expert_probability, local, thresholds[global_target]),
                state,
                logisticregression__sample_weight=weight,
            )
            router = AlignedThreeStateRouter(pipeline, pipeline.named_steps["logisticregression"].classes_)
        routers.append(router)
        audits.append(audit)
    return routers, audits


def predict_three_state(
    routers,
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    return np.stack([
        router.predict_proba(
            three_state_features(main_probability, expert_probability, local, thresholds[global_target])
        )
        for local, (router, global_target) in enumerate(zip(routers, TARGET_INDICES))
    ], axis=1).astype(np.float32)


def three_state_blend(
    main_probability: np.ndarray,
    expert_probability: np.ndarray,
    state_probability: np.ndarray,
    alpha: float,
    accept_threshold: float,
    veto_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    final = np.asarray(main_probability, dtype=np.float32).copy()
    if alpha <= 0.0:
        return final, np.zeros((len(final), 3), dtype=bool)
    accept = (
        (state_probability[:, :, ACCEPT_EXPERT] >= float(accept_threshold))
        & ((state_probability[:, :, ACCEPT_EXPERT] - state_probability[:, :, ABSTAIN]) >= float(veto_margin))
    )
    base = final[:, TARGET_INDICES]
    final[:, TARGET_INDICES] = np.clip(base + float(alpha) * accept * (expert_probability - base), 0.0, 1.0)
    return final, accept


def _power_metrics(records, thresholds, alpha, accept_threshold, veto_margin):
    rows = []
    for record in records:
        final, accept = three_state_blend(
            record["probabilities"], record["expert"], record["state_probability"],
            alpha, accept_threshold, veto_margin,
        )
        rows.append({
            "power": int(record["power"][0]),
            "base": classification_metrics(record["labels"], record["probabilities"], thresholds),
            "final": classification_metrics(record["labels"], final, thresholds),
            "accept_coverage": accept.mean(axis=0),
            "abstain_argmax_coverage": (record["state_probability"].argmax(axis=2) == ABSTAIN).mean(axis=0),
        })
    return rows


def _score(rows):
    extreme_gain = float(np.mean([
        row["final"]["per_class"][index]["recall"] - row["base"]["per_class"][index]["recall"]
        for row in rows if row["power"] in (10, 50) for index in TARGET_INDICES
    ]))
    macro_delta = float(np.mean([row["final"]["macro_aupr"] - row["base"]["macro_aupr"] for row in rows]))
    fpr_delta = [float(np.mean([
        row["final"]["per_class"][index]["fpr"] - row["base"]["per_class"][index]["fpr"] for row in rows
    ])) for index in TARGET_INDICES]
    return {
        "extreme_recall_gain": extreme_gain,
        "macro_aupr_delta": macro_delta,
        "target_fpr_delta": fpr_delta,
        "accept_coverage": [float(np.mean([row["accept_coverage"][local] for row in rows])) for local in range(3)],
        "abstain_coverage": [float(np.mean([row["abstain_argmax_coverage"][local] for row in rows])) for local in range(3)],
    }


def select_candidate(records, thresholds):
    candidates = []
    for alpha in (0.25, 0.5, 0.75):
        for accept_threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
            for veto_margin in (0.0, 0.1, 0.2, 0.3):
                score = _score(_power_metrics(records, thresholds, alpha, accept_threshold, veto_margin))
                score.update({
                    "alpha": alpha, "accept_threshold": accept_threshold, "veto_margin": veto_margin,
                    "eligible": bool(score["macro_aupr_delta"] >= -0.005 and max(score["target_fpr_delta"]) <= 0.005),
                })
                candidates.append(score)
    eligible = [row for row in candidates if row["eligible"]]
    if eligible:
        selected = max(eligible, key=lambda row: (row["extreme_recall_gain"], row["macro_aupr_delta"], -max(row["target_fpr_delta"])))
    else:
        selected = {
            "alpha": 0.0, "accept_threshold": 1.0, "veto_margin": 1.0,
            "extreme_recall_gain": 0.0, "macro_aupr_delta": 0.0,
            "target_fpr_delta": [0.0, 0.0, 0.0], "accept_coverage": [0.0] * 3,
            "abstain_coverage": [0.0] * 3, "eligible": False,
            "fallback_reason": "no candidate met fixed-threshold non-degradation constraints",
        }
    return selected, candidates


def calibrate_target_thresholds(records, selected, base: bool) -> np.ndarray:
    thresholds = np.asarray(selected["main_thresholds"], dtype=np.float32).copy()
    labels, probabilities = [], []
    for record in records:
        if base:
            probability = record["probabilities"]
        else:
            probability, _ = three_state_blend(
                record["probabilities"], record["expert"], record["state_probability"],
                selected["alpha"], selected["accept_threshold"], selected["veto_margin"],
            )
        labels.append(record["labels"])
        probabilities.append(probability)
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    for index in TARGET_INDICES:
        thresholds[index] = threshold_at_fpr(labels[:, index], probabilities[:, index], 0.10)
    return thresholds


def evaluate_tune(records, main_thresholds, selected, base_recalibrated, final_recalibrated, scale):
    by_power = []
    for record in records:
        final, accept = three_state_blend(
            record["probabilities"], record["expert"], record["state_probability"],
            selected["alpha"], selected["accept_threshold"], selected["veto_margin"],
        )
        level = int(record["power"][0])
        weak = _weak_mask(record["labels"].astype(bool), level)
        common = {
            "sample_indices": record["sample_indices"], "labels": record["labels"],
            "prediction": record["prediction"], "target": record["target"], "mask": record["mask"],
            "scale": scale,
        }
        by_power.append({
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
            "abstain_coverage": dict(zip(TARGET_NAMES, (record["state_probability"].argmax(axis=2) == ABSTAIN).mean(axis=0))),
        })

    def aggregate(prefix_base: str, prefix_final: str):
        gain = float(np.mean([
            row[prefix_final]["per_class"][index]["recall"] - row[prefix_base]["per_class"][index]["recall"]
            for row in by_power if row["power_level_db"] in (10, 50) for index in TARGET_INDICES
        ]))
        macro_delta = float(np.mean([row[prefix_final]["macro_aupr"] - row[prefix_base]["macro_aupr"] for row in by_power]))
        fpr = {TARGET_NAMES[local]: float(np.mean([row[prefix_final]["per_class"][index]["fpr"] for row in by_power])) for local, index in enumerate(TARGET_INDICES)}
        fpr_delta = {TARGET_NAMES[local]: float(np.mean([
            row[prefix_final]["per_class"][index]["fpr"] - row[prefix_base]["per_class"][index]["fpr"] for row in by_power
        ])) for local, index in enumerate(TARGET_INDICES)}
        gate_base = prefix_base + "_parameter_gate"
        gate_final = prefix_final + "_parameter_gate"
        return {
            "extreme_three_class_recall_gain": gain,
            "mean_nine_power_macro_aupr_delta": macro_delta,
            "mean_target_fpr": fpr,
            "mean_target_fpr_delta": fpr_delta,
            "mean_parameter_coverage_delta": float(np.mean([
                row[gate_final]["predicted_gate_parameter_coverage"] - row[gate_base]["predicted_gate_parameter_coverage"] for row in by_power
            ])),
            "mean_miss_penalty_parameter_nmae_delta": float(np.mean([
                row[gate_final]["miss_penalty_parameter_nmae"] - row[gate_base]["miss_penalty_parameter_nmae"] for row in by_power
            ])),
        }
    return by_power, {
        "fixed_threshold": aggregate("base_fixed", "final_fixed"),
        "recalibrated": aggregate("base_recalibrated", "final_recalibrated"),
    }


def _merge(records):
    names = ("sample_indices", "labels", "probabilities", "prediction", "target", "mask", "expert", "state_probability", "power")
    return {name: np.concatenate([record[name] for record in records]) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/p1b_three_state_negative_protection_20260726")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    run_root, separator_root = project / args.run_root, project / args.separator_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    role_file = np.load(run_root / "role_indices.npz")
    roles = {name: role_file[name].astype(np.int64) for name in ("train", "early", "calibration", "tune")}
    files = discover_power_train_files(args.power_root)
    alignment = audit_power_alignment(files)
    root_identities, root_labels = read_identity_metadata(protocol["data_file"])
    power_identities, power_labels = read_identity_metadata(files[30])
    if not np.array_equal(root_identities, power_identities) or not np.array_equal(root_labels, power_labels):
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
        train_unet = extract_unet_role(unet, files[10], files[50], root_labels, roles["train"], args.batch_size, seed, device)
        early_unet = extract_unet_role(unet, files[10], files[50], root_labels, roles["early"], args.batch_size, seed, device)
        experts = fit_frozen_experts(train_unet, early_unet, seed)
        fit_indices, safety_indices = _identity_split(roles["calibration"], root_identities, seed + 7000)
        if np.intersect1d(root_identities[fit_indices], root_identities[safety_indices]).size:
            raise RuntimeError("fit/safety identity leakage")
        records = {"fit": [], "safety": [], "tune": []}
        for role_name, indices in (("fit", fit_indices), ("safety", safety_indices), ("tune", roles["tune"])):
            for level in POWER_LEVELS:
                main_record = collect_main(main_model, files[level], indices, args.batch_size, device, normalizer)
                unet_record = extract_unet_role(unet, files[level], files[50], root_labels, indices, args.batch_size, seed + level, device)
                align_payloads(main_record, unet_record)
                records[role_name].append({
                    **{name: main_record[name] for name in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")},
                    "expert": expert_probabilities(unet_record, experts),
                    "power": np.full(len(indices), level, dtype=np.int16),
                })
        fit = {name: np.concatenate([record[name] for record in records["fit"]]) for name in ("labels", "probabilities", "expert")}
        routers, state_audits = fit_three_state_routers(fit["probabilities"], fit["expert"], fit["labels"], thresholds, seed)
        for role_name in records:
            for record in records[role_name]:
                record["state_probability"] = predict_three_state(routers, record["probabilities"], record["expert"], thresholds)
        selected, candidates = select_candidate(records["safety"], thresholds)
        selected["main_thresholds"] = thresholds.tolist()
        base_recalibrated = calibrate_target_thresholds(records["safety"], selected, base=True)
        final_recalibrated = calibrate_target_thresholds(records["safety"], selected, base=False)
        by_power, aggregate = evaluate_tune(
            records["tune"], thresholds, selected, base_recalibrated, final_recalibrated, normalizer.scale.numpy()
        )
        tune = _merge(records["tune"])
        final_parts, accept_parts = [], []
        for record in records["tune"]:
            final, accept = three_state_blend(
                record["probabilities"], record["expert"], record["state_probability"],
                selected["alpha"], selected["accept_threshold"], selected["veto_margin"],
            )
            final_parts.append(final)
            accept_parts.append(accept)
        np.savez_compressed(
            seed_dir / "tune_per_sample.npz",
            sample_index=tune["sample_indices"], receiver_noise_seed=root_identities[tune["sample_indices"]],
            power_db=tune["power"], labels=tune["labels"].astype(np.uint8), main_probability=tune["probabilities"],
            expert_probability=tune["expert"], state_probability=tune["state_probability"],
            accept_mask=np.concatenate(accept_parts).astype(np.uint8), final_probability=np.concatenate(final_parts),
            main_thresholds=thresholds, base_recalibrated_thresholds=base_recalibrated,
            final_recalibrated_thresholds=final_recalibrated, parameter_prediction=tune["prediction"],
            parameter_target=tune["target"], parameter_mask=tune["mask"].astype(np.uint8),
        )
        result = {
            "seed": seed,
            "frozen_main_checkpoint": str(main_checkpoint), "frozen_unet_checkpoint": str(unet_checkpoint),
            "state_training_audit": dict(zip(TARGET_NAMES, state_audits)),
            "selected": selected, "safety_candidates": candidates,
            "thresholds": {"main": thresholds, "base_recalibrated": base_recalibrated, "final_recalibrated": final_recalibrated},
            "tune_by_power": by_power, "tune_aggregate": aggregate,
            "identity_audit": {
                "fit_hash": _hash(fit_indices), "safety_hash": _hash(safety_indices),
                "fit_safety_overlap": 0, "tune_fit_overlap": int(np.intersect1d(roles["tune"], fit_indices).size),
                "tune_safety_overlap": int(np.intersect1d(roles["tune"], safety_indices).size),
            },
            "confirm_used": False, "locked_test_used": False,
        }
        (seed_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)
        del main_model, unet, experts, routers
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
        "experiment": "P1b matched-hard-negative three-state router",
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
        "identity_alignment_audit": alignment,
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
