"""Fit composition-safe thresholds and complexity-aware parameter gates.

Only fresh development identities are used.  Identity roles are disjoint:
threshold calibration fits empirical-FPR thresholds, rule fit chooses the
largest safe expert-complexity gate, and validation is read once for the
development promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from cj0724.p1k_joint_level3_system import MODEL_SEEDS, SAW_PARAMETER_INDEX, VFT_PARAMETER_INDEX
from cj0724.p2e_stage_b_weak_generalization import CONDITIONS, sha256_file
from cj0724.paper_core_evidence import load_normalizer


TARGETS = {
    "SawFM": (1, (0, 1, 2), SAW_PARAMETER_INDEX[1]),
    "VFT": (4, (3, 4, 5), VFT_PARAMETER_INDEX[1]),
    "ISRJ": (5, (6, 7, 8), None),
}


def split_identity_roles(identities: np.ndarray, seed: int = 20260801) -> dict[str, np.ndarray]:
    unique = np.unique(np.asarray(identities, dtype=np.int64))
    shuffled = unique[np.random.default_rng(seed).permutation(len(unique))]
    first = int(round(0.40 * len(unique)))
    second = first + int(round(0.30 * len(unique)))
    return {
        "rule_fit": np.sort(shuffled[:first]),
        "threshold_calibration": np.sort(shuffled[first:second]),
        "development_validation": np.sort(shuffled[second:]),
    }


def role_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def clopper_pearson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes >= trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def select_fpr_buffered_threshold(
    labels: np.ndarray, probability: np.ndarray, condition: np.ndarray,
    class_index: int, condition_ids: tuple[int, ...], rows: np.ndarray,
    maximum_condition_fpr_upper: float = 0.08, confidence: float = 0.95,
) -> dict[str, object]:
    relevant = rows & np.isin(condition, condition_ids)
    negative_values = probability[relevant & ~labels[:, class_index], class_index]
    candidates = np.unique(np.concatenate((
        np.array([0.0], dtype=np.float64),
        np.nextafter(negative_values.astype(np.float64), np.inf),
        np.array([1.0 + np.finfo(np.float64).eps]),
    )))
    audits = []
    for threshold in candidates:
        per_condition = {}
        per_condition_upper = {}
        for condition_id in condition_ids:
            negative = rows & (condition == condition_id) & ~labels[:, class_index]
            false_positives = int((probability[negative, class_index] >= threshold).sum())
            trials = int(negative.sum())
            per_condition[str(condition_id)] = float(false_positives / trials)
            per_condition_upper[str(condition_id)] = clopper_pearson_upper(
                false_positives, trials, confidence,
            )
        worst_fpr = max(per_condition.values())
        worst_upper = max(per_condition_upper.values())
        if worst_upper <= maximum_condition_fpr_upper + 1e-12:
            positive = relevant & labels[:, class_index]
            recall = float((probability[positive, class_index] >= threshold).mean())
            audits.append((
                float(threshold), recall, worst_fpr, worst_upper,
                per_condition, per_condition_upper,
            ))
            break
    if not audits:
        raise RuntimeError("No FPR-safe threshold candidate found")
    threshold, recall, worst_fpr, worst_upper, per_condition, per_condition_upper = audits[0]
    return {
        "threshold": threshold,
        "calibration_recall": recall,
        "maximum_condition_empirical_fpr": worst_fpr,
        "maximum_condition_fpr_upper": worst_upper,
        "per_condition_empirical_fpr": per_condition,
        "per_condition_fpr_upper": per_condition_upper,
        "fpr_upper_cap": maximum_condition_fpr_upper,
        "confidence": confidence,
    }


def parameter_nmae(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    prediction: np.ndarray, scale: np.ndarray, class_index: int, parameter_index: int,
    condition_ids: tuple[int, ...], rows: np.ndarray, miss_penalty: bool = False,
) -> float:
    valid = (
        rows & np.isin(saved["condition_id"], condition_ids)
        & saved["labels"][:, class_index].astype(bool)
        & saved["mask"][:, class_index, parameter_index].astype(bool)
    )
    predicted = probability[:, class_index] >= thresholds[class_index]
    error = np.abs(
        prediction[:, class_index, parameter_index] - saved["target"][:, class_index, parameter_index]
    ) / float(scale[class_index, parameter_index])
    if miss_penalty:
        return float(np.where(predicted, error, 1.0)[valid].mean())
    gated = valid & predicted
    return float(error[gated].mean()) if gated.any() else float("nan")


def class_miss_penalty_nmae(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    prediction: np.ndarray, scale: np.ndarray, class_index: int,
    condition_ids: tuple[int, ...], rows: np.ndarray,
) -> float:
    values = []
    for parameter_index in range(saved["target"].shape[2]):
        valid = (
            rows & np.isin(saved["condition_id"], condition_ids)
            & saved["labels"][:, class_index].astype(bool)
            & saved["mask"][:, class_index, parameter_index].astype(bool)
        )
        if not valid.any():
            continue
        predicted = probability[:, class_index] >= thresholds[class_index]
        error = np.abs(
            prediction[:, class_index, parameter_index] - saved["target"][:, class_index, parameter_index]
        ) / float(scale[class_index, parameter_index])
        values.append(float(np.where(predicted, error, 1.0)[valid].mean()))
    return float(np.mean(values))


def target_recall_fpr(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    class_index: int, condition_ids: tuple[int, ...], rows: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    predicted = probability[:, class_index] >= thresholds[class_index]
    positive = rows & np.isin(saved["condition_id"], condition_ids) & saved["labels"][:, class_index].astype(bool)
    per_condition_fpr = {}
    for condition_id in condition_ids:
        negative = rows & (saved["condition_id"] == condition_id) & ~saved["labels"][:, class_index].astype(bool)
        if negative.any():
            per_condition_fpr[str(condition_id)] = float(predicted[negative].mean())
    maximum_fpr = max(per_condition_fpr.values()) if per_condition_fpr else float("nan")
    return float(predicted[positive].mean()), maximum_fpr, per_condition_fpr


def worst_combination_parameter_nmae(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    prediction: np.ndarray, scale: np.ndarray, class_index: int, parameter_index: int,
    condition_ids: tuple[int, ...], rows: np.ndarray, minimum_gated_count: int = 10,
) -> float:
    combinations = np.packbits(saved["labels"].astype(np.uint8), axis=1, bitorder="little")[:, 0]
    values = []
    predicted = probability[:, class_index] >= thresholds[class_index]
    for combination in np.unique(combinations[rows]):
        selected = rows & (combinations == combination)
        gated_count = int((
            selected & np.isin(saved["condition_id"], condition_ids)
            & saved["labels"][:, class_index].astype(bool)
            & saved["mask"][:, class_index, parameter_index].astype(bool)
            & predicted
        ).sum())
        if gated_count < minimum_gated_count:
            continue
        value = parameter_nmae(
            saved, probability, thresholds, prediction, scale,
            class_index, parameter_index, condition_ids, selected,
        )
        if np.isfinite(value):
            values.append(value)
    return float(max(values)) if values else float("nan")


def gated_prediction(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    saw_max_complexity: int, vft_max_complexity: int,
    saw_max_disagreement: float = float("inf"),
    vft_max_disagreement: float = float("inf"),
    saw_allowed_patterns: tuple[int, ...] | list[int] | None = None,
    vft_allowed_patterns: tuple[int, ...] | list[int] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    prediction = saved["base_prediction"].copy()
    predicted = probability >= thresholds[None, :]
    complexity = predicted.sum(axis=1)
    patterns = np.packbits(predicted.astype(np.uint8), axis=1, bitorder="little")[:, 0]
    coverage = {}
    for name, class_index, parameter_index, maximum, maximum_disagreement, allowed_patterns in (
        ("SawFM", SAW_PARAMETER_INDEX[0], SAW_PARAMETER_INDEX[1], saw_max_complexity, saw_max_disagreement, saw_allowed_patterns),
        ("VFT", VFT_PARAMETER_INDEX[0], VFT_PARAMETER_INDEX[1], vft_max_complexity, vft_max_disagreement, vft_allowed_patterns),
    ):
        disagreement = np.abs(
            saved["joint_prediction"][:, class_index, parameter_index]
            - saved["base_prediction"][:, class_index, parameter_index]
        )
        use = predicted[:, class_index] & (complexity <= maximum) & (disagreement <= maximum_disagreement)
        if allowed_patterns is not None:
            use &= np.isin(patterns, np.asarray(allowed_patterns, dtype=np.uint8))
        prediction[use, class_index, parameter_index] = saved["joint_prediction"][use, class_index, parameter_index]
        coverage[name] = float(use.mean())
    return prediction, coverage


def reliable_predicted_patterns(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    scale: np.ndarray, class_index: int, parameter_index: int,
    condition_ids: tuple[int, ...], rows: np.ndarray, maximum_complexity: int,
    maximum_disagreement: float, minimum_count: int = 10,
    minimum_relative_improvement: float = 0.05,
) -> list[int]:
    predicted = probability >= thresholds[None, :]
    complexity = predicted.sum(axis=1)
    patterns = np.packbits(predicted.astype(np.uint8), axis=1, bitorder="little")[:, 0]
    disagreement = np.abs(
        saved["joint_prediction"][:, class_index, parameter_index]
        - saved["base_prediction"][:, class_index, parameter_index]
    )
    eligible = (
        rows & np.isin(saved["condition_id"], condition_ids)
        & saved["labels"][:, class_index].astype(bool)
        & saved["mask"][:, class_index, parameter_index].astype(bool)
        & predicted[:, class_index] & (complexity <= maximum_complexity)
        & (disagreement <= maximum_disagreement)
    )
    allowed = []
    for pattern in np.unique(patterns[eligible]):
        selected = eligible & (patterns == pattern)
        if selected.sum() < minimum_count:
            continue
        full_base_error = np.abs(
            saved["base_prediction"][:, class_index, parameter_index]
            - saved["target"][:, class_index, parameter_index]
        ) / float(scale[class_index, parameter_index])
        full_expert_error = np.abs(
            saved["joint_prediction"][:, class_index, parameter_index]
            - saved["target"][:, class_index, parameter_index]
        ) / float(scale[class_index, parameter_index])
        base_error = full_base_error[selected]
        expert_error = full_expert_error[selected]
        relative_improvement = float((base_error.mean() - expert_error.mean()) / base_error.mean())
        consistent = True
        subgroups = [selected & (saved["condition_id"] == condition_id) for condition_id in condition_ids]
        identity_fold = np.asarray(saved["receiver_noise_seed"], dtype=np.int64) % 3
        subgroups.extend(selected & (identity_fold == fold) for fold in range(3))
        for subgroup in subgroups:
            if subgroup.sum() < 5 or full_expert_error[subgroup].mean() > full_base_error[subgroup].mean():
                consistent = False
                break
        if relative_improvement >= minimum_relative_improvement and consistent:
            allowed.append(int(pattern))
    return sorted(allowed)


def choose_complexity_gate(
    saved: dict[str, np.ndarray], probability: np.ndarray, thresholds: np.ndarray,
    scale: np.ndarray, class_index: int, parameter_index: int,
    condition_ids: tuple[int, ...], rows: np.ndarray, maximum_allowed_complexity: int = 3,
) -> dict[str, object]:
    baseline = saved["base_prediction"]
    baseline_nmae = parameter_nmae(
        saved, probability, thresholds, baseline, scale,
        class_index, parameter_index, condition_ids, rows,
    )
    baseline_worst = worst_combination_parameter_nmae(
        saved, probability, thresholds, baseline, scale,
        class_index, parameter_index, condition_ids, rows,
    )
    candidates = []
    predicted = probability >= thresholds[None, :]
    complexity = predicted.sum(axis=1)
    disagreement = np.abs(
        saved["joint_prediction"][:, class_index, parameter_index]
        - saved["base_prediction"][:, class_index, parameter_index]
    )
    for maximum in range(0, maximum_allowed_complexity + 1):
        eligible_for_quantile = rows & predicted[:, class_index] & (complexity <= maximum)
        quantile_rows = [(0.0, -1.0)] if maximum == 0 or not eligible_for_quantile.any() else [
            (quantile, float(np.quantile(disagreement[eligible_for_quantile], quantile)))
            for quantile in (0.25, 0.50, 0.75, 0.90, 1.00)
        ]
        for disagreement_quantile, maximum_disagreement in quantile_rows:
            saw_max = maximum if class_index == SAW_PARAMETER_INDEX[0] else 0
            vft_max = maximum if class_index == VFT_PARAMETER_INDEX[0] else 0
            saw_disagreement = maximum_disagreement if class_index == SAW_PARAMETER_INDEX[0] else -1.0
            vft_disagreement = maximum_disagreement if class_index == VFT_PARAMETER_INDEX[0] else -1.0
            allowed_patterns = reliable_predicted_patterns(
                saved, probability, thresholds, scale, class_index, parameter_index,
                condition_ids, rows, maximum, maximum_disagreement,
            )
            saw_patterns = allowed_patterns if class_index == SAW_PARAMETER_INDEX[0] else []
            vft_patterns = allowed_patterns if class_index == VFT_PARAMETER_INDEX[0] else []
            prediction, coverage = gated_prediction(
                saved, probability, thresholds, saw_max, vft_max,
                saw_disagreement, vft_disagreement,
                saw_patterns, vft_patterns,
            )
            candidate_nmae = parameter_nmae(
                saved, probability, thresholds, prediction, scale,
                class_index, parameter_index, condition_ids, rows,
            )
            candidate_worst = worst_combination_parameter_nmae(
                saved, probability, thresholds, prediction, scale,
                class_index, parameter_index, condition_ids, rows,
            )
            improvement = float((baseline_nmae - candidate_nmae) / baseline_nmae)
            candidates.append({
                "maximum_predicted_complexity": maximum,
                "maximum_specialist_base_disagreement": maximum_disagreement,
                "disagreement_quantile": disagreement_quantile,
                "allowed_predicted_patterns": allowed_patterns,
                "predicted_gate_nmae": candidate_nmae,
                "relative_improvement": improvement,
                "worst_combination_nmae": candidate_worst,
                "worst_nonworse": bool(candidate_worst <= baseline_worst + 1e-12),
                "expert_call_coverage": coverage["SawFM" if class_index == 1 else "VFT"],
            })
    eligible = [row for row in candidates if row["worst_nonworse"]]
    selected = max(eligible, key=lambda row: (
        row["relative_improvement"], -row["maximum_predicted_complexity"],
        -row["maximum_specialist_base_disagreement"],
    ))
    return {
        "baseline_predicted_gate_nmae": baseline_nmae,
        "baseline_worst_combination_nmae": baseline_worst,
        "selected": selected,
        "maximum_allowed_complexity": maximum_allowed_complexity,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", default="outputs/p2f_stage_b_development_predictions_20260801")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--output-root", default="outputs/p2f_composition_safe_development_20260801")
    parser.add_argument("--calibration-fpr-upper-cap", type=float, default=0.08)
    parser.add_argument("--calibration-confidence", type=float, default=0.99)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    prediction_root = project / args.prediction_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((project / args.run_root / "protocol.json").read_text(encoding="utf-8"))
    scale = load_normalizer(protocol["normalizer"]).scale.numpy()

    first = np.load(prediction_root / f"seed_{MODEL_SEEDS[0]}" / "test_per_sample.npz")
    roles = split_identity_roles(first["receiver_noise_seed"])
    role_receipt = {
        name: {"identity_count": int(len(values)), "identity_sha256": role_hash(values)}
        for name, values in roles.items()
    }
    role_receipt["pairwise_overlap"] = {
        f"{left}__{right}": int(np.intersect1d(roles[left], roles[right]).size)
        for index, left in enumerate(roles) for right in tuple(roles)[index + 1:]
    }
    first.close()

    seed_results = []
    freeze_rows = []
    for seed in MODEL_SEEDS:
        prediction_path = prediction_root / f"seed_{seed}" / "test_per_sample.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        labels = saved["labels"].astype(bool)
        identity = saved["receiver_noise_seed"]
        role_masks = {name: np.isin(identity, values) for name, values in roles.items()}
        candidate_probability = saved["routed_probability"].copy()
        main_probability = saved["original_probability"].copy()
        candidate_thresholds = saved["routed_thresholds"].copy().astype(np.float64)
        main_thresholds = saved["original_thresholds"].copy().astype(np.float64)
        threshold_audit = {}
        for target_name, (class_index, condition_ids, _) in TARGETS.items():
            candidate_fit = select_fpr_buffered_threshold(
                labels, candidate_probability, saved["condition_id"], class_index,
                condition_ids, role_masks["threshold_calibration"],
                args.calibration_fpr_upper_cap, args.calibration_confidence,
            )
            main_fit = select_fpr_buffered_threshold(
                labels, main_probability, saved["condition_id"], class_index,
                condition_ids, role_masks["threshold_calibration"],
                args.calibration_fpr_upper_cap, args.calibration_confidence,
            )
            candidate_thresholds[class_index] = candidate_fit["threshold"]
            main_thresholds[class_index] = main_fit["threshold"]
            threshold_audit[target_name] = {"candidate": candidate_fit, "fpr_matched_main": main_fit}

        saw_gate = choose_complexity_gate(
            saved, candidate_probability, candidate_thresholds, scale,
            TARGETS["SawFM"][0], TARGETS["SawFM"][2], TARGETS["SawFM"][1], role_masks["rule_fit"],
        )
        vft_gate = choose_complexity_gate(
            saved, candidate_probability, candidate_thresholds, scale,
            TARGETS["VFT"][0], TARGETS["VFT"][2], TARGETS["VFT"][1], role_masks["rule_fit"],
        )
        saw_max = int(saw_gate["selected"]["maximum_predicted_complexity"])
        vft_max = int(vft_gate["selected"]["maximum_predicted_complexity"])
        saw_disagreement = float(saw_gate["selected"]["maximum_specialist_base_disagreement"])
        vft_disagreement = float(vft_gate["selected"]["maximum_specialist_base_disagreement"])
        saw_patterns = [int(value) for value in saw_gate["selected"]["allowed_predicted_patterns"]]
        vft_patterns = [int(value) for value in vft_gate["selected"]["allowed_predicted_patterns"]]
        candidate_prediction, call_coverage = gated_prediction(
            saved, candidate_probability, candidate_thresholds, saw_max, vft_max,
            saw_disagreement, vft_disagreement,
            saw_patterns, vft_patterns,
        )
        validation = role_masks["development_validation"]
        target_metrics = {}
        for target_name, (class_index, condition_ids, parameter_index) in TARGETS.items():
            candidate_recall, candidate_fpr, per_condition_fpr = target_recall_fpr(
                saved, candidate_probability, candidate_thresholds, class_index, condition_ids, validation,
            )
            main_recall, main_fpr, _ = target_recall_fpr(
                saved, main_probability, main_thresholds, class_index, condition_ids, validation,
            )
            row = {
                "candidate_recall": candidate_recall,
                "fpr_matched_main_recall": main_recall,
                "recall_gain": candidate_recall - main_recall,
                "candidate_maximum_condition_fpr": candidate_fpr,
                "fpr_matched_main_maximum_condition_fpr": main_fpr,
                "candidate_per_condition_fpr": per_condition_fpr,
            }
            if parameter_index is not None:
                base_nmae = parameter_nmae(
                    saved, candidate_probability, candidate_thresholds, saved["base_prediction"], scale,
                    class_index, parameter_index, condition_ids, validation,
                )
                candidate_nmae = parameter_nmae(
                    saved, candidate_probability, candidate_thresholds, candidate_prediction, scale,
                    class_index, parameter_index, condition_ids, validation,
                )
                row.update({
                    "baseline_predicted_gate_nmae": base_nmae,
                    "candidate_predicted_gate_nmae": candidate_nmae,
                    "relative_parameter_improvement": (base_nmae - candidate_nmae) / base_nmae,
                    "baseline_worst_combination_nmae": worst_combination_parameter_nmae(
                        saved, candidate_probability, candidate_thresholds, saved["base_prediction"], scale,
                        class_index, parameter_index, condition_ids, validation,
                    ),
                    "candidate_worst_combination_nmae": worst_combination_parameter_nmae(
                        saved, candidate_probability, candidate_thresholds, candidate_prediction, scale,
                        class_index, parameter_index, condition_ids, validation,
                    ),
                })
            if saved["mask"][:, class_index, :].any():
                candidate_miss = class_miss_penalty_nmae(
                    saved, candidate_probability, candidate_thresholds, candidate_prediction, scale,
                    class_index, condition_ids, validation,
                )
                main_miss = class_miss_penalty_nmae(
                    saved, main_probability, main_thresholds, saved["base_prediction"], scale,
                    class_index, condition_ids, validation,
                )
                row.update({
                    "candidate_miss_penalty_nmae": candidate_miss,
                    "fpr_matched_main_miss_penalty_nmae": main_miss,
                    "miss_penalty_delta": candidate_miss - main_miss,
                })
            target_metrics[target_name] = row
        seed_result = {
            "seed": seed,
            "threshold_audit": threshold_audit,
            "sawfm_complexity_gate": saw_gate,
            "vft_complexity_gate": vft_gate,
            "selected_candidate_thresholds": candidate_thresholds.tolist(),
            "selected_fpr_matched_main_thresholds": main_thresholds.tolist(),
            "selected_sawfm_max_predicted_complexity": saw_max,
            "selected_vft_max_predicted_complexity": vft_max,
            "selected_sawfm_max_disagreement": saw_disagreement,
            "selected_vft_max_disagreement": vft_disagreement,
            "selected_sawfm_allowed_predicted_patterns": saw_patterns,
            "selected_vft_allowed_predicted_patterns": vft_patterns,
            "expert_call_coverage_all_rows": call_coverage,
            "development_validation": target_metrics,
        }
        seed_results.append(seed_result)
        freeze_rows.append({
            "seed": seed,
            "candidate_thresholds": candidate_thresholds.tolist(),
            "fpr_matched_main_thresholds": main_thresholds.tolist(),
            "sawfm_max_predicted_complexity": saw_max,
            "vft_max_predicted_complexity": vft_max,
            "sawfm_max_disagreement": saw_disagreement,
            "vft_max_disagreement": vft_disagreement,
            "sawfm_allowed_predicted_patterns": saw_patterns,
            "vft_allowed_predicted_patterns": vft_patterns,
            "source_prediction_sha256": sha256_file(prediction_path),
        })

    maximum_fpr = max(
        row["development_validation"][target]["candidate_maximum_condition_fpr"]
        for row in seed_results for target in TARGETS
    )
    isrj_gain = [row["development_validation"]["ISRJ"]["recall_gain"] for row in seed_results]
    saw_gain = [row["development_validation"]["SawFM"]["relative_parameter_improvement"] for row in seed_results]
    vft_gain = [row["development_validation"]["VFT"]["relative_parameter_improvement"] for row in seed_results]
    miss_deltas = {
        target: [row["development_validation"][target].get("miss_penalty_delta", float("nan")) for row in seed_results]
        for target in TARGETS
    }
    worst_deltas = {
        target: [
            row["development_validation"][target]["candidate_worst_combination_nmae"]
            - row["development_validation"][target]["baseline_worst_combination_nmae"]
            for row in seed_results
        ]
        for target in ("SawFM", "VFT")
    }
    gates = {
        "maximum_target_fpr_at_most_0p10": bool(maximum_fpr <= 0.10 + 1e-12),
        "isrj_mean_recall_gain_at_least_0p05": bool(np.mean(isrj_gain) >= 0.05),
        "isrj_three_of_three_positive": bool(np.all(np.asarray(isrj_gain) > 0)),
        "sawfm_mean_parameter_improvement_at_least_0p05": bool(np.mean(saw_gain) >= 0.05),
        "sawfm_three_of_three_positive": bool(np.all(np.asarray(saw_gain) > 0)),
        "vft_mean_parameter_improvement_at_least_0p05": bool(np.mean(vft_gain) >= 0.05),
        "vft_three_of_three_positive": bool(np.all(np.asarray(vft_gain) > 0)),
        "miss_penalty_nonworse_vs_fpr_matched_main": bool(
            all(np.nanmean(values) <= 0 for values in miss_deltas.values())
        ),
        "worst_combination_nonworse_vs_same_threshold_baseline": bool(
            all(np.mean(values) <= 0 for values in worst_deltas.values())
        ),
    }
    method_freeze = {
        "experiment": "P2f composition-safe calibration and complexity-aware parameter experts",
        "source_development_prediction_root": str(prediction_root.resolve()),
        "source_method_freeze_sha256": json.loads(
            (prediction_root / "method_freeze.json").read_text(encoding="utf-8")
        )["source_method_freeze_sha256"],
        "identity_roles": role_receipt,
        "calibration_fpr_upper_cap": args.calibration_fpr_upper_cap,
        "calibration_confidence": args.calibration_confidence,
        "threshold_rule": "lowest score threshold with every target-condition one-sided Clopper-Pearson FPR upper bound <= cap",
        "expert_rule": "use specialist only when target is predicted, total predicted component count is within the fitted integer, and specialist-main disagreement is below the fitted quantile threshold",
        "expert_complexity_search_space": [0, 1, 2, 3],
        "expert_disagreement_quantiles": [0.25, 0.50, 0.75, 0.90, 1.00],
        "expert_pattern_rule": "predicted label bitmask observed at least 10 times, improves pooled target-parameter NMAE by at least 5 percent, and is nondegrading in every weakness condition and each of three identity folds on rule_fit",
        "worst_combination_minimum_gated_count": 10,
        "selection_roles": {
            "thresholds": "threshold_calibration only",
            "complexity_gates": "rule_fit only",
            "promotion": "development_validation only",
        },
        "seed_freezes": freeze_rows,
        "confirmation_data_used": False,
        "locked_test_used": False,
    }
    summary = {
        "development_results": seed_results,
        "aggregate": {
            "maximum_target_fpr": maximum_fpr,
            "isrj_recall_gain_per_seed": isrj_gain,
            "mean_isrj_recall_gain": float(np.mean(isrj_gain)),
            "sawfm_parameter_improvement_per_seed": saw_gain,
            "mean_sawfm_parameter_improvement": float(np.mean(saw_gain)),
            "vft_parameter_improvement_per_seed": vft_gain,
            "mean_vft_parameter_improvement": float(np.mean(vft_gain)),
            "miss_penalty_deltas": miss_deltas,
            "worst_combination_deltas": worst_deltas,
        },
        "gates": gates,
        "development_promoted": bool(all(gates.values())),
        "evidence_role": "development only",
        "locked_test_used": False,
    }
    (output_root / "method_freeze.json").write_text(json.dumps(method_freeze, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"] | {"gates": gates, "development_promoted": summary["development_promoted"]}, indent=2))


if __name__ == "__main__":
    main()
