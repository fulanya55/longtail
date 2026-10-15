"""Calibrate frozen C2 SawFM scores across ordinary and weak-power identities."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.engine.calibration import apply_temperature  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _evaluate_variant, _sha256  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_label_local_saw import F2_MODEL_NAME, _load_h2_head  # noqa: E402
from experiments.run_route_a_sawfm_expert_routing import (  # noqa: E402
    _calibrate_expert,
    _expert_scores,
    _extract_base_indices,
    _extract_power_identities,
    _load_expert,
)
from experiments.run_route_a_sawfm_fixed_fusion_cross_power import _fuse, _sigmoid  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import (  # noqa: E402
    _development_pairs,
    _level_lookup,
    _stratified_triplets,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


SAW_INDEX = 1
VARIANTS = ("D0_OrdinaryThreshold", "D1_PowerPlatt0p5", "D2_DualDomainRobust")


def _split_matched_positions_by_combo(selection, labels, seed):
    positive_indices = selection["positive_base_indices"]
    keys = [tuple(np.delete(labels[index], SAW_INDEX).tolist()) for index in positive_indices]
    rng = np.random.default_rng(int(seed))
    calibration = []
    threshold = []
    for key in sorted(set(keys)):
        positions = np.asarray([index for index, value in enumerate(keys) if value == key], dtype=np.int64)
        rng.shuffle(positions)
        split = len(positions) // 2
        calibration.extend(positions[:split].tolist())
        threshold.extend(positions[split:].tolist())
    return np.asarray(calibration, dtype=np.int64), np.asarray(threshold, dtype=np.int64)


def _fit_dual_platt(scores, targets, domains):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    targets = np.asarray(targets, dtype=np.uint8)
    domains = np.asarray(domains, dtype=np.uint8)
    weights = np.zeros(len(targets), dtype=np.float64)
    for domain in (0, 1):
        mask = domains == domain
        weights[mask] = len(targets) / (2.0 * max(int(mask.sum()), 1))
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(scores, targets, sample_weight=weights)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _platt_probability(scores, coefficient, intercept):
    return _sigmoid(coefficient * np.asarray(scores, dtype=np.float64) + intercept)


def _select_robust_threshold(ordinary_prob, ordinary_targets, power_by_level, max_fpr):
    rows = []
    best = None
    ordinary_targets = np.asarray(ordinary_targets, dtype=np.uint8)
    ordinary_negative = ordinary_targets == 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        ordinary_prediction = ordinary_prob >= threshold
        ordinary_fpr = float(ordinary_prediction[ordinary_negative].mean())
        ordinary_f1 = float(f1_score(ordinary_targets, ordinary_prediction, zero_division=0))
        recalls = []
        fprs = []
        for payload in power_by_level.values():
            recalls.append(float((payload["positive"] >= threshold).mean()))
            fprs.append(float((payload["negative"] >= threshold).mean()))
        feasible = ordinary_fpr <= max_fpr + 1e-12 and all(
            value <= max_fpr + 1e-12 for value in fprs
        )
        row = {
            "threshold": float(threshold),
            "mean_power_recall": float(np.mean(recalls)),
            "min_power_recall": float(np.min(recalls)),
            "max_power_fpr": float(np.max(fprs)),
            "ordinary_fpr": ordinary_fpr,
            "ordinary_f1": ordinary_f1,
            "feasible": bool(feasible),
        }
        rows.append(row)
        key = (row["mean_power_recall"], row["min_power_recall"], ordinary_f1, -abs(threshold - 0.5))
        if feasible and (best is None or key > best[0]):
            best = (key, row)
    if best is None:
        raise RuntimeError("No threshold satisfies ordinary and power FPR constraints")
    return float(best[1]["threshold"]), rows


def _decision_metrics(positive_probability, negative_probability, threshold):
    return {
        "positive_recall": float((positive_probability >= threshold).mean()),
        "hard_negative_fpr": float((negative_probability >= threshold).mean()),
        "positive_probability_mean": float(positive_probability.mean()),
        "negative_probability_mean": float(negative_probability.mean()),
        "threshold": float(threshold),
    }


def _run_seed(args, config, labels, identities, seed, device):
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    receipt = _read(source_dir / "run_receipt.json")
    if receipt.get("locked_test_used") or receipt["split_sha256"] != plan["split_sha256"]:
        raise ValueError("F2 source invalid or split drifted")
    backbone, f2_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    base_head, h2_checkpoint, h2_path, _ = _load_h2_head(seed, args.frozen_head_root, device)
    expert_path = args.expert_root / "seed_{}".format(seed) / "S2_TeacherHardNeg" / "best_saw_stage2_evidence.pth"
    expert, _ = _load_expert(expert_path, device)
    routing_path = args.routing_root / "seed_{}".format(seed) / "best_router.pth"
    routing = torch.load(routing_path, map_location="cpu", weights_only=False)
    expert_coefficient = float(routing["platt_coefficient"])
    expert_intercept = float(routing["platt_intercept"])

    expert_selection = _stratified_triplets(plan["train_indices"], labels, identities, 40, 161000 + seed)
    expert_used = set(
        expert_selection["positive_base_indices"].tolist()
        + expert_selection["hard_negative_base_indices"].tolist()
        + expert_selection["random_negative_base_indices"].tolist()
    )
    remaining_after_expert = np.asarray(
        [index for index in plan["train_indices"] if int(index) not in expert_used], dtype=np.int64
    )
    route_selection = _stratified_triplets(remaining_after_expert, labels, identities, 20, 201000 + seed)
    route_used = set(
        route_selection["positive_base_indices"].tolist()
        + route_selection["hard_negative_base_indices"].tolist()
        + route_selection["random_negative_base_indices"].tolist()
    )
    remaining = np.asarray(
        [index for index in remaining_after_expert if int(index) not in route_used], dtype=np.int64
    )
    decision_selection = _stratified_triplets(
        remaining, labels, identities, args.pairs_per_combo, 241000 + seed
    )
    calibration_positions, threshold_positions = _split_matched_positions_by_combo(
        decision_selection, labels, 242000 + seed
    )
    development = _development_pairs(plan["early_stop_indices"], labels, identities, 162000 + seed)

    excluded_ids = set(identities[list(expert_used | route_used)].tolist())
    decision_ids = set(
        decision_selection["positive_identities"].tolist()
        + decision_selection["hard_negative_identities"].tolist()
    )
    development_ids = set(
        development["positive_identities"].tolist() + development["hard_negative_identities"].tolist()
    )
    overlap = excluded_ids.intersection(decision_ids) | decision_ids.intersection(development_ids)
    if overlap:
        raise ValueError("Decision calibration identities overlap excluded or development roles")

    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    level_info = _level_lookup(power_files, args.levels)
    print("seed={} extracting dual-domain decision roles".format(seed), flush=True)

    def scored_base(indices):
        group = _expert_scores(expert, _extract_base_indices(
            args, config, backbone, base_head, h2_checkpoint, indices, device
        ), device)
        expert_cal = _calibrate_expert(group["expert"], expert_coefficient, expert_intercept)
        group["c2"] = _fuse(group["base_logits"], expert_cal, 0.80)[:, SAW_INDEX]
        return group

    def scored_power(identity_values, level):
        group = _expert_scores(expert, _extract_power_identities(
            args, config, backbone, base_head, h2_checkpoint,
            power_files[int(level)], level_info[int(level)], identity_values, device
        ), device)
        expert_cal = _calibrate_expert(group["expert"], expert_coefficient, expert_intercept)
        group["c2"] = _fuse(group["base_logits"], expert_cal, 0.80)[:, SAW_INDEX]
        return group

    ordinary = {
        "calibration": scored_base(plan["calibration_indices"]),
        "threshold": scored_base(plan["threshold_indices"]),
        "early": scored_base(plan["early_stop_indices"]),
    }
    power_roles = {"calibration": {}, "threshold": {}, "development": {}}
    for level in args.levels:
        for role, positions in (("calibration", calibration_positions), ("threshold", threshold_positions)):
            positive = scored_power(decision_selection["positive_identities"][positions], level)
            negative = scored_power(decision_selection["hard_negative_identities"][positions], level)
            power_roles[role][str(level)] = {"positive": positive, "negative": negative}
        power_roles["development"][str(level)] = {
            "positive": scored_power(development["positive_identities"], level),
            "negative": scored_power(development["hard_negative_identities"], level),
        }

    fit_scores = [ordinary["calibration"]["c2"]]
    fit_targets = [labels[plan["calibration_indices"], SAW_INDEX]]
    fit_domains = [np.zeros(len(plan["calibration_indices"]), dtype=np.uint8)]
    for level in args.levels:
        role = power_roles["calibration"][str(level)]
        fit_scores.extend([role["positive"]["c2"], role["negative"]["c2"]])
        fit_targets.extend([np.ones(len(role["positive"]["c2"])), np.zeros(len(role["negative"]["c2"]))])
        fit_domains.extend([np.ones(len(role["positive"]["c2"])), np.ones(len(role["negative"]["c2"]))])
    coefficient, intercept = _fit_dual_platt(
        np.concatenate(fit_scores), np.concatenate(fit_targets), np.concatenate(fit_domains)
    )
    ordinary_threshold_probability = _platt_probability(ordinary["threshold"]["c2"], coefficient, intercept)
    power_threshold_probability = {}
    for level in args.levels:
        role = power_roles["threshold"][str(level)]
        power_threshold_probability[str(level)] = {
            "positive": _platt_probability(role["positive"]["c2"], coefficient, intercept),
            "negative": _platt_probability(role["negative"]["c2"], coefficient, intercept),
        }
    robust_threshold, threshold_search = _select_robust_threshold(
        ordinary_threshold_probability,
        labels[plan["threshold_indices"], SAW_INDEX],
        power_threshold_probability,
        args.max_negative_fpr,
    )

    cross_power_root = args.cross_power_root / "seed_{}".format(seed) / "C2_A080"
    d0_calibration = _read(cross_power_root / "calibration.json")
    d0_threshold = float(_read(cross_power_root / "thresholds.json")["modes"]["weak_recall_fpr"][SAW_INDEX])
    variants = {
        "D0_OrdinaryThreshold": {"threshold": d0_threshold},
        "D1_PowerPlatt0p5": {"threshold": 0.5},
        "D2_DualDomainRobust": {"threshold": robust_threshold},
    }
    rows = []
    seed_root = args.output_root / "seed_{}".format(seed)
    for variant, specification in variants.items():
        threshold = float(specification["threshold"])
        if variant == "D0_OrdinaryThreshold":
            ordinary_probability = _sigmoid(apply_temperature(
                ordinary["early"]["c2"], d0_calibration["temperature"]
            ))
        else:
            ordinary_probability = _platt_probability(ordinary["early"]["c2"], coefficient, intercept)
        ordinary_target = labels[plan["early_stop_indices"], SAW_INDEX]
        ordinary_positive = ordinary_target == 1
        ordinary_negative = ~ordinary_positive
        power_metrics = {}
        for level in args.levels:
            role = power_roles["development"][str(level)]
            if variant == "D0_OrdinaryThreshold":
                positive_probability = _sigmoid(apply_temperature(role["positive"]["c2"], d0_calibration["temperature"]))
                negative_probability = _sigmoid(apply_temperature(role["negative"]["c2"], d0_calibration["temperature"]))
            else:
                positive_probability = _platt_probability(role["positive"]["c2"], coefficient, intercept)
                negative_probability = _platt_probability(role["negative"]["c2"], coefficient, intercept)
            power_metrics[str(level)] = _decision_metrics(
                positive_probability, negative_probability, threshold
            )
        recalls = [power_metrics[str(level)]["positive_recall"] for level in args.levels]
        fprs = [power_metrics[str(level)]["hard_negative_fpr"] for level in args.levels]
        row = {
            "variant": variant,
            "seed": int(seed),
            "threshold": threshold,
            "ordinary_saw_recall": float((ordinary_probability[ordinary_positive] >= threshold).mean()),
            "ordinary_saw_fpr": float((ordinary_probability[ordinary_negative] >= threshold).mean()),
            "power": power_metrics,
            "mean_power_recall": float(np.mean(recalls)),
            "min_power_recall": float(np.min(recalls)),
            "max_power_fpr": float(np.max(fprs)),
            "platt_coefficient": coefficient if variant != "D0_OrdinaryThreshold" else None,
            "platt_intercept": intercept if variant != "D0_OrdinaryThreshold" else None,
            "split_sha256": plan["split_sha256"],
            "selection_sha256": decision_selection["selection_sha256"],
            "identity_overlap": 0,
            "locked_test_used": False,
        }
        output_dir = seed_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_payload = {
            "ordinary_probabilities": ordinary_probability,
            "ordinary_targets": ordinary_target,
        }
        for level in args.levels:
            role = power_roles["development"][str(level)]
            if variant == "D0_OrdinaryThreshold":
                prediction_payload["positive_{}db".format(level)] = _sigmoid(
                    apply_temperature(role["positive"]["c2"], d0_calibration["temperature"])
                )
                prediction_payload["negative_{}db".format(level)] = _sigmoid(
                    apply_temperature(role["negative"]["c2"], d0_calibration["temperature"])
                )
            else:
                prediction_payload["positive_{}db".format(level)] = _platt_probability(
                    role["positive"]["c2"], coefficient, intercept
                )
                prediction_payload["negative_{}db".format(level)] = _platt_probability(
                    role["negative"]["c2"], coefficient, intercept
                )
        np.savez_compressed(str(output_dir / "predictions.npz"), **prediction_payload)
        _write(output_dir / "metrics.json", row)
        _write(output_dir / "run_receipt.json", {
            "status": "completed", "variant": variant, "seed": int(seed),
            "split_sha256": plan["split_sha256"], "selection_sha256": decision_selection["selection_sha256"],
            "identity_overlap": 0, "locked_test_used": False,
        })
        rows.append(row)
    _write(seed_root / "threshold_search.json", {
        "selected_threshold": robust_threshold,
        "platt_coefficient": coefficient,
        "platt_intercept": intercept,
        "candidates": threshold_search,
    })
    audit = {
        "seed": int(seed), "calibration_pair_count": int(len(calibration_positions)),
        "threshold_pair_count": int(len(threshold_positions)), "levels": list(args.levels),
        "excluded_decision_overlap": 0, "decision_development_overlap": 0,
        "selection_sha256": decision_selection["selection_sha256"],
        "f2_checkpoint_sha256": _sha256(f2_path), "h2_checkpoint_sha256": _sha256(h2_path),
        "expert_checkpoint_sha256": _sha256(expert_path), "locked_test_used": False,
    }
    _write(seed_root / "identity_audit.json", audit)
    return rows, audit


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base labels and identities are misaligned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits = [], []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(args, config, labels, identities, int(seed), device)
        rows.extend(seed_rows)
        audits.append(audit)
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: _mean_ci([row[metric] for row in selected], 251000 + variant_index * 100 + metric_index)
            for metric_index, metric in enumerate(("ordinary_saw_recall", "ordinary_saw_fpr", "mean_power_recall", "min_power_recall", "max_power_fpr"))
        }
        aggregate[variant]["by_level"] = {
            str(level): {
                metric: _mean_ci([row["power"][str(level)][metric] for row in selected], 252000 + variant_index * 1000 + int(level) * 10 + metric_index)
                for metric_index, metric in enumerate(("positive_recall", "hard_negative_fpr"))
            }
            for level in args.levels
        }
    d0 = {row["seed"]: row for row in rows if row["variant"] == "D0_OrdinaryThreshold"}
    d2 = [row for row in rows if row["variant"] == "D2_DualDomainRobust"]
    per_seed_pass = []
    for row in d2:
        baseline = d0[row["seed"]]
        passed = (
            row["mean_power_recall"] >= 0.50
            and row["min_power_recall"] >= 0.35
            and row["max_power_fpr"] <= 0.20 + 1e-12
            and row["ordinary_saw_fpr"] - baseline["ordinary_saw_fpr"] <= 0.02 + 1e-12
            and row["ordinary_saw_recall"] - baseline["ordinary_saw_recall"] >= -0.05 - 1e-12
        )
        per_seed_pass.append({"seed": row["seed"], "passed": bool(passed)})
    checks = {
        "mean_power_recall_0p50": aggregate["D2_DualDomainRobust"]["mean_power_recall"]["mean"] >= 0.50,
        "each_level_recall_0p35": all(aggregate["D2_DualDomainRobust"]["by_level"][str(level)]["positive_recall"]["mean"] >= 0.35 for level in args.levels),
        "each_level_fpr_0p20": all(aggregate["D2_DualDomainRobust"]["by_level"][str(level)]["hard_negative_fpr"]["mean"] <= 0.20 for level in args.levels),
        "ordinary_fpr_delta_0p02": aggregate["D2_DualDomainRobust"]["ordinary_saw_fpr"]["mean"] - aggregate["D0_OrdinaryThreshold"]["ordinary_saw_fpr"]["mean"] <= 0.02,
        "ordinary_recall_delta_minus_0p05": aggregate["D2_DualDomainRobust"]["ordinary_saw_recall"]["mean"] - aggregate["D0_OrdinaryThreshold"]["ordinary_saw_recall"]["mean"] >= -0.05,
        "two_seed_pass": sum(row["passed"] for row in per_seed_pass) >= 2,
        "identity_and_locked_test_audit": all(a["excluded_decision_overlap"] == 0 and a["decision_development_overlap"] == 0 and not a["locked_test_used"] for a in audits),
    }
    gate = {"checks": checks, "per_seed": per_seed_pass, "passed": all(checks.values())}
    payload = {"status": "completed", "device": str(device), "locked_test_used": False,
               "protocol": {"seeds": list(args.seeds), "levels": list(args.levels), "variants": list(VARIANTS), "pairs_per_combo": args.pairs_per_combo, "max_negative_fpr": args.max_negative_fpr},
               "audits": audits, "per_run": rows, "aggregate": aggregate, "promotion_gate": gate}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_power_robust_decision_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--routing-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_expert_routing")
    parser.add_argument("--cross-power-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_fixed_fusion_cross_power")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_power_robust_decision")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=20)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
