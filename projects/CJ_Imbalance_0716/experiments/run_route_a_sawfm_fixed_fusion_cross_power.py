"""Confirm one fixed SawFM fusion weight across power and composition strata."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature  # noqa: E402
from experiments.analyze_route_a_pam_iq_v2_scores import _component_power  # noqa: E402
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
from experiments.run_route_a_sawfm_stage2_power_teacher_student import (  # noqa: E402
    _binary_metrics,
    _development_pairs,
    _level_lookup,
    _stratified_triplets,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


SAW_INDEX = 1
SWEPT_INDICES = (3, 4, 5)
CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
VARIANTS = {
    "C0_H2": 0.0,
    "C1_A075": 0.75,
    "C2_A080": 0.80,
    "C3_A085": 0.85,
}


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _fuse(base_logits, expert_logits, alpha):
    base = np.asarray(base_logits, dtype=np.float32)
    result = base.copy()
    result[:, SAW_INDEX] = (
        (1.0 - float(alpha)) * base[:, SAW_INDEX]
        + float(alpha) * np.asarray(expert_logits, dtype=np.float32)
    )
    return result


def _power_component_matrix(labels, level):
    labels = np.asarray(labels, dtype=np.uint8)
    values = np.full(labels.shape, np.nan, dtype=np.float32)
    values[:, :3] = np.where(labels[:, :3] > 0, 30.0, np.nan)
    values[:, 3:] = np.where(labels[:, 3:] > 0, float(level), np.nan)
    return values


def _stratum_keys(labels):
    labels = np.asarray(labels, dtype=np.uint8)
    cardinality = labels.sum(axis=1).astype(np.int64)
    signature = np.asarray(
        [
            "+".join(
                CLASS_NAMES[index]
                for index in SWEPT_INDICES
                if int(row[index]) == 1
            )
            for row in labels
        ],
        dtype=object,
    )
    return {"cardinality": cardinality, "swept_signature": signature}


def _stratified_metrics(positive_scores, negative_scores, positive_labels, min_pairs):
    result = []
    for stratum_type, keys in _stratum_keys(positive_labels).items():
        for value in sorted(set(keys.tolist()), key=str):
            mask = keys == value
            if int(mask.sum()) < int(min_pairs):
                continue
            metrics = _binary_metrics(positive_scores[mask], negative_scores[mask])
            result.append(
                {
                    "stratum_type": stratum_type,
                    "stratum_value": str(value),
                    "pair_count": int(mask.sum()),
                    "auroc": metrics["auroc"],
                    "aupr": metrics["aupr"],
                }
            )
    return result


def _power_metrics(
    positive_logits,
    negative_logits,
    positive_labels,
    level,
    temperature,
    threshold,
    min_stratum_pairs,
):
    positive_scores = np.asarray(positive_logits[:, SAW_INDEX], dtype=np.float32)
    negative_scores = np.asarray(negative_logits[:, SAW_INDEX], dtype=np.float32)
    rank = _binary_metrics(positive_scores, negative_scores)
    positive_probability = _sigmoid(apply_temperature(positive_scores, temperature))
    negative_probability = _sigmoid(apply_temperature(negative_scores, temperature))
    weak = compute_weak_component_mask(
        positive_labels,
        _power_component_matrix(positive_labels, level),
        weak_threshold_db=10.0,
    )[:, SAW_INDEX]
    return {
        "auroc": rank["auroc"],
        "aupr": rank["aupr"],
        "positive_count": rank["positive_count"],
        "negative_count": rank["negative_count"],
        "positive_recall": float((positive_probability >= threshold).mean()),
        "weak_positive_count": int(weak.sum()),
        "weak_positive_recall": (
            float((positive_probability[weak] >= threshold).mean())
            if weak.any()
            else None
        ),
        "hard_negative_fpr": float((negative_probability >= threshold).mean()),
        "positive_probability_mean": float(positive_probability.mean()),
        "negative_probability_mean": float(negative_probability.mean()),
        "strata": _stratified_metrics(
            positive_scores,
            negative_scores,
            positive_labels,
            min_stratum_pairs,
        ),
    }


def _identity_audit(plan, labels, identities, development, seed):
    expert_selection = _stratified_triplets(
        plan["train_indices"], labels, identities, 40, 161000 + int(seed)
    )
    expert_ids = set(
        expert_selection["positive_identities"].tolist()
        + expert_selection["hard_negative_identities"].tolist()
        + expert_selection["random_negative_identities"].tolist()
    )
    development_ids = set(
        development["positive_identities"].tolist()
        + development["hard_negative_identities"].tolist()
    )
    overlap = expert_ids.intersection(development_ids)
    if overlap:
        raise ValueError("Expert training and cross-power evaluation identities overlap")
    return {
        "expert_development_identity_overlap": int(len(overlap)),
        "expert_selection_sha256": expert_selection["selection_sha256"],
        "development_selection_sha256": development["selection_sha256"],
    }


def _run_seed(args, config, labels, weak_mask, identities, seed, device):
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    source_receipt = _read(source_dir / "run_receipt.json")
    if source_receipt.get("locked_test_used") or source_receipt["split_sha256"] != plan["split_sha256"]:
        raise ValueError("F2 source receipt invalid or split drifted")
    backbone, f2_checkpoint_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    base_head, h2_checkpoint, h2_checkpoint_path, _ = _load_h2_head(
        seed, args.frozen_head_root, device
    )
    expert_path = (
        args.expert_root
        / "seed_{}".format(seed)
        / "S2_TeacherHardNeg"
        / "best_saw_stage2_evidence.pth"
    )
    expert, _ = _load_expert(expert_path, device)
    router_path = args.routing_root / "seed_{}".format(seed) / "best_router.pth"
    router_checkpoint = torch.load(router_path, map_location="cpu", weights_only=False)
    coefficient = float(router_checkpoint["platt_coefficient"])
    intercept = float(router_checkpoint["platt_intercept"])
    development = _development_pairs(
        plan["early_stop_indices"], labels, identities, 162000 + int(seed)
    )
    audit = _identity_audit(plan, labels, identities, development, seed)
    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    missing = sorted(set(args.levels).difference(power_files))
    if missing:
        raise ValueError("Missing power levels: {}".format(missing))
    level_info = _level_lookup(power_files, args.levels)
    print("seed={} extracting ordinary and cross-power roles".format(seed), flush=True)
    role_indices = {
        "early": np.asarray(plan["early_stop_indices"], dtype=np.int64),
        "calibration": np.asarray(plan["calibration_indices"], dtype=np.int64),
        "threshold": np.asarray(plan["threshold_indices"], dtype=np.int64),
    }
    ordinary = {}
    for role, indices in role_indices.items():
        ordinary[role] = _expert_scores(
            expert,
            _extract_base_indices(
                args,
                config,
                backbone,
                base_head,
                h2_checkpoint,
                indices,
                device,
            ),
            device,
        )
        ordinary[role]["expert_cal"] = _calibrate_expert(
            ordinary[role]["expert"], coefficient, intercept
        )
    cross_power = {}
    for level in args.levels:
        positive = _expert_scores(
            expert,
            _extract_power_identities(
                args,
                config,
                backbone,
                base_head,
                h2_checkpoint,
                power_files[int(level)],
                level_info[int(level)],
                development["positive_identities"],
                device,
            ),
            device,
        )
        negative = _expert_scores(
            expert,
            _extract_power_identities(
                args,
                config,
                backbone,
                base_head,
                h2_checkpoint,
                power_files[int(level)],
                level_info[int(level)],
                development["hard_negative_identities"],
                device,
            ),
            device,
        )
        positive["expert_cal"] = _calibrate_expert(positive["expert"], coefficient, intercept)
        negative["expert_cal"] = _calibrate_expert(negative["expert"], coefficient, intercept)
        cross_power[int(level)] = {"positive": positive, "negative": negative}
    targets_by_role = {role: labels[indices] for role, indices in role_indices.items()}
    weak_by_role = {role: weak_mask[indices] for role, indices in role_indices.items()}
    positive_labels = labels[development["positive_base_indices"]]
    negative_labels = labels[development["hard_negative_base_indices"]]
    if not np.array_equal(
        np.delete(positive_labels, SAW_INDEX, axis=1),
        np.delete(negative_labels, SAW_INDEX, axis=1),
    ):
        raise ValueError("Development hard-negative combinations are not aligned")
    seed_root = args.output_root / "seed_{}".format(seed)
    rows = []
    for variant, alpha in VARIANTS.items():
        output_dir = seed_root / variant
        logits_by_role = {
            role: _fuse(group["base_logits"], group["expert_cal"], alpha)
            for role, group in ordinary.items()
        }
        ordinary_metrics = _evaluate_variant(
            variant,
            logits_by_role,
            targets_by_role,
            weak_by_role,
            output_dir,
            args,
        )
        calibration = _read(output_dir / "calibration.json")
        threshold_payload = _read(output_dir / "thresholds.json")
        saw_threshold = float(
            threshold_payload["modes"]["weak_recall_fpr"][SAW_INDEX]
        )
        power_metrics = {}
        for level, groups in cross_power.items():
            positive_logits = _fuse(
                groups["positive"]["base_logits"],
                groups["positive"]["expert_cal"],
                alpha,
            )
            negative_logits = _fuse(
                groups["negative"]["base_logits"],
                groups["negative"]["expert_cal"],
                alpha,
            )
            metrics = _power_metrics(
                positive_logits,
                negative_logits,
                positive_labels,
                level,
                calibration["temperature"],
                saw_threshold,
                args.min_stratum_pairs,
            )
            power_metrics[str(level)] = metrics
            level_dir = output_dir / "power_{}db".format(level)
            level_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(level_dir / "predictions.npz"),
                positive_logits=positive_logits,
                negative_logits=negative_logits,
                positive_targets=positive_labels,
                negative_targets=negative_labels,
                positive_identities=development["positive_identities"],
                negative_identities=development["hard_negative_identities"],
            )
            _write(level_dir / "metrics.json", metrics)
        aurocs = np.asarray(
            [power_metrics[str(level)]["auroc"] for level in args.levels],
            dtype=np.float64,
        )
        row = dict(ordinary_metrics)
        row.update(
            {
                "seed": int(seed),
                "alpha": float(alpha),
                "cross_power": power_metrics,
                "cross_power_auroc_mean": float(aurocs.mean()),
                "cross_power_auroc_std": float(aurocs.std(ddof=1)),
                "cross_power_auroc_worst": float(aurocs.min()),
                "platt_coefficient": coefficient,
                "platt_intercept": intercept,
                "split_sha256": plan["split_sha256"],
                "development_selection_sha256": development["selection_sha256"],
                "f2_checkpoint_sha256": _sha256(f2_checkpoint_path),
                "h2_checkpoint_sha256": _sha256(h2_checkpoint_path),
                "expert_checkpoint_sha256": _sha256(expert_path),
                "router_parameter_source_sha256": _sha256(router_path),
                "identity_overlap": audit["expert_development_identity_overlap"],
                "locked_test_used": False,
            }
        )
        _write(output_dir / "variant_summary.json", row)
        _write(
            output_dir / "run_receipt.json",
            {
                "status": "completed",
                "seed": int(seed),
                "variant": variant,
                "alpha": float(alpha),
                "split_sha256": plan["split_sha256"],
                "development_selection_sha256": development["selection_sha256"],
                "identity_overlap": audit["expert_development_identity_overlap"],
                "locked_test_used": False,
            },
        )
        rows.append(row)
    audit.update(
        {
            "seed": int(seed),
            "levels": [int(level) for level in args.levels],
            "positive_identity_count": int(len(development["positive_identities"])),
            "negative_identity_count": int(len(development["hard_negative_identities"])),
            "split_sha256": plan["split_sha256"],
            "locked_test_used": False,
        }
    )
    _write(seed_root / "cross_power_audit.json", audit)
    return rows, audit


def _aggregate(rows, levels):
    metrics = (
        "macro_aupr",
        "macro_f1",
        "hamming_loss",
        "saw_aupr",
        "saw_negative_fpr",
        "cross_power_auroc_mean",
        "cross_power_auroc_std",
        "cross_power_auroc_worst",
    )
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                231000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        aggregate[variant]["by_level"] = {
            str(level): {
                metric: _mean_ci(
                    [row["cross_power"][str(level)][metric] for row in selected],
                    232000 + variant_index * 1000 + int(level) * 10 + metric_index,
                )
                for metric_index, metric in enumerate(
                    ("auroc", "aupr", "positive_recall", "hard_negative_fpr")
                )
            }
            for level in levels
        }
        level_means = np.asarray(
            [aggregate[variant]["by_level"][str(level)]["auroc"]["mean"] for level in levels]
        )
        aggregate[variant]["level_mean_auroc_std"] = float(level_means.std(ddof=1))
        aggregate[variant]["level_mean_auroc_worst"] = float(level_means.min())
    return aggregate


def _comparison(rows, candidate, baseline, levels):
    per_seed_level = []
    for seed in sorted({row["seed"] for row in rows}):
        left = next(row for row in rows if row["seed"] == seed and row["variant"] == candidate)
        right = next(row for row in rows if row["seed"] == seed and row["variant"] == baseline)
        for level in levels:
            left_metrics = left["cross_power"][str(level)]
            right_metrics = right["cross_power"][str(level)]
            per_seed_level.append(
                {
                    "seed": int(seed),
                    "level": int(level),
                    "auroc_delta": left_metrics["auroc"] - right_metrics["auroc"],
                    "aupr_delta": left_metrics["aupr"] - right_metrics["aupr"],
                    "positive_recall_delta": left_metrics["positive_recall"] - right_metrics["positive_recall"],
                    "hard_negative_fpr_delta": left_metrics["hard_negative_fpr"] - right_metrics["hard_negative_fpr"],
                }
            )
    ordinary = []
    for seed in sorted({row["seed"] for row in rows}):
        left = next(row for row in rows if row["seed"] == seed and row["variant"] == candidate)
        right = next(row for row in rows if row["seed"] == seed and row["variant"] == baseline)
        ordinary.append(
            {
                "seed": int(seed),
                "macro_aupr_delta": left["macro_aupr"] - right["macro_aupr"],
                "saw_aupr_delta": left["saw_aupr"] - right["saw_aupr"],
                "saw_negative_fpr_delta": left["saw_negative_fpr"] - right["saw_negative_fpr"],
            }
        )
    by_level = {}
    for level in levels:
        selected = [row for row in per_seed_level if row["level"] == int(level)]
        by_level[str(level)] = {
            "auroc_delta": _mean_ci([row["auroc_delta"] for row in selected], 234000 + int(level)),
            "aupr_delta": _mean_ci([row["aupr_delta"] for row in selected], 234100 + int(level)),
            "positive_seed_count": int(sum(row["auroc_delta"] > 0.0 for row in selected)),
        }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "per_seed_level": per_seed_level,
        "by_level": by_level,
        "cross_power_auroc_delta": _mean_ci(
            [row["auroc_delta"] for row in per_seed_level], 235000
        ),
        "cross_power_aupr_delta": _mean_ci(
            [row["aupr_delta"] for row in per_seed_level], 235001
        ),
        "ordinary_per_seed": ordinary,
        "macro_aupr_delta": _mean_ci([row["macro_aupr_delta"] for row in ordinary], 235100),
        "saw_aupr_delta": _mean_ci([row["saw_aupr_delta"] for row in ordinary], 235101),
        "saw_negative_fpr_delta": _mean_ci([row["saw_negative_fpr_delta"] for row in ordinary], 235102),
    }


def _stratum_comparison(rows, candidate, baseline, levels):
    comparisons = []
    for seed in sorted({row["seed"] for row in rows}):
        left = next(row for row in rows if row["seed"] == seed and row["variant"] == candidate)
        right = next(row for row in rows if row["seed"] == seed and row["variant"] == baseline)
        for level in levels:
            baseline_lookup = {
                (row["stratum_type"], row["stratum_value"]): row
                for row in right["cross_power"][str(level)]["strata"]
            }
            for row in left["cross_power"][str(level)]["strata"]:
                key = (row["stratum_type"], row["stratum_value"])
                if key not in baseline_lookup:
                    continue
                comparisons.append(
                    {
                        "seed": int(seed),
                        "level": int(level),
                        "stratum_type": key[0],
                        "stratum_value": key[1],
                        "pair_count": row["pair_count"],
                        "auroc_delta": row["auroc"] - baseline_lookup[key]["auroc"],
                        "aupr_delta": row["aupr"] - baseline_lookup[key]["aupr"],
                    }
                )
    return {
        "rows": comparisons,
        "count": int(len(comparisons)),
        "noninferior_count": int(sum(row["auroc_delta"] >= 0.0 for row in comparisons)),
        "noninferior_fraction": float(
            sum(row["auroc_delta"] >= 0.0 for row in comparisons) / max(len(comparisons), 1)
        ),
    }


def run(args):
    config = load_config(args.config)
    base_path = _data_path(config)
    labels = select_labels(read_labels(base_path), "six")
    identities, identity_labels = load_level_identity(base_path)
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base identity labels are misaligned")
    parameters, _ = read_parameters(base_path)
    weak_mask = compute_weak_component_mask(
        labels,
        _component_power(parameters, list(CLASS_NAMES)),
        args.weak_threshold_db,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    audits = []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(
            args, config, labels, weak_mask, identities, int(seed), device
        )
        rows.extend(seed_rows)
        audits.append(audit)
    aggregate = _aggregate(rows, args.levels)
    comparisons = {
        variant + "_vs_C0_H2": _comparison(rows, variant, "C0_H2", args.levels)
        for variant in VARIANTS
        if variant != "C0_H2"
    }
    candidate_comparison = comparisons["C2_A080_vs_C0_H2"]
    stratum_comparison = _stratum_comparison(rows, "C2_A080", "C0_H2", args.levels)
    level_positive_means = sum(
        candidate_comparison["by_level"][str(level)]["auroc_delta"]["mean"] > 0.0
        for level in args.levels
    )
    every_level_two_seeds = all(
        candidate_comparison["by_level"][str(level)]["positive_seed_count"] >= 2
        for level in args.levels
    )
    candidate_aggregate = aggregate["C2_A080"]
    checks = {
        "three_of_four_level_means_positive": level_positive_means >= 3,
        "every_level_two_positive_seeds": every_level_two_seeds,
        "mean_auroc_gain_0p15": candidate_comparison["cross_power_auroc_delta"]["mean"] >= 0.15,
        "mean_cross_power_auroc_0p65": candidate_aggregate["cross_power_auroc_mean"]["mean"] >= 0.65,
        "worst_level_auroc_0p58": candidate_aggregate["level_mean_auroc_worst"] >= 0.58,
        "level_mean_auroc_std_0p08": candidate_aggregate["level_mean_auroc_std"] <= 0.08,
        "macro_aupr_delta_ge_minus_0p003": candidate_comparison["macro_aupr_delta"]["mean"] >= -0.003,
        "saw_aupr_delta_ge_minus_0p005": candidate_comparison["saw_aupr_delta"]["mean"] >= -0.005,
        "saw_fpr_delta_le_0p02": candidate_comparison["saw_negative_fpr_delta"]["mean"] <= 0.02,
        "strata_noninferior_fraction_0p75": stratum_comparison["noninferior_fraction"] >= 0.75,
        "identity_and_locked_test_audit": all(
            audit["expert_development_identity_overlap"] == 0
            and not audit["locked_test_used"]
            for audit in audits
        ),
    }
    gate = {"checks": checks, "passed": all(checks.values())}
    payload = {
        "status": "completed",
        "device": str(device),
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds),
            "levels": list(args.levels),
            "variants": VARIANTS,
            "promotion_candidate": "C2_A080",
            "min_stratum_pairs": args.min_stratum_pairs,
            "max_negative_fpr": args.max_negative_fpr,
            "weak_threshold_db": args.weak_threshold_db,
        },
        "audits": audits,
        "per_run": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "candidate_stratum_comparison": stratum_comparison,
        "promotion_gate": gate,
        "next_decision": (
            "run_last_stage_small_lr_unfreeze"
            if gate["passed"]
            else "stop_before_unfreeze_and_stabilize_expert"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(
        args.output_root / "route_a_sawfm_fixed_fusion_cross_power_summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--routing-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_expert_routing")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_fixed_fusion_cross_power")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--levels", type=int, nargs="+", default=[35, 40, 45, 50])
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--min-stratum-pairs", type=int, default=4)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
