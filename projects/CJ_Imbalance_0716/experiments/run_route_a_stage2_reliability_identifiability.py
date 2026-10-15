"""Frozen Stage2 reliability-identifiability screen for Route A."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.models.evidence_readout import VisibilityAffineCalibrator  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_sawfm_expert_routing import _load_expert  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import _model_scores  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
import experiments.run_route_a_selective_unknown_decision as selective  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


SAW_INDEX = 1
VARIANTS = (
    "R0_Constant",
    "R1_A0Margin",
    "R2_VisibilityStats",
    "R3_Stage2Temporal",
    "R4_Combined",
)


def _safe_metrics(targets, scores):
    targets = np.asarray(targets, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    prevalence = float(targets.mean()) if len(targets) else float("nan")
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return {
            "count": int(len(targets)), "positive_count": int(targets.sum()),
            "prevalence": prevalence, "auroc": None, "aupr": None,
        }
    return {
        "count": int(len(targets)), "positive_count": int(targets.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(targets, scores)),
        "aupr": float(average_precision_score(targets, scores)),
    }


def _temporal_descriptors(features, evidence_map, bins=25):
    """Create fixed temporal descriptors without fitting or target access."""
    features = np.asarray(features, dtype=np.float32)
    evidence_map = np.asarray(evidence_map, dtype=np.float32)
    if features.ndim != 3 or evidence_map.ndim != 2:
        raise ValueError("Expected Stage2 [N,C,T] and evidence map [N,T]")
    if features.shape[0] != evidence_map.shape[0] or features.shape[2] != evidence_map.shape[1]:
        raise ValueError("Stage2 and evidence-map dimensions do not align")
    if features.shape[2] % int(bins):
        raise ValueError("Temporal length must be divisible by bins")
    traces = np.stack(
        [
            evidence_map,
            np.sqrt(np.mean(np.square(features), axis=1) + 1e-8),
            np.mean(np.abs(features), axis=1),
            np.std(features, axis=1),
        ],
        axis=1,
    )
    n, channels, length = traces.shape
    blocked = traces.reshape(n, channels, int(bins), length // int(bins))
    parts = [
        blocked.mean(axis=3).reshape(n, -1),
        blocked.max(axis=3).reshape(n, -1),
        blocked.std(axis=3).reshape(n, -1),
    ]
    quantiles = np.quantile(
        traces, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], axis=2
    ).transpose(1, 2, 0)
    parts.append(quantiles.reshape(n, -1))
    delta = np.diff(traces, axis=2)
    top_count = max(1, delta.shape[2] // 20)
    delta_top = np.partition(np.abs(delta), -top_count, axis=2)[:, :, -top_count:].mean(axis=2)
    parts.append(
        np.stack(
            [
                np.mean(np.abs(delta), axis=2),
                np.std(delta, axis=2),
                np.max(np.abs(delta), axis=2),
                delta_top,
            ],
            axis=2,
        ).reshape(n, -1)
    )
    autocorrelations = []
    centered = traces - traces.mean(axis=2, keepdims=True)
    scale = np.mean(np.square(centered), axis=2) + 1e-8
    for lag in (1, 4, 16, 32):
        autocorrelations.append(
            (np.mean(centered[:, :, :-lag] * centered[:, :, lag:], axis=2) / scale)
        )
    parts.append(np.stack(autocorrelations, axis=2).reshape(n, -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _domain_class_weights(domains, targets):
    domains = np.asarray(domains)
    targets = np.asarray(targets, dtype=np.uint8)
    weights = np.zeros(len(targets), dtype=np.float64)
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        domain_mask = domains == domain
        present_classes = np.unique(targets[domain_mask])
        for target in present_classes:
            mask = domain_mask & (targets == target)
            weights[mask] = len(targets) / (
                len(unique_domains) * len(present_classes) * max(int(mask.sum()), 1)
            )
    return weights


def _standardize_fit(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def _macro_auroc(targets, scores, domains):
    values = []
    for domain in np.unique(domains):
        metric = _safe_metrics(targets[np.asarray(domains) == domain], scores[np.asarray(domains) == domain])
        if metric["auroc"] is not None:
            values.append(metric["auroc"])
    return None if not values else float(np.mean(values))


def _fit_logistic_variant(fit_x, fit_y, fit_domains, validation_x, validation_y, validation_domains, c_values):
    mean, std = _standardize_fit(fit_x)
    fit_normalized = (fit_x - mean) / std
    validation_normalized = (validation_x - mean) / std
    weights = _domain_class_weights(fit_domains, fit_y)
    candidates = []
    selected = None
    for c_value in c_values:
        model = LogisticRegression(C=float(c_value), max_iter=3000, solver="lbfgs")
        model.fit(fit_normalized, fit_y, sample_weight=weights)
        scores = model.predict_proba(validation_normalized)[:, 1]
        metrics = _safe_metrics(validation_y, scores)
        macro = _macro_auroc(validation_y, scores, validation_domains)
        row = {"c": float(c_value), "metrics": metrics, "macro_domain_auroc": macro}
        candidates.append(row)
        key = (
            -1.0 if macro is None else macro,
            -1.0 if metrics["auroc"] is None else metrics["auroc"],
            -1.0 if metrics["aupr"] is None else metrics["aupr"],
            -float(c_value),
        )
        if selected is None or key > selected[0]:
            selected = (key, model, float(c_value))
    return {
        "model": selected[1], "mean": mean, "std": std,
        "selected_c": selected[2], "candidates": candidates,
    }


def _predict_bundle(model_bundle, values):
    normalized = (np.asarray(values, dtype=np.float64) - model_bundle["mean"]) / model_bundle["std"]
    return model_bundle["model"].predict_proba(normalized)[:, 1]


def _selection_metrics_frozen(targets, frozen_predictions, reliability_score, reliability_threshold):
    targets = np.asarray(targets, dtype=bool)
    predictions = np.asarray(frozen_predictions, dtype=bool)
    score = np.asarray(reliability_score, dtype=np.float64)
    retained = score < float(reliability_threshold)
    errors = predictions[retained] != targets[retained]
    positive_total = max(int(targets.sum()), 1)
    negative_total = max(int((~targets).sum()), 1)
    return {
        "count": int(len(targets)), "retained_count": int(retained.sum()),
        "coverage": float(retained.mean()),
        "selective_risk": float(errors.mean()) if len(errors) else 1.0,
        "confirmed_positive_false_negative_rate": float(((~predictions) & targets & retained).sum() / positive_total),
        "confirmed_negative_false_positive_rate": float((predictions & (~targets) & retained).sum() / negative_total),
    }


def _search_reliability_threshold(groups, scores):
    all_scores = np.concatenate([scores[name] for name in groups])
    candidates = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, 201)))
    best = None
    rows = []
    for threshold in candidates:
        metrics = {
            name: _selection_metrics_frozen(
                groups[name]["targets"], groups[name]["frozen_predictions"], scores[name], threshold
            )
            for name in groups
        }
        power = [metrics[name] for name in metrics if name != "root"]
        feasible = (
            metrics["root"]["coverage"] >= 0.70 - 1e-12
            and float(np.mean([item["coverage"] for item in power])) >= 0.40 - 1e-12
        )
        row = {"threshold": float(threshold), "feasible": bool(feasible), "domains": metrics}
        rows.append(row)
        key = (
            -max(item["selective_risk"] for item in metrics.values()),
            float(np.mean([item["coverage"] for item in metrics.values()])),
            float(threshold),
        )
        if feasible and (best is None or key > best[0]):
            best = (key, row)
    return None if best is None else best[1], rows


def _load_visibility_bundle(args, seed, device):
    checkpoint_path = args.frozen_visibility_root / f"seed_{seed}" / "best_visibility_calibrator.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VisibilityAffineCalibrator(10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (model, checkpoint["feature_mean"], checkpoint["feature_std"]), checkpoint_path


def _payload_a0(group, uniform, visibility_bundle, device):
    return selective._probabilities(group, uniform, visibility_bundle, device)["a0"]


def _make_group(name, group, targets, threshold, expert, uniform, visibility_bundle, device):
    targets = np.asarray(targets, dtype=np.uint8)
    probability = _payload_a0(group, uniform, visibility_bundle, device)
    frozen_predictions = probability >= float(threshold)
    _, evidence_map = _model_scores(expert, group["features"], device, return_maps=True)
    temporal = _temporal_descriptors(group["features"], evidence_map)
    margin = np.stack(
        [probability, probability - float(threshold), np.abs(probability - float(threshold))], axis=1
    ).astype(np.float32)
    return {
        "name": name,
        "targets": targets,
        "a0_probability": probability.astype(np.float32),
        "frozen_predictions": frozen_predictions.astype(np.uint8),
        "decision_error": (frozen_predictions != targets.astype(bool)).astype(np.uint8),
        "margin": margin,
        "visibility": np.concatenate([group["visibility_features"], margin], axis=1).astype(np.float32),
        "temporal": np.concatenate([temporal, margin], axis=1).astype(np.float32),
        "combined": np.concatenate([group["visibility_features"], temporal, margin], axis=1).astype(np.float32),
    }


def _concat_groups(groups, feature_key=None):
    result = {
        "targets": np.concatenate([group["targets"] for group in groups]),
        "decision_error": np.concatenate([group["decision_error"] for group in groups]),
        "domains": np.concatenate([np.full(len(group["targets"]), group["name"], dtype=object) for group in groups]),
    }
    if feature_key is not None:
        result["features"] = np.concatenate([group[feature_key] for group in groups], axis=0)
    return result


def _evaluate_scores(groups, score_by_name):
    targets = np.concatenate([group["decision_error"] for group in groups])
    scores = np.concatenate([score_by_name[group["name"]] for group in groups])
    domains = np.concatenate([np.full(len(group["targets"]), group["name"], dtype=object) for group in groups])
    positive_mask = np.concatenate([group["targets"].astype(bool) for group in groups])
    return {
        "aggregate": _safe_metrics(targets, scores),
        "macro_domain_auroc": _macro_auroc(targets, scores, domains),
        "positive_miss": _safe_metrics(targets[positive_mask], scores[positive_mask]),
        "by_domain": {
            group["name"]: _safe_metrics(group["decision_error"], score_by_name[group["name"]])
            for group in groups
        },
    }


def _group_score(variant, group, model_bundle=None, fit_error_rate=None):
    if variant == "R0_Constant":
        return np.full(len(group["targets"]), float(fit_error_rate), dtype=np.float64)
    if variant == "R1_A0Margin":
        return -group["margin"][:, 2].astype(np.float64)
    key = {
        "R2_VisibilityStats": "visibility",
        "R3_Stage2Temporal": "temporal",
        "R4_Combined": "combined",
    }[variant]
    return _predict_bundle(model_bundle, group[key])


def _run_seed(args, config, labels, identities, seed, device, prior_summary):
    # Reuse the audited extraction path and frozen A0 replay from the previous stage.
    _, source_audit = selective._run_seed(args, config, labels, identities, seed, device, prior_summary)
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    fit_groups_raw = selective._FIT_GROUPS
    threshold_groups_raw = selective._THRESHOLD_GROUPS
    payloads = selective._PAYLOADS
    if fit_groups_raw is None or threshold_groups_raw is None:
        raise RuntimeError("Selective replay did not expose frozen role groups")

    fit_score_groups = [(group["c2"], targets) for group, targets in fit_groups_raw]
    uniform = stable._fit_uniform_platt(fit_score_groups)
    visibility_bundle, visibility_path = _load_visibility_bundle(args, seed, device)
    threshold = next(
        float(row["threshold"]) for row in prior_summary["per_run"]
        if int(row["seed"]) == int(seed) and row["variant"] == "V2_VisibilityAffine"
    )
    expert_path = Path(source_audit["selected_expert_path"])
    expert, _ = _load_expert(expert_path, device)

    fit_names = ["root", "root", "root"]
    for level in args.levels:
        fit_names.extend([str(level), str(level)])
    fit = [
        _make_group(name, group, targets, threshold, expert, uniform, visibility_bundle, device)
        for name, (group, targets) in zip(fit_names, fit_groups_raw)
    ]
    ordinary_validation, ordinary_targets = threshold_groups_raw[0]
    validation = [
        _make_group("root", ordinary_validation, ordinary_targets, threshold, expert, uniform, visibility_bundle, device)
    ]
    for level, positive, negative in threshold_groups_raw[1]:
        validation.extend([
            _make_group(str(level), positive, np.ones(len(positive["c2"]), dtype=np.uint8), threshold, expert, uniform, visibility_bundle, device),
            _make_group(str(level), negative, np.zeros(len(negative["c2"]), dtype=np.uint8), threshold, expert, uniform, visibility_bundle, device),
        ])
    early_targets = labels[plan["early_stop_indices"], SAW_INDEX].astype(np.uint8)
    early = [
        _make_group("root", payloads[("ordinary_early", "root", "mixed")], early_targets, threshold, expert, uniform, visibility_bundle, device)
    ]
    for level in args.levels:
        positive = payloads[("development", str(level), "positive")]
        negative = payloads[("development", str(level), "negative")]
        early.extend([
            _make_group(str(level), positive, np.ones(len(positive["c2"]), dtype=np.uint8), threshold, expert, uniform, visibility_bundle, device),
            _make_group(str(level), negative, np.zeros(len(negative["c2"]), dtype=np.uint8), threshold, expert, uniform, visibility_bundle, device),
        ])

    # Combine positive/negative groups sharing a domain so threshold constraints are domain based.
    def merge_by_domain(groups):
        merged = []
        for name in ["root"] + [str(level) for level in args.levels]:
            selected = [group for group in groups if group["name"] == name]
            value = {"name": name}
            for key in selected[0]:
                if key != "name":
                    value[key] = np.concatenate([group[key] for group in selected], axis=0)
            merged.append(value)
        return merged

    fit_merged = merge_by_domain(fit)
    validation_merged = merge_by_domain(validation)
    early_merged = merge_by_domain(early)
    fit_error_rate = float(np.concatenate([group["decision_error"] for group in fit_merged]).mean())
    model_bundles = {}
    selection_records = {}
    for variant, key in (
        ("R2_VisibilityStats", "visibility"),
        ("R3_Stage2Temporal", "temporal"),
        ("R4_Combined", "combined"),
    ):
        fit_values = _concat_groups(fit_merged, key)
        validation_values = _concat_groups(validation_merged, key)
        bundle = _fit_logistic_variant(
            fit_values["features"], fit_values["decision_error"], fit_values["domains"],
            validation_values["features"], validation_values["decision_error"], validation_values["domains"],
            args.c_values,
        )
        model_bundles[variant] = bundle
        selection_records[variant] = {
            "selected_c": bundle["selected_c"], "candidates": bundle["candidates"],
            "feature_dimension": int(fit_values["features"].shape[1]),
        }

    rows = []
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        bundle = model_bundles.get(variant)
        validation_scores = {
            group["name"]: _group_score(variant, group, bundle, fit_error_rate)
            for group in validation_merged
        }
        early_scores = {
            group["name"]: _group_score(variant, group, bundle, fit_error_rate)
            for group in early_merged
        }
        validation_metrics = _evaluate_scores(validation_merged, validation_scores)
        early_metrics = _evaluate_scores(early_merged, early_scores)
        selected_threshold, threshold_rows = _search_reliability_threshold(
            {group["name"]: group for group in validation_merged}, validation_scores
        )
        if selected_threshold is None:
            selective_metrics = None
        else:
            selected_value = selected_threshold["threshold"]
            selective_metrics = {
                group["name"]: _selection_metrics_frozen(
                    group["targets"], group["frozen_predictions"], early_scores[group["name"]], selected_value
                )
                for group in early_merged
            }
        validation_auroc = validation_metrics["aggregate"]["auroc"]
        early_auroc = early_metrics["aggregate"]["auroc"]
        row = {
            "seed": int(seed), "variant": variant,
            "fit_error_rate": fit_error_rate,
            "validation": validation_metrics, "early": early_metrics,
            "validation_early_auroc_gap": None if validation_auroc is None or early_auroc is None else float(validation_auroc - early_auroc),
            "reliability_threshold_feasible": selected_threshold is not None,
            "selected_reliability_threshold": None if selected_threshold is None else float(selected_threshold["threshold"]),
            "early_selective": selective_metrics,
            "locked_test_used": False,
        }
        rows.append(row)
        variant_root = seed_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        _write(variant_root / "metrics.json", row)
        _write(variant_root / "threshold_search.json", {"selected": selected_threshold, "rows": threshold_rows})
        np.savez_compressed(
            str(variant_root / "early_predictions.npz"),
            **{f"{name}_error_score": values for name, values in early_scores.items()},
            **{f"{group['name']}_target": group["targets"] for group in early_merged},
            **{f"{group['name']}_a0_probability": group["a0_probability"] for group in early_merged},
        )
        _write(variant_root / "run_receipt.json", {
            "status": "completed", "seed": int(seed), "variant": variant,
            "split_sha256": plan["split_sha256"],
            "f2_checkpoint_sha256": source_audit["f2_checkpoint_sha256"],
            "h2_checkpoint_sha256": source_audit["h2_checkpoint_sha256"],
            "selected_expert_sha256": source_audit["selected_expert_sha256"],
            "frozen_visibility_sha256": _sha256(visibility_path),
            "identity_overlap": source_audit["identity_overlap"], "locked_test_used": False,
        })
    _write(seed_root / "probe_selection.json", selection_records)
    _write(seed_root / "source_and_identity_audit.json", source_audit)
    return rows, source_audit


def _aggregate(rows):
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {}
        for role in ("validation", "early"):
            result[variant][role] = {
                metric: _mean_ci(
                    [row[role]["aggregate"][metric] for row in selected],
                    391000 + variant_index * 100 + (0 if role == "validation" else 20) + metric_index,
                )
                for metric_index, metric in enumerate(("auroc", "aupr", "prevalence"))
            }
            result[variant][role]["macro_domain_auroc"] = _mean_ci(
                [row[role]["macro_domain_auroc"] for row in selected],
                392000 + variant_index * 10 + (0 if role == "validation" else 1),
            )
        result[variant]["validation_early_auroc_gap"] = _mean_ci(
            [row["validation_early_auroc_gap"] for row in selected], 393000 + variant_index
        )
        result[variant]["feasible_seed_count"] = int(sum(row["reliability_threshold_feasible"] for row in selected))
    return result


def _promotion_gate(rows, audits, levels):
    r2 = {row["seed"]: row for row in rows if row["variant"] == "R2_VisibilityStats"}
    r4 = {row["seed"]: row for row in rows if row["variant"] == "R4_Combined"}
    per_seed = []
    for seed, row in sorted(r4.items()):
        selective_metrics = row["early_selective"] or {}
        ordinary = selective_metrics.get("root", {})
        power = [selective_metrics.get(str(level), {}) for level in levels]
        early_auroc = row["early"]["aggregate"]["auroc"]
        r2_auroc = r2[seed]["early"]["aggregate"]["auroc"]
        gap = row["validation_early_auroc_gap"]
        max_risk = max([ordinary.get("selective_risk", 1.0)] + [item.get("selective_risk", 1.0) for item in power])
        mean_power_coverage = float(np.mean([item.get("coverage", 0.0) for item in power]))
        checks = {
            "early_auroc_0p65": early_auroc is not None and early_auroc >= 0.65,
            "r4_minus_r2_auroc_0p03": early_auroc is not None and r2_auroc is not None and early_auroc >= r2_auroc + 0.03,
            "validation_early_gap_0p15": gap is not None and gap <= 0.15,
            "threshold_feasible": row["reliability_threshold_feasible"],
            "ordinary_coverage_0p70": ordinary.get("coverage", 0.0) >= 0.70,
            "mean_power_coverage_0p40": mean_power_coverage >= 0.40,
            "max_selective_risk_0p15": max_risk <= 0.15,
        }
        per_seed.append({
            "seed": int(seed), "checks": checks, "passed": all(checks.values()),
            "r4_early_auroc": early_auroc, "r2_early_auroc": r2_auroc,
            "mean_power_coverage": mean_power_coverage, "max_selective_risk": max_risk,
        })
    audit_pass = all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits)
    passed_seed_count = int(sum(item["passed"] for item in per_seed))
    return {
        "per_seed": per_seed, "passed_seed_count": passed_seed_count,
        "two_of_three_seed_pass": passed_seed_count >= 2,
        "identity_and_locked_test_audit": audit_pass,
        "passed": passed_seed_count >= 2 and audit_pass,
    }


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base labels and identities are misaligned")
    prior_summary = _read(args.prior_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits = [], []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(args, config, labels, identities, int(seed), device, prior_summary)
        rows.extend(seed_rows)
        audits.append(audit)
    aggregate = _aggregate(rows)
    gate = _promotion_gate(rows, audits, args.levels)
    r4 = aggregate["R4_Combined"]
    temporal_independent_signal = (
        r4["early"]["auroc"]["mean"] >= 0.65
        and r4["early"]["auroc"]["mean"] >= aggregate["R2_VisibilityStats"]["early"]["auroc"]["mean"] + 0.03
        and r4["validation_early_auroc_gap"]["mean"] <= 0.15
    )
    if gate["passed"]:
        next_decision = "train_exist_reliability_dual_head_without_backbone_unfreeze"
    elif temporal_independent_signal:
        next_decision = "try_tiny_temporal_reliability_head_keep_backbone_frozen"
    else:
        next_decision = "stop_stage2_reliability_extension_keep_c2_ranking_h2_decision"
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "levels": list(args.levels), "variants": list(VARIANTS),
            "c_values": list(args.c_values), "frozen_sources": True,
            "reliability_target": "frozen_A0_decision_error",
        },
        "audits": audits, "per_run": rows, "aggregate": aggregate,
        "promotion_gate": gate,
        "temporal_independent_signal": bool(temporal_independent_signal),
        "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "route_a_stage2_reliability_identifiability_summary.json"
    _write(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--stability-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_s2_initialization_stability")
    parser.add_argument("--frozen-visibility-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration")
    parser.add_argument("--prior-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration" / "route_a_stable_expert_visibility_calibration_summary.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stage2_reliability_identifiability")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    # Arguments consumed by the frozen selective replay.
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-positive-fpr", type=float, default=0.20)
    parser.add_argument("--max-positive-fn", type=float, default=0.10)
    parser.add_argument("--min-ordinary-coverage", type=float, default=0.70)
    parser.add_argument("--min-power-coverage", type=float, default=0.40)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
