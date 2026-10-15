"""Frozen selective positive/negative/unknown decision study for Route A."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.models.evidence_readout import VisibilityAffineCalibrator  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_ordinal_visibility_anchor import _tag_sequence  # noqa: E402
from experiments.run_route_a_stable_expert_visibility_calibration import (  # noqa: E402
    SAW_INDEX,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


VARIANTS = (
    "U0_A0SingleThreshold",
    "U1_C2DualThreshold",
    "U2_A0Selective",
    "U3_OraclePowerSelective",
)
_TAG_QUEUE = []
_PAYLOADS = {}
_FIT_GROUPS = None
_THRESHOLD_GROUPS = None


def _selective_metrics(positive_probability, negative_probability, t_negative, t_positive):
    positive_probability = np.asarray(positive_probability, dtype=np.float64)
    negative_probability = np.asarray(negative_probability, dtype=np.float64)
    t_negative = float(t_negative)
    t_positive = float(t_positive)
    if not 0.0 <= t_negative < t_positive <= 1.0:
        raise ValueError("dual thresholds must satisfy 0 <= t_negative < t_positive <= 1")
    positive_as_positive = positive_probability >= t_positive
    positive_as_negative = positive_probability <= t_negative
    negative_as_positive = negative_probability >= t_positive
    negative_as_negative = negative_probability <= t_negative
    positive_unknown = ~(positive_as_positive | positive_as_negative)
    negative_unknown = ~(negative_as_positive | negative_as_negative)
    decided = int(
        positive_as_positive.sum()
        + positive_as_negative.sum()
        + negative_as_positive.sum()
        + negative_as_negative.sum()
    )
    errors = int(positive_as_negative.sum() + negative_as_positive.sum())
    unknown_count = int(positive_unknown.sum() + negative_unknown.sum())
    total = int(len(positive_probability) + len(negative_probability))
    return {
        "t_negative": t_negative,
        "t_positive": t_positive,
        "positive_count": int(len(positive_probability)),
        "negative_count": int(len(negative_probability)),
        "confirmed_positive_recall": float(positive_as_positive.mean()),
        "confirmed_positive_fpr": float(negative_as_positive.mean()),
        "positive_false_negative_rate": float(positive_as_negative.mean()),
        "recoverable_positive_coverage": float(1.0 - positive_as_negative.mean()),
        "unknown_coverage": float(unknown_count / max(total, 1)),
        "unknown_positive_rate": float(positive_unknown.sum() / max(unknown_count, 1)),
        "decision_coverage": float(decided / max(total, 1)),
        "selective_risk": float(errors / max(decided, 1)),
        "positive_unknown_rate": float(positive_unknown.mean()),
        "negative_unknown_rate": float(negative_unknown.mean()),
    }


def _single_threshold_metrics(positive_probability, negative_probability, threshold):
    positive_probability = np.asarray(positive_probability, dtype=np.float64)
    negative_probability = np.asarray(negative_probability, dtype=np.float64)
    threshold = float(threshold)
    positive_as_positive = positive_probability >= threshold
    negative_as_positive = negative_probability >= threshold
    errors = int((~positive_as_positive).sum() + negative_as_positive.sum())
    total = int(len(positive_probability) + len(negative_probability))
    return {
        "threshold": threshold,
        "positive_count": int(len(positive_probability)),
        "negative_count": int(len(negative_probability)),
        "confirmed_positive_recall": float(positive_as_positive.mean()),
        "confirmed_positive_fpr": float(negative_as_positive.mean()),
        "positive_false_negative_rate": float((~positive_as_positive).mean()),
        "recoverable_positive_coverage": float(positive_as_positive.mean()),
        "unknown_coverage": 0.0,
        "unknown_positive_rate": 0.0,
        "decision_coverage": 1.0,
        "selective_risk": float(errors / max(total, 1)),
        "positive_unknown_rate": 0.0,
        "negative_unknown_rate": 0.0,
    }


def _mixed_selective_metrics(probability, targets, t_negative, t_positive):
    targets = np.asarray(targets, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    return _selective_metrics(
        probability[targets], probability[~targets], t_negative, t_positive
    )


def _mixed_single_metrics(probability, targets, threshold):
    targets = np.asarray(targets, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    return _single_threshold_metrics(
        probability[targets], probability[~targets], threshold
    )


def _grid_values(step):
    values = np.arange(float(step), 1.0, float(step), dtype=np.float64)
    return np.round(values, 10)


def _search_unified_thresholds(
    ordinary_probability,
    ordinary_targets,
    power_probability,
    step,
    max_fpr,
    max_positive_fn,
    min_ordinary_coverage,
    min_power_coverage,
):
    rows = []
    best = None
    values = _grid_values(step)
    for t_negative in values[:-1]:
        for t_positive in values[values > t_negative]:
            ordinary = _mixed_selective_metrics(
                ordinary_probability, ordinary_targets, t_negative, t_positive
            )
            power = {
                str(level): _selective_metrics(
                    payload["positive"], payload["negative"], t_negative, t_positive
                )
                for level, payload in power_probability.items()
            }
            power_values = list(power.values())
            feasible = (
                ordinary["confirmed_positive_fpr"] <= max_fpr + 1e-12
                and all(item["confirmed_positive_fpr"] <= max_fpr + 1e-12 for item in power_values)
                and all(item["positive_false_negative_rate"] <= max_positive_fn + 1e-12 for item in power_values)
                and ordinary["decision_coverage"] >= min_ordinary_coverage - 1e-12
                and float(np.mean([item["decision_coverage"] for item in power_values])) >= min_power_coverage - 1e-12
            )
            row = {
                "t_negative": float(t_negative),
                "t_positive": float(t_positive),
                "feasible": bool(feasible),
                "ordinary": ordinary,
                "power": power,
            }
            rows.append(row)
            key = (
                min(item["decision_coverage"] for item in power_values),
                float(np.mean([item["decision_coverage"] for item in power_values])),
                float(np.mean([item["confirmed_positive_recall"] for item in power_values])),
                ordinary["confirmed_positive_recall"],
                -max([ordinary["selective_risk"]] + [item["selective_risk"] for item in power_values]),
                -float(t_positive - t_negative),
            )
            if feasible and (best is None or key > best[0]):
                best = (key, row)
    return (None if best is None else best[1]), rows


def _search_single_domain(
    positive_probability,
    negative_probability,
    step,
    max_fpr,
    max_positive_fn,
    min_coverage,
):
    rows = []
    best = None
    values = _grid_values(step)
    for t_negative in values[:-1]:
        for t_positive in values[values > t_negative]:
            metrics = _selective_metrics(
                positive_probability, negative_probability, t_negative, t_positive
            )
            feasible = (
                metrics["confirmed_positive_fpr"] <= max_fpr + 1e-12
                and metrics["positive_false_negative_rate"] <= max_positive_fn + 1e-12
                and metrics["decision_coverage"] >= min_coverage - 1e-12
            )
            row = {"feasible": bool(feasible), **metrics}
            rows.append(row)
            key = (
                metrics["decision_coverage"],
                metrics["confirmed_positive_recall"],
                -metrics["selective_risk"],
                -float(t_positive - t_negative),
            )
            if feasible and (best is None or key > best[0]):
                best = (key, row)
    return (None if best is None else best[1]), rows


def _wrap_add_c2(original):
    def tagged(group, coefficient, intercept):
        result = original(group, coefficient, intercept)
        if not _TAG_QUEUE:
            raise RuntimeError("Selective payload tag queue exhausted")
        role, level, target = _TAG_QUEUE.pop(0)
        _PAYLOADS[(role, level, target)] = result
        return result
    return tagged


def _load_frozen_visibility(args, fit_groups, threshold_groups, seed, device):
    global _FIT_GROUPS, _THRESHOLD_GROUPS
    _FIT_GROUPS = fit_groups
    _THRESHOLD_GROUPS = threshold_groups
    source = args.frozen_visibility_root / "seed_{}".format(seed) / "best_visibility_calibrator.pth"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = VisibilityAffineCalibrator(feature_dim=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    thresholds = _read(
        args.frozen_visibility_root / "seed_{}".format(seed) / "threshold_searches.json"
    )["thresholds"]
    history = [{
        "epoch": int(checkpoint["best_epoch"]),
        "loss": 0.0,
        "classification_loss": 0.0,
        "threshold": float(thresholds["V2_VisibilityAffine"]),
        "mean_power_recall": 0.0,
        "min_power_recall": 0.0,
        "ordinary_f1": 0.0,
        "parameter_norm": float(sum(parameter.pow(2).sum() for parameter in model.parameters())),
    }]
    return (
        model,
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
        float(thresholds["V2_VisibilityAffine"]),
        int(checkpoint["best_epoch"]),
        history,
    )


def _probabilities(payload, uniform, visibility_bundle, device):
    return {
        "c2": stable._calibrator_probability(
            "V0_UniformPlatt", payload, uniform, None, visibility_bundle, device
        ),
        "a0": stable._calibrator_probability(
            "V2_VisibilityAffine", payload, None, None, visibility_bundle, device
        ),
    }


def _decisions(probability, t_negative, t_positive):
    probability = np.asarray(probability, dtype=np.float64)
    result = np.zeros(len(probability), dtype=np.int8)
    result[probability <= float(t_negative)] = -1
    result[probability >= float(t_positive)] = 1
    return result


def _single_decisions(probability, threshold):
    return np.where(np.asarray(probability) >= float(threshold), 1, -1).astype(np.int8)


def _evaluate_variant(variant, early, thresholds, levels):
    if variant == "U0_A0SingleThreshold":
        ordinary = _mixed_single_metrics(
            early["ordinary"]["a0"], early["ordinary_targets"], thresholds["single"]
        )
        power = {
            str(level): _single_threshold_metrics(
                early["power"][str(level)]["a0_positive"],
                early["power"][str(level)]["a0_negative"],
                thresholds["single"],
            )
            for level in levels
        }
        return ordinary, power
    source = "c2" if variant == "U1_C2DualThreshold" else "a0"
    if variant == "U3_OraclePowerSelective":
        ordinary_threshold = thresholds["ordinary"]
        ordinary = _mixed_selective_metrics(
            early["ordinary"][source], early["ordinary_targets"],
            ordinary_threshold["t_negative"], ordinary_threshold["t_positive"],
        )
        power = {}
        for level in levels:
            selected = thresholds["power"][str(level)]
            power[str(level)] = _selective_metrics(
                early["power"][str(level)][source + "_positive"],
                early["power"][str(level)][source + "_negative"],
                selected["t_negative"], selected["t_positive"],
            )
        return ordinary, power
    selected = thresholds["unified"]
    ordinary = _mixed_selective_metrics(
        early["ordinary"][source], early["ordinary_targets"],
        selected["t_negative"], selected["t_positive"],
    )
    power = {
        str(level): _selective_metrics(
            early["power"][str(level)][source + "_positive"],
            early["power"][str(level)][source + "_negative"],
            selected["t_negative"], selected["t_positive"],
        )
        for level in levels
    }
    return ordinary, power


def _prediction_payload(variant, early, thresholds, levels):
    source = "c2" if variant == "U1_C2DualThreshold" else "a0"
    result = {
        "ordinary_probability": early["ordinary"][source],
        "ordinary_target": early["ordinary_targets"],
    }
    if variant == "U0_A0SingleThreshold":
        result["ordinary_decision"] = _single_decisions(
            early["ordinary"]["a0"], thresholds["single"]
        )
    elif variant == "U3_OraclePowerSelective":
        selected = thresholds["ordinary"]
        result["ordinary_decision"] = _decisions(
            early["ordinary"][source], selected["t_negative"], selected["t_positive"]
        )
    else:
        selected = thresholds["unified"]
        result["ordinary_decision"] = _decisions(
            early["ordinary"][source], selected["t_negative"], selected["t_positive"]
        )
    for level in levels:
        positive = early["power"][str(level)][source + "_positive"]
        negative = early["power"][str(level)][source + "_negative"]
        if variant == "U0_A0SingleThreshold":
            positive_decision = _single_decisions(positive, thresholds["single"])
            negative_decision = _single_decisions(negative, thresholds["single"])
        else:
            selected = thresholds["power"][str(level)] if variant == "U3_OraclePowerSelective" else thresholds["unified"]
            positive_decision = _decisions(positive, selected["t_negative"], selected["t_positive"])
            negative_decision = _decisions(negative, selected["t_negative"], selected["t_positive"])
        result["positive_{}db".format(level)] = positive
        result["negative_{}db".format(level)] = negative
        result["positive_decision_{}db".format(level)] = positive_decision
        result["negative_decision_{}db".format(level)] = negative_decision
    return result


def _run_seed(args, config, labels, identities, seed, device, prior_summary):
    global _TAG_QUEUE, _PAYLOADS, _FIT_GROUPS, _THRESHOLD_GROUPS
    _TAG_QUEUE = _tag_sequence(args.levels)
    _PAYLOADS = {}
    _FIT_GROUPS = None
    _THRESHOLD_GROUPS = None
    replay_root = args.output_root / "source_replay" / "seed_{}".format(seed)
    replay_args = copy.copy(args)
    replay_args.output_root = args.output_root / "source_replay"
    original_add = stable._add_c2_features
    original_train = stable._train_visibility
    try:
        stable._add_c2_features = _wrap_add_c2(original_add)
        stable._train_visibility = _load_frozen_visibility
        _, audit, selection = stable._run_seed(
            replay_args, config, labels, identities, seed, device
        )
    finally:
        stable._add_c2_features = original_add
        stable._train_visibility = original_train
    if _TAG_QUEUE:
        raise RuntimeError("Unused selective payload tags: {}".format(len(_TAG_QUEUE)))
    if _FIT_GROUPS is None or _THRESHOLD_GROUPS is None:
        raise RuntimeError("Frozen threshold roles were not captured")
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    fit_score_groups = [(group["c2"], targets) for group, targets in _FIT_GROUPS]
    uniform = stable._fit_uniform_platt(fit_score_groups)
    frozen_checkpoint = torch.load(
        args.frozen_visibility_root / "seed_{}".format(seed) / "best_visibility_calibrator.pth",
        map_location="cpu", weights_only=False,
    )
    visibility_model = VisibilityAffineCalibrator(10).to(device)
    visibility_model.load_state_dict(frozen_checkpoint["model_state_dict"])
    visibility_model.eval()
    visibility_bundle = (
        visibility_model,
        frozen_checkpoint["feature_mean"],
        frozen_checkpoint["feature_std"],
    )
    ordinary_threshold_group, ordinary_threshold_targets = _THRESHOLD_GROUPS[0]
    threshold = {
        "ordinary": _probabilities(ordinary_threshold_group, uniform, visibility_bundle, device),
        "ordinary_targets": np.asarray(ordinary_threshold_targets, dtype=bool),
        "power": {},
    }
    for level, positive, negative in _THRESHOLD_GROUPS[1]:
        positive_probability = _probabilities(positive, uniform, visibility_bundle, device)
        negative_probability = _probabilities(negative, uniform, visibility_bundle, device)
        threshold["power"][str(level)] = {
            "c2_positive": positive_probability["c2"],
            "c2_negative": negative_probability["c2"],
            "a0_positive": positive_probability["a0"],
            "a0_negative": negative_probability["a0"],
        }
    ordinary_early = _probabilities(
        _PAYLOADS[("ordinary_early", "root", "mixed")], uniform, visibility_bundle, device
    )
    early = {
        "ordinary": ordinary_early,
        "ordinary_targets": labels[plan["early_stop_indices"], SAW_INDEX].astype(bool),
        "power": {},
    }
    for level in args.levels:
        positive_probability = _probabilities(
            _PAYLOADS[("development", str(level), "positive")], uniform, visibility_bundle, device
        )
        negative_probability = _probabilities(
            _PAYLOADS[("development", str(level), "negative")], uniform, visibility_bundle, device
        )
        early["power"][str(level)] = {
            "c2_positive": positive_probability["c2"],
            "c2_negative": negative_probability["c2"],
            "a0_positive": positive_probability["a0"],
            "a0_negative": negative_probability["a0"],
        }
    previous = next(
        row for row in prior_summary["per_run"]
        if row["seed"] == seed and row["variant"] == "V2_VisibilityAffine"
    )
    searches = {}
    thresholds = {
        "U0_A0SingleThreshold": {"single": float(previous["threshold"])}
    }
    for variant, source in (
        ("U1_C2DualThreshold", "c2"),
        ("U2_A0Selective", "a0"),
    ):
        selected, rows = _search_unified_thresholds(
            threshold["ordinary"][source], threshold["ordinary_targets"],
            {
                str(level): {
                    "positive": threshold["power"][str(level)][source + "_positive"],
                    "negative": threshold["power"][str(level)][source + "_negative"],
                }
                for level in args.levels
            },
            args.grid_step, args.max_positive_fpr, args.max_positive_fn,
            args.min_ordinary_coverage, args.min_power_coverage,
        )
        searches[variant] = {"selected": selected, "rows": rows}
        thresholds[variant] = {"unified": selected}
    oracle_thresholds = {"power": {}}
    ordinary_targets = threshold["ordinary_targets"]
    selected, rows = _search_single_domain(
        threshold["ordinary"]["a0"][ordinary_targets],
        threshold["ordinary"]["a0"][~ordinary_targets],
        args.grid_step, args.max_positive_fpr, 1.0, args.min_ordinary_coverage,
    )
    oracle_thresholds["ordinary"] = selected
    searches["U3_OraclePowerSelective"] = {"ordinary": {"selected": selected, "rows": rows}, "power": {}}
    for level in args.levels:
        selected, rows = _search_single_domain(
            threshold["power"][str(level)]["a0_positive"],
            threshold["power"][str(level)]["a0_negative"],
            args.grid_step, args.max_positive_fpr, args.max_positive_fn,
            args.min_power_coverage,
        )
        oracle_thresholds["power"][str(level)] = selected
        searches["U3_OraclePowerSelective"]["power"][str(level)] = {"selected": selected, "rows": rows}
    thresholds["U3_OraclePowerSelective"] = oracle_thresholds
    seed_root = args.output_root / "seed_{}".format(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    _write(seed_root / "threshold_search.json", searches)
    source_audit = {
        **audit,
        "frozen_visibility_checkpoint": str(args.frozen_visibility_root / "seed_{}".format(seed) / "best_visibility_calibrator.pth"),
        "frozen_visibility_sha256": _sha256(args.frozen_visibility_root / "seed_{}".format(seed) / "best_visibility_calibrator.pth"),
        "prior_summary_sha256": _sha256(args.prior_summary),
        "source_replay_root": str(replay_root),
        "selection": selection,
        "locked_test_used": False,
    }
    _write(seed_root / "source_and_identity_audit.json", source_audit)
    rows = []
    for variant in VARIANTS:
        selected_thresholds = thresholds[variant]
        feasible = True
        if variant in ("U1_C2DualThreshold", "U2_A0Selective"):
            feasible = selected_thresholds["unified"] is not None
        elif variant == "U3_OraclePowerSelective":
            feasible = selected_thresholds["ordinary"] is not None and all(
                selected_thresholds["power"][str(level)] is not None for level in args.levels
            )
        if feasible:
            ordinary_metrics, power_metrics = _evaluate_variant(
                variant, early, selected_thresholds, args.levels
            )
        else:
            ordinary_metrics = _mixed_selective_metrics(
                early["ordinary"]["a0"], early["ordinary_targets"], 0.0, 1.0
            )
            power_metrics = {
                str(level): _selective_metrics(
                    early["power"][str(level)]["a0_positive"],
                    early["power"][str(level)]["a0_negative"], 0.0, 1.0
                )
                for level in args.levels
            }
        row = {
            "seed": int(seed),
            "variant": variant,
            "threshold_feasible": bool(feasible),
            "thresholds": selected_thresholds,
            "ordinary": ordinary_metrics,
            "power": power_metrics,
            "mean_power_decision_coverage": float(np.mean([power_metrics[str(level)]["decision_coverage"] for level in args.levels])),
            "min_power_decision_coverage": float(np.min([power_metrics[str(level)]["decision_coverage"] for level in args.levels])),
            "mean_power_confirmed_recall": float(np.mean([power_metrics[str(level)]["confirmed_positive_recall"] for level in args.levels])),
            "max_power_fpr": float(np.max([power_metrics[str(level)]["confirmed_positive_fpr"] for level in args.levels])),
            "max_selective_risk": float(np.max([ordinary_metrics["selective_risk"]] + [power_metrics[str(level)]["selective_risk"] for level in args.levels])),
            "split_sha256": plan["split_sha256"],
            "identity_overlap": 0,
            "locked_test_used": False,
        }
        output_dir = seed_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        if feasible:
            np.savez_compressed(
                str(output_dir / "predictions.npz"),
                **_prediction_payload(variant, early, selected_thresholds, args.levels),
            )
        _write(output_dir / "metrics.json", row)
        _write(output_dir / "run_receipt.json", {
            "status": "completed" if feasible else "no_feasible_thresholds",
            "seed": int(seed), "variant": variant,
            "threshold_feasible": bool(feasible),
            "frozen_visibility_sha256": source_audit["frozen_visibility_sha256"],
            "split_sha256": plan["split_sha256"],
            "identity_overlap": 0, "locked_test_used": False,
        })
        rows.append(row)
    return rows, source_audit


def _aggregate(rows, levels):
    metrics = (
        "mean_power_decision_coverage", "min_power_decision_coverage",
        "mean_power_confirmed_recall", "max_power_fpr", "max_selective_risk",
    )
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                321000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        aggregate[variant]["ordinary"] = {
            metric: _mean_ci(
                [row["ordinary"][metric] for row in selected],
                322000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate((
                "confirmed_positive_recall", "confirmed_positive_fpr",
                "positive_false_negative_rate", "recoverable_positive_coverage",
                "unknown_coverage", "decision_coverage", "selective_risk",
            ))
        }
        aggregate[variant]["by_level"] = {
            str(level): {
                metric: _mean_ci(
                    [row["power"][str(level)][metric] for row in selected],
                    323000 + variant_index * 1000 + int(level) * 10 + metric_index,
                )
                for metric_index, metric in enumerate((
                    "confirmed_positive_recall", "confirmed_positive_fpr",
                    "positive_false_negative_rate", "recoverable_positive_coverage",
                    "unknown_coverage", "decision_coverage", "selective_risk",
                ))
            }
            for level in levels
        }
        aggregate[variant]["feasible_seed_count"] = int(sum(row["threshold_feasible"] for row in selected))
    return aggregate


def _promotion_gate(rows, aggregate, audits, levels):
    u0 = {row["seed"]: row for row in rows if row["variant"] == "U0_A0SingleThreshold"}
    u1 = {row["seed"]: row for row in rows if row["variant"] == "U1_C2DualThreshold"}
    u2 = {row["seed"]: row for row in rows if row["variant"] == "U2_A0Selective"}
    u3 = {row["seed"]: row for row in rows if row["variant"] == "U3_OraclePowerSelective"}
    per_seed = []
    for seed, row in sorted(u2.items()):
        level50 = row["power"]["50"]
        passed = (
            row["threshold_feasible"]
            and row["ordinary"]["confirmed_positive_fpr"] <= 0.20 + 1e-12
            and all(row["power"][str(level)]["confirmed_positive_fpr"] <= 0.20 + 1e-12 for level in levels)
            and level50["recoverable_positive_coverage"] >= 0.60
            and level50["positive_false_negative_rate"] <= 0.10 + 1e-12
            and row["ordinary"]["decision_coverage"] >= 0.70
            and row["mean_power_decision_coverage"] >= 0.40
            and level50["decision_coverage"] >= 0.25
            and row["max_selective_risk"] <= 0.15 + 1e-12
            and level50["confirmed_positive_recall"] >= 0.1368
            and row["ordinary"]["confirmed_positive_recall"] >= u0[seed]["ordinary"]["confirmed_positive_recall"] - 0.05 - 1e-12
            and row["mean_power_decision_coverage"] >= u1[seed]["mean_power_decision_coverage"] - 1e-12
        )
        per_seed.append({"seed": int(seed), "passed": bool(passed)})
    oracle_per_seed = []
    for seed, row in sorted(u3.items()):
        passed = (
            row["threshold_feasible"]
            and row["ordinary"]["confirmed_positive_fpr"] <= 0.20 + 1e-12
            and all(row["power"][str(level)]["confirmed_positive_fpr"] <= 0.20 + 1e-12 for level in levels)
            and all(row["power"][str(level)]["positive_false_negative_rate"] <= 0.10 + 1e-12 for level in levels)
            and row["ordinary"]["decision_coverage"] >= 0.70
            and row["mean_power_decision_coverage"] >= 0.40
            and row["max_selective_risk"] <= 0.15 + 1e-12
        )
        oracle_per_seed.append({"seed": int(seed), "passed": bool(passed)})
    u2a = aggregate["U2_A0Selective"]
    u3a = aggregate["U3_OraclePowerSelective"]
    checks = {
        "u2_all_seed_threshold_feasible": u2a["feasible_seed_count"] == len(u2),
        "all_domain_fpr_0p20": u2a["ordinary"]["confirmed_positive_fpr"]["mean"] <= 0.20 and all(u2a["by_level"][str(level)]["confirmed_positive_fpr"]["mean"] <= 0.20 for level in levels),
        "50db_recoverable_coverage_0p60": u2a["by_level"]["50"]["recoverable_positive_coverage"]["mean"] >= 0.60,
        "50db_positive_fn_0p10": u2a["by_level"]["50"]["positive_false_negative_rate"]["mean"] <= 0.10,
        "ordinary_coverage_0p70": u2a["ordinary"]["decision_coverage"]["mean"] >= 0.70,
        "mean_power_coverage_0p40": u2a["mean_power_decision_coverage"]["mean"] >= 0.40,
        "50db_coverage_0p25": u2a["by_level"]["50"]["decision_coverage"]["mean"] >= 0.25,
        "max_selective_risk_0p15": u2a["max_selective_risk"]["mean"] <= 0.15,
        "50db_confirmed_recall_0p1368": u2a["by_level"]["50"]["confirmed_positive_recall"]["mean"] >= 0.1368,
        "ordinary_recall_delta_minus_0p05": u2a["ordinary"]["confirmed_positive_recall"]["mean"] >= aggregate["U0_A0SingleThreshold"]["ordinary"]["confirmed_positive_recall"]["mean"] - 0.05,
        "two_seed_pass": sum(item["passed"] for item in per_seed) >= 2,
        "u2_coverage_not_below_u1": u2a["mean_power_decision_coverage"]["mean"] >= aggregate["U1_C2DualThreshold"]["mean_power_decision_coverage"]["mean"],
        "u3_oracle_mean_coverage_0p60": u3a["mean_power_decision_coverage"]["mean"] >= 0.60,
        "u3_early_all_domain_fpr_0p20": u3a["ordinary"]["confirmed_positive_fpr"]["mean"] <= 0.20 and all(u3a["by_level"][str(level)]["confirmed_positive_fpr"]["mean"] <= 0.20 for level in levels),
        "u3_early_each_power_fn_0p10": all(u3a["by_level"][str(level)]["positive_false_negative_rate"]["mean"] <= 0.10 for level in levels),
        "u3_early_max_selective_risk_0p15": u3a["max_selective_risk"]["mean"] <= 0.15,
        "u3_two_seed_pass": sum(item["passed"] for item in oracle_per_seed) >= 2,
        "identity_checkpoint_locked_test_audit": all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits),
    }
    return {
        "checks": checks,
        "per_seed": per_seed,
        "oracle_per_seed": oracle_per_seed,
        "passed": all(checks.values()),
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
        seed_rows, audit = _run_seed(
            args, config, labels, identities, int(seed), device, prior_summary
        )
        rows.extend(seed_rows)
        audits.append(audit)
    aggregate = _aggregate(rows, args.levels)
    gate = _promotion_gate(rows, aggregate, audits, args.levels)
    u2_feasible = aggregate["U2_A0Selective"]["feasible_seed_count"]
    oracle_selective_passed = sum(
        item["passed"] for item in gate["oracle_per_seed"]
    ) >= 2
    if gate["passed"]:
        next_decision = "expand_selective_unknown_to_five_seeds"
    elif oracle_selective_passed:
        next_decision = "train_stage2_label_conditioned_exist_reliability_head"
    else:
        next_decision = "stop_selective_calibration_keep_c2_ranking_h2_decision"
    payload = {
        "status": "completed",
        "device": str(device),
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "levels": list(args.levels),
            "variants": list(VARIANTS), "grid_step": args.grid_step,
            "max_positive_fpr": args.max_positive_fpr,
            "max_positive_fn": args.max_positive_fn,
            "min_ordinary_coverage": args.min_ordinary_coverage,
            "min_power_coverage": args.min_power_coverage,
            "frozen_sources": True,
        },
        "audits": audits,
        "per_run": rows,
        "aggregate": aggregate,
        "promotion_gate": gate,
        "u2_feasible_seed_count": u2_feasible,
        "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_selective_unknown_decision_summary.json", payload)
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
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_selective_unknown_decision")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
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
