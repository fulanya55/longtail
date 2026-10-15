"""Audit the frozen Raw-IQ SawFM frontend as an independent classifier."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_trainable_raw_stage1_sawfm_frontend import (  # noqa: E402
    _score_model,
    _wrap_base_extract,
    _wrap_power_extract,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
import experiments.run_route_a_selective_unknown_decision as selective  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


SAW_INDEX = 1
SWEPT_INDICES = (3, 4, 5)
VARIANTS = ("I0_H2Platt", "I1_RawPlatt", "I2_RawIsotonic")


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _fit_platt(scores, targets):
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(np.asarray(scores).reshape(-1, 1), np.asarray(targets, dtype=np.uint8))
    return model


def _platt_probability(model, scores):
    return model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


def _fit_isotonic(scores, targets):
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(np.asarray(scores, dtype=np.float64), np.asarray(targets, dtype=np.uint8))
    return model


def _ece(probability, targets, bins=10):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(len(probability), 1)
    value = 0.0
    for index in range(int(bins)):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            value += float(mask.sum()) / total * abs(float(probability[mask].mean()) - float(targets[mask].mean()))
    return float(value)


def _binary_rank_metrics(probability, targets):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.uint8)
    return {
        "auroc": float(roc_auc_score(targets, probability)),
        "aupr": float(average_precision_score(targets, probability)),
        "positive_count": int(targets.sum()),
        "negative_count": int((1 - targets).sum()),
    }


def _decision_metrics(probability, targets, threshold):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    decision = probability >= float(threshold)
    positive = targets
    negative = ~targets
    return {
        "threshold": float(threshold),
        "recall": float(decision[positive].mean()) if positive.any() else None,
        "fpr": float(decision[negative].mean()) if negative.any() else None,
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
    }


def _calibration_metrics(probability, targets):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return {
        "brier": float(np.mean((probability - targets) ** 2)),
        "ece_10bin": _ece(probability, targets, bins=10),
    }


def _raw_logits(model, group, device, batch_size):
    proxy = {
        "name": "score",
        "targets": np.zeros(len(group["raw_real"]), dtype=np.float32),
        "raw_real": group["raw_real"],
        "raw_imag": group["raw_imag"],
    }
    return _score_model(model, [proxy], device, batch_size)["score"]


def _h2_logits(group):
    return np.asarray(group["base_logits"][:, SAW_INDEX], dtype=np.float64)


def _power_probability(payloads, role, levels, score_fn, calibrate):
    result = {}
    for level in levels:
        positive = score_fn(payloads[(role, str(level), "positive")])
        negative = score_fn(payloads[(role, str(level), "negative")])
        result[str(level)] = {
            "positive": calibrate(positive),
            "negative": calibrate(negative),
        }
    return result


def _search_unified_threshold(ordinary_probability, ordinary_targets, power, step, max_fpr):
    rows = []
    best = None
    for threshold in np.arange(float(step), 1.0, float(step)):
        ordinary = _decision_metrics(ordinary_probability, ordinary_targets, threshold)
        by_level = {
            level: {
                "recall": float((values["positive"] >= threshold).mean()),
                "fpr": float((values["negative"] >= threshold).mean()),
            }
            for level, values in power.items()
        }
        feasible = (
            ordinary["fpr"] <= max_fpr + 1e-12
            and all(item["fpr"] <= max_fpr + 1e-12 for item in by_level.values())
        )
        row = {
            "threshold": float(threshold),
            "feasible": bool(feasible),
            "ordinary_recall": ordinary["recall"],
            "ordinary_fpr": ordinary["fpr"],
            "mean_power_recall": float(np.mean([item["recall"] for item in by_level.values()])),
            "min_power_recall": float(np.min([item["recall"] for item in by_level.values()])),
            "max_power_fpr": float(np.max([item["fpr"] for item in by_level.values()])),
            "power": by_level,
        }
        rows.append(row)
        key = (
            row["mean_power_recall"],
            row["ordinary_recall"],
            by_level.get("50", {"recall": 0.0})["recall"],
            float(threshold),
        )
        if feasible and (best is None or key > best[0]):
            best = (key, row)
    return None if best is None else best[1], rows


def _subgroup_metrics(probability, targets, full_labels, threshold):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    full_labels = np.asarray(full_labels)
    swept = full_labels[:, list(SWEPT_INDICES)].astype(bool).any(axis=1)
    negative = ~targets
    result = {}
    for name, positive in (
        ("with_swept", targets & swept),
        ("without_swept", targets & ~swept),
    ):
        subset = positive | negative
        rank = _binary_rank_metrics(probability[subset], positive[subset].astype(np.uint8))
        rank["recall"] = float((probability[positive] >= threshold).mean()) if positive.any() else None
        result[name] = rank
    return result


def _load_frontend(args, seed, device):
    path = args.frontend_root / f"seed_{seed}" / "N0_RawOnly" / "best_frontend.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=args.hidden_channels,
        topk=args.topk, dropout=args.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


def _run_seed(args, config, labels, identities, seed, device, prior_summary):
    replay_args = copy.copy(args)
    replay_args.output_root = args.output_root / "frozen_replay"
    original_base = stable._extract_base_indices
    original_power = stable._extract_power_identities
    try:
        stable._extract_base_indices = _wrap_base_extract(original_base)
        stable._extract_power_identities = _wrap_power_extract(original_power)
        _, audit = selective._run_seed(
            replay_args, config, labels, identities, seed, device, prior_summary
        )
    finally:
        stable._extract_base_indices = original_base
        stable._extract_power_identities = original_power

    payloads = selective._PAYLOADS
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    frontend, frontend_path = _load_frontend(args, seed, device)
    raw_score = lambda group: _raw_logits(frontend, group, device, args.batch_size)
    h2_score = _h2_logits

    calibration_group = payloads[("ordinary_calibration", "root", "mixed")]
    calibration_indices = np.asarray(plan["calibration_indices"], dtype=np.int64)
    calibration_targets = labels[calibration_indices, SAW_INDEX].astype(np.uint8)
    h2_calibrator = _fit_platt(h2_score(calibration_group), calibration_targets)
    raw_calibration_scores = raw_score(calibration_group)
    raw_platt = _fit_platt(raw_calibration_scores, calibration_targets)
    raw_isotonic = _fit_isotonic(raw_calibration_scores, calibration_targets)

    variants = {
        "I0_H2Platt": (h2_score, lambda values: _platt_probability(h2_calibrator, values)),
        "I1_RawPlatt": (raw_score, lambda values: _platt_probability(raw_platt, values)),
        "I2_RawIsotonic": (raw_score, lambda values: raw_isotonic.predict(np.asarray(values))),
    }
    threshold_group = payloads[("ordinary_threshold", "root", "mixed")]
    threshold_indices = np.asarray(plan["threshold_indices"], dtype=np.int64)
    threshold_targets = labels[threshold_indices, SAW_INDEX].astype(np.uint8)
    early_group = payloads[("ordinary_early", "root", "mixed")]
    early_indices = np.asarray(plan["early_stop_indices"], dtype=np.int64)
    early_targets = labels[early_indices, SAW_INDEX].astype(np.uint8)
    early_labels = labels[early_indices]

    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    rows = []
    threshold_searches = {}
    calibration_payload = {
        "calibration_role": "ordinary_calibration",
        "target_count": int(len(calibration_targets)),
        "positive_count": int(calibration_targets.sum()),
        "h2_platt": {
            "coefficient": float(h2_calibrator.coef_[0, 0]),
            "intercept": float(h2_calibrator.intercept_[0]),
        },
        "raw_platt": {
            "coefficient": float(raw_platt.coef_[0, 0]),
            "intercept": float(raw_platt.intercept_[0]),
        },
        "raw_isotonic": {
            "x_thresholds": raw_isotonic.X_thresholds_.tolist(),
            "y_thresholds": raw_isotonic.y_thresholds_.tolist(),
        },
    }
    _write(seed_root / "calibration.json", calibration_payload)

    for variant, (score_fn, calibrate) in variants.items():
        ordinary_threshold_probability = calibrate(score_fn(threshold_group))
        power_threshold_probability = _power_probability(
            payloads, "threshold", args.levels, score_fn, calibrate
        )
        selected, search = _search_unified_threshold(
            ordinary_threshold_probability, threshold_targets,
            power_threshold_probability, args.grid_step, args.max_fpr,
        )
        threshold_searches[variant] = {"selected": selected, "rows": search}
        feasible = selected is not None
        threshold = float(selected["threshold"]) if feasible else 1.0

        ordinary_probability = calibrate(score_fn(early_group))
        ordinary = {
            **_binary_rank_metrics(ordinary_probability, early_targets),
            **_decision_metrics(ordinary_probability, early_targets, threshold),
            **_calibration_metrics(ordinary_probability, early_targets),
        }
        ordinary["subgroups"] = _subgroup_metrics(
            ordinary_probability, early_targets, early_labels, threshold
        )
        power_probability = _power_probability(
            payloads, "development", args.levels, score_fn, calibrate
        )
        power = {}
        for level, values in power_probability.items():
            probability = np.concatenate([values["positive"], values["negative"]])
            targets = np.concatenate([
                np.ones(len(values["positive"]), dtype=np.uint8),
                np.zeros(len(values["negative"]), dtype=np.uint8),
            ])
            power[level] = {
                **_binary_rank_metrics(probability, targets),
                **_decision_metrics(probability, targets, threshold),
            }
        row = {
            "seed": int(seed),
            "variant": variant,
            "threshold_feasible": bool(feasible),
            "selected_threshold": threshold,
            "ordinary": ordinary,
            "power": power,
            "mean_power_recall": float(np.mean([item["recall"] for item in power.values()])),
            "worst_power_recall": float(np.min([item["recall"] for item in power.values()])),
            "max_power_fpr": float(np.max([item["fpr"] for item in power.values()])),
            "mean_power_auroc": float(np.mean([item["auroc"] for item in power.values()])),
            "mean_power_aupr": float(np.mean([item["aupr"] for item in power.values()])),
            "split_sha256": plan["split_sha256"],
            "identity_overlap": int(audit["identity_overlap"]),
            "locked_test_used": False,
        }
        rows.append(row)
        output_dir = seed_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(output_dir / "predictions.npz"),
            ordinary_probability=ordinary_probability,
            ordinary_target=early_targets,
            ordinary_source_index=early_indices,
            **{
                f"{target}_{level}db": values[target]
                for level, values in power_probability.items()
                for target in ("positive", "negative")
            },
        )
        _write(output_dir / "metrics.json", row)
        _write(output_dir / "run_receipt.json", {
            "status": "completed" if feasible else "no_feasible_threshold",
            "seed": int(seed), "variant": variant,
            "frontend_checkpoint": str(frontend_path),
            "frontend_checkpoint_sha256": _sha256(frontend_path),
            "split_sha256": plan["split_sha256"],
            "identity_overlap": int(audit["identity_overlap"]),
            "locked_test_used": False,
        })

    _write(seed_root / "threshold_search.json", threshold_searches)
    source_audit = {
        **audit,
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
        "prior_summary_sha256": _sha256(args.prior_summary),
        "split_sha256": plan["split_sha256"],
        "calibration_role": "ordinary_calibration",
        "threshold_roles": ["ordinary_threshold", "power_threshold"],
        "evaluation_roles": ["ordinary_early", "power_development"],
        "identity_overlap": int(audit["identity_overlap"]),
        "locked_test_used": False,
    }
    _write(seed_root / "source_and_identity_audit.json", source_audit)
    return rows, source_audit


def _aggregate(rows, levels):
    aggregate = {}
    scalar_metrics = (
        "mean_power_recall", "worst_power_recall", "max_power_fpr",
        "mean_power_auroc", "mean_power_aupr",
    )
    ordinary_metrics = ("auroc", "aupr", "recall", "fpr", "brier", "ece_10bin")
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                611000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(scalar_metrics)
        }
        aggregate[variant]["ordinary"] = {
            metric: _mean_ci(
                [row["ordinary"][metric] for row in selected],
                612000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(ordinary_metrics)
        }
        aggregate[variant]["ordinary"]["subgroups"] = {
            subgroup: {
                metric: _mean_ci(
                    [row["ordinary"]["subgroups"][subgroup][metric] for row in selected],
                    613000 + variant_index * 1000 + subgroup_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("auroc", "aupr", "recall"))
            }
            for subgroup_index, subgroup in enumerate(("with_swept", "without_swept"))
        }
        aggregate[variant]["by_level"] = {
            str(level): {
                metric: _mean_ci(
                    [row["power"][str(level)][metric] for row in selected],
                    614000 + variant_index * 1000 + int(level) * 10 + metric_index,
                )
                for metric_index, metric in enumerate(("auroc", "aupr", "recall", "fpr"))
            }
            for level in levels
        }
        aggregate[variant]["feasible_seed_count"] = int(sum(row["threshold_feasible"] for row in selected))
    return aggregate


def _promotion_gate(rows, audits, levels):
    h2 = {row["seed"]: row for row in rows if row["variant"] == "I0_H2Platt"}
    result = {}
    for variant in ("I1_RawPlatt", "I2_RawIsotonic"):
        per_seed = []
        for row in [item for item in rows if item["variant"] == variant]:
            baseline = h2[row["seed"]]
            checks = {
                "threshold_feasible": row["threshold_feasible"],
                "ordinary_aupr_not_below_h2_minus_0p02": row["ordinary"]["aupr"] >= baseline["ordinary"]["aupr"] - 0.02,
                "ordinary_fpr_0p20": row["ordinary"]["fpr"] <= 0.20 + 1e-12,
                "ordinary_recall_gain_0p05": row["ordinary"]["recall"] >= baseline["ordinary"]["recall"] + 0.05,
                "without_swept_recall_0p60": row["ordinary"]["subgroups"]["without_swept"]["recall"] >= 0.60,
                "mean_power_recall_0p70": row["mean_power_recall"] >= 0.70,
                "50db_recall_0p60": row["power"]["50"]["recall"] >= 0.60,
                "each_power_fpr_0p20": all(row["power"][str(level)]["fpr"] <= 0.20 + 1e-12 for level in levels),
                "brier_0p20": row["ordinary"]["brier"] <= 0.20,
                "ece_0p10": row["ordinary"]["ece_10bin"] <= 0.10,
                "identity_locked_test_audit": row["identity_overlap"] == 0 and not row["locked_test_used"],
            }
            mechanism_checks = [
                checks["ordinary_aupr_not_below_h2_minus_0p02"],
                checks["ordinary_recall_gain_0p05"],
                checks["without_swept_recall_0p60"],
                checks["mean_power_recall_0p70"],
                checks["50db_recall_0p60"],
            ]
            per_seed.append({
                "seed": int(row["seed"]), "checks": checks,
                "mechanism_passed": bool(all(mechanism_checks)),
                "passed": bool(all(checks.values())),
            })
        result[variant] = {
            "per_seed": per_seed,
            "passed_seed_count": int(sum(item["passed"] for item in per_seed)),
            "mechanism_passed_seed_count": int(sum(item["mechanism_passed"] for item in per_seed)),
            "passed": sum(item["passed"] for item in per_seed) >= 2,
        }
    audit_passed = all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits)
    raw_passed = any(value["passed"] for value in result.values()) and audit_passed
    return {"variants": result, "audit_passed": audit_passed, "passed": raw_passed}


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
    gate = _promotion_gate(rows, audits, args.levels)
    mechanism_count = max(
        value["mechanism_passed_seed_count"] for value in gate["variants"].values()
    )
    no_swept_count = max(
        sum(item["checks"]["without_swept_recall_0p60"] for item in value["per_seed"])
        for value in gate["variants"].values()
    )
    if gate["passed"]:
        next_decision = "expand_raw_independent_sawfm_classifier_to_five_seeds"
    elif mechanism_count >= 2:
        next_decision = "continue_calibration_only_raw_ranking_and_recall_are_sufficient"
    elif no_swept_count < 2:
        next_decision = "restrict_raw_frontend_to_swept_background_sawfm_expert"
    else:
        next_decision = "stop_raw_frontend_independent_decision_route"
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "levels": list(args.levels),
            "variants": list(VARIANTS), "grid_step": args.grid_step,
            "max_fpr": args.max_fpr, "frozen_frontend": True,
            "calibration_role": "ordinary_calibration",
            "threshold_roles": ["ordinary_threshold", "power_threshold"],
            "evaluation_roles": ["ordinary_early", "power_development"],
        },
        "audits": audits, "per_run": rows, "aggregate": aggregate,
        "promotion_gate": gate, "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "route_a_raw_frontend_independent_classifier_audit_summary.json"
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
    parser.add_argument("--frontend-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_trainable_raw_stage1_sawfm_frontend")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_frontend_independent_classifier_audit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-fpr", type=float, default=0.20)
    parser.add_argument("--max-positive-fpr", type=float, default=0.20)
    parser.add_argument("--max-positive-fn", type=float, default=0.10)
    parser.add_argument("--min-ordinary-coverage", type=float, default=0.70)
    parser.add_argument("--min-power-coverage", type=float, default=0.40)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
