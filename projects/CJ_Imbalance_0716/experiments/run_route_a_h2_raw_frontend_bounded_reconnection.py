"""Bounded reconnection of the frozen H2 head and Raw-only SawFM frontend."""

import argparse
import copy
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
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import CLASS_NAMES, _sha256, _sigmoid  # noqa: E402
from experiments.run_route_a_stage2_reliability_identifiability import _domain_class_weights  # noqa: E402
from experiments.run_route_a_trainable_raw_stage1_sawfm_frontend import (  # noqa: E402
    _score_model, _wrap_base_extract, _wrap_power_extract,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
import experiments.run_route_a_selective_unknown_decision as selective  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


SAW_INDEX = 1
ALPHAS = (0.05, 0.10, 0.20)
GATES = (-1.0, -0.5, 0.0)
RAW_THRESHOLDS = (0.70, 0.80, 0.90)


def _bounded_residual(h2_logit, raw_z, alpha):
    h2_logit = np.asarray(h2_logit, dtype=np.float64)
    raw_z = np.asarray(raw_z, dtype=np.float64)
    return h2_logit + float(alpha) * np.tanh(raw_z) * np.exp(-np.abs(h2_logit))


def _rescue_residual(h2_logit, raw_z, gate, raw_threshold, alpha=0.20):
    h2_logit = np.asarray(h2_logit, dtype=np.float64)
    raw_probability = _sigmoid(raw_z)
    rescue = (h2_logit < float(gate)) & (raw_probability >= float(raw_threshold))
    return h2_logit + float(alpha) * rescue.astype(np.float64)


def _load_frontend(args, seed, device):
    path = args.frontend_root / f"seed_{seed}" / "N0_RawOnly" / "best_frontend.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = RawStage1SawFMFrontend(mode="raw", hidden_channels=16, topk=16, dropout=0.1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, path


def _raw_scores(model, group, device, batch_size):
    proxy = {
        "name": "raw", "targets": np.zeros(len(group["raw_real"]), dtype=np.float32),
        "raw_real": group["raw_real"], "raw_imag": group["raw_imag"],
    }
    return _score_model(model, [proxy], device, batch_size)["raw"]


def _fit_raw_platt(groups, raw_scores):
    scores, targets, domains = [], [], []
    for domain_index, (group, target) in enumerate(groups):
        scores.append(raw_scores[id(group)])
        targets.append(np.asarray(target, dtype=np.uint8))
        domains.append(np.full(len(target), domain_index, dtype=np.int64))
    x = np.concatenate(scores).reshape(-1, 1)
    y = np.concatenate(targets)
    domain = np.concatenate(domains)
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    model.fit(x, y, sample_weight=_domain_class_weights(domain, y))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _candidate_logit(specification, base_logits, raw_z):
    result = np.asarray(base_logits, dtype=np.float64).copy()
    h2 = result[:, SAW_INDEX]
    kind = specification["kind"]
    if kind == "baseline":
        return result
    if kind == "signed":
        result[:, SAW_INDEX] = _bounded_residual(h2, raw_z, specification["alpha"])
    elif kind == "rescue":
        result[:, SAW_INDEX] = _rescue_residual(
            h2, raw_z, specification["gate"], specification["raw_threshold"], specification["alpha"]
        )
    else:
        raise ValueError(f"unknown candidate kind: {kind}")
    return result


def _power_metrics(positive_logit, negative_logit, threshold):
    positive_probability = _sigmoid(positive_logit)
    negative_probability = _sigmoid(negative_logit)
    targets = np.concatenate([np.ones(len(positive_probability)), np.zeros(len(negative_probability))])
    scores = np.concatenate([positive_probability, negative_probability])
    return {
        "positive_recall": float((positive_probability >= threshold).mean()),
        "hard_negative_fpr": float((negative_probability >= threshold).mean()),
        "auroc": float(roc_auc_score(targets, scores)),
        "aupr": float(average_precision_score(targets, scores)),
    }


def _ordinary_metrics(logits, targets, saw_threshold, other_thresholds):
    probability = _sigmoid(logits)
    thresholds = np.asarray(other_thresholds, dtype=np.float64).copy()
    thresholds[SAW_INDEX] = float(saw_threshold)
    prediction = (probability >= thresholds[None, :]).astype(np.uint8)
    metrics = compute_multilabel_metrics(
        targets, prediction, probabilities=probability,
        class_names=CLASS_NAMES, tail_indices=[SAW_INDEX],
    )
    negative = np.asarray(targets[:, SAW_INDEX] == 0)
    positive = ~negative
    metrics.update({
        "saw_negative_fpr": float(prediction[negative, SAW_INDEX].mean()),
        "saw_recall": float(prediction[positive, SAW_INDEX].mean()),
        "saw_aupr": float(average_precision_score(targets[:, SAW_INDEX], probability[:, SAW_INDEX])),
        "threshold": float(saw_threshold),
    })
    return metrics, probability, prediction


def _search_threshold(specification, ordinary, ordinary_targets, power, baseline, other_thresholds, max_fpr):
    rows = []
    best = None
    for threshold in np.arange(0.01, 1.0, 0.01):
        ordinary_metrics, _, _ = _ordinary_metrics(
            ordinary, ordinary_targets, threshold, other_thresholds
        )
        power_metrics = {
            level: _power_metrics(values["positive"], values["negative"], threshold)
            for level, values in power.items()
        }
        feasible = (
            ordinary_metrics["saw_negative_fpr"] <= max_fpr + 1e-12
            and all(item["hard_negative_fpr"] <= max_fpr + 1e-12 for item in power_metrics.values())
            and (
                baseline is None
                or (
                    ordinary_metrics["macro_aupr"] >= baseline["macro_aupr"] - 0.005 - 1e-12
                    and ordinary_metrics["macro_f1"] >= baseline["macro_f1"] - 0.02 - 1e-12
                )
            )
        )
        row = {
            "threshold": float(threshold), "feasible": bool(feasible),
            "ordinary": ordinary_metrics, "power": power_metrics,
        }
        rows.append(row)
        key = (
            float(np.mean([item["positive_recall"] for item in power_metrics.values()])),
            power_metrics["50"]["positive_recall"],
            ordinary_metrics["saw_aupr"],
            -float(specification.get("alpha", 0.0)),
            -abs(float(threshold) - 0.5),
        )
        if feasible and (best is None or key > best[0]): best = (key, row)
    return None if best is None else best[1], rows


def _role_raw(model, payloads, role, levels, device, batch_size):
    result = {}
    if role.startswith("ordinary"):
        group = payloads[(role, "root", "mixed")]
        result["root"] = {"group": group, "raw": _raw_scores(model, group, device, batch_size)}
        return result
    for level in levels:
        result[str(level)] = {}
        for target in ("positive", "negative"):
            group = payloads[(role, str(level), target)]
            result[str(level)][target] = {"group": group, "raw": _raw_scores(model, group, device, batch_size)}
    return result


def _specifications():
    result = [{"name": "B0_H2", "kind": "baseline", "alpha": 0.0}]
    result.extend({"name": f"B{index}_SignedA{alpha:.2f}", "kind": "signed", "alpha": alpha} for index, alpha in enumerate(ALPHAS, 1))
    for gate in GATES:
        for raw_threshold in RAW_THRESHOLDS:
            result.append({
                "name": f"B4_Rescue_g{gate:+.1f}_r{raw_threshold:.2f}",
                "kind": "rescue", "alpha": 0.20, "gate": gate, "raw_threshold": raw_threshold,
            })
    return result


def _run_seed(args, config, labels, identities, seed, device, prior_summary):
    replay_args = copy.copy(args); replay_args.output_root = args.output_root / "frozen_replay"
    original_base = stable._extract_base_indices; original_power = stable._extract_power_identities
    try:
        stable._extract_base_indices = _wrap_base_extract(original_base)
        stable._extract_power_identities = _wrap_power_extract(original_power)
        _, audit = selective._run_seed(replay_args, config, labels, identities, seed, device, prior_summary)
    finally:
        stable._extract_base_indices = original_base; stable._extract_power_identities = original_power
    payloads = selective._PAYLOADS
    fit_groups = selective._FIT_GROUPS
    threshold_groups = selective._THRESHOLD_GROUPS
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    frontend, frontend_path = _load_frontend(args, seed, device)
    raw_cache = {}
    for group, _ in fit_groups:
        raw_cache[id(group)] = _raw_scores(frontend, group, device, args.batch_size)
    coefficient, intercept = _fit_raw_platt(fit_groups, raw_cache)
    def z(group, raw=None):
        values = _raw_scores(frontend, group, device, args.batch_size) if raw is None else raw
        return coefficient * values + intercept

    ordinary_threshold_group, _ = threshold_groups[0]
    ordinary_threshold_targets = labels[plan["threshold_indices"]].astype(np.uint8)
    threshold_payload = {
        "ordinary_logits": ordinary_threshold_group["base_logits"],
        "ordinary_raw_z": z(ordinary_threshold_group),
        "ordinary_targets": np.asarray(ordinary_threshold_targets, dtype=np.uint8),
        "power": {},
    }
    for level, positive, negative in threshold_groups[1]:
        threshold_payload["power"][str(level)] = {
            "positive_logits": positive["base_logits"], "negative_logits": negative["base_logits"],
            "positive_raw_z": z(positive), "negative_raw_z": z(negative),
        }
    early_group = payloads[("ordinary_early", "root", "mixed")]
    early = {
        "ordinary_logits": early_group["base_logits"], "ordinary_raw_z": z(early_group),
        "ordinary_targets": labels[plan["early_stop_indices"]].astype(np.uint8), "power": {},
    }
    for level in args.levels:
        positive = payloads[("development", str(level), "positive")]
        negative = payloads[("development", str(level), "negative")]
        early["power"][str(level)] = {
            "positive_logits": positive["base_logits"], "negative_logits": negative["base_logits"],
            "positive_raw_z": z(positive), "negative_raw_z": z(negative),
        }
    other_thresholds = np.asarray(_read(
        args.frozen_head_root / f"seed_{seed}" / "H2_PosBCE" / "thresholds.json"
    )["modes"]["weak_recall_fpr"], dtype=np.float64)

    candidates = []
    baseline_selected = None
    for specification in _specifications():
        ordinary_logits = _candidate_logit(specification, threshold_payload["ordinary_logits"], threshold_payload["ordinary_raw_z"])
        power_logits = {
            level: {
                "positive": _candidate_logit(specification, values["positive_logits"], values["positive_raw_z"])[:, SAW_INDEX],
                "negative": _candidate_logit(specification, values["negative_logits"], values["negative_raw_z"])[:, SAW_INDEX],
            }
            for level, values in threshold_payload["power"].items()
        }
        selected, search = _search_threshold(
            specification, ordinary_logits, threshold_payload["ordinary_targets"], power_logits,
            None if specification["kind"] == "baseline" else baseline_selected["ordinary"],
            other_thresholds, args.max_negative_fpr,
        )
        if specification["kind"] == "baseline":
            if selected is None: raise RuntimeError("H2 baseline has no feasible threshold")
            baseline_selected = selected
        candidates.append({"specification": specification, "selected": selected, "search": search})
    feasible_new = [item for item in candidates[1:] if item["selected"] is not None]
    if feasible_new:
        chosen = max(feasible_new, key=lambda item: (
            np.mean([value["positive_recall"] for value in item["selected"]["power"].values()]),
            item["selected"]["power"]["50"]["positive_recall"],
            item["selected"]["ordinary"]["saw_aupr"],
            -item["specification"].get("alpha", 0.0),
        ))
    else:
        chosen = candidates[0]
    evaluated = []
    for item in (candidates[0], chosen):
        spec = item["specification"]; threshold = item["selected"]["threshold"]
        ordinary_logits = _candidate_logit(spec, early["ordinary_logits"], early["ordinary_raw_z"])
        ordinary_metrics, probabilities, predictions = _ordinary_metrics(
            ordinary_logits, early["ordinary_targets"], threshold, other_thresholds
        )
        power = {}
        for level, values in early["power"].items():
            positive = _candidate_logit(spec, values["positive_logits"], values["positive_raw_z"])[:, SAW_INDEX]
            negative = _candidate_logit(spec, values["negative_logits"], values["negative_raw_z"])[:, SAW_INDEX]
            power[level] = _power_metrics(positive, negative, threshold)
        evaluated.append({
            "seed": int(seed), "variant": spec["name"], "specification": spec,
            "threshold": float(threshold), "ordinary": ordinary_metrics, "power": power,
            "mean_power_recall": float(np.mean([value["positive_recall"] for value in power.values()])),
            "min_power_recall": float(np.min([value["positive_recall"] for value in power.values()])),
            "max_power_fpr": float(np.max([value["hard_negative_fpr"] for value in power.values()])),
            "other_probability_max_abs_diff": float(np.max(np.abs(
                probabilities[:, [0, 2, 3, 4, 5]] - _sigmoid(early["ordinary_logits"])[:, [0, 2, 3, 4, 5]]
            ))),
            "other_prediction_difference_count": int(np.sum(
                predictions[:, [0, 2, 3, 4, 5]] != (
                    _sigmoid(early["ordinary_logits"])[:, [0, 2, 3, 4, 5]]
                    >= other_thresholds[[0, 2, 3, 4, 5]][None, :]
                )
            )),
            "locked_test_used": False,
        })
    seed_root = args.output_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True)
    _write(seed_root / "candidate_search.json", {"candidates": candidates, "chosen": chosen["specification"]})
    for row in evaluated:
        output = seed_root / row["variant"]; output.mkdir(parents=True, exist_ok=True)
        _write(output / "metrics.json", row)
        _write(output / "run_receipt.json", {
            "status": "completed", "seed": seed, "variant": row["variant"],
            "frontend_sha256": _sha256(frontend_path), "f2_checkpoint_sha256": audit["f2_checkpoint_sha256"],
            "h2_checkpoint_sha256": audit["h2_checkpoint_sha256"],
            "identity_overlap": audit["identity_overlap"], "locked_test_used": False,
        })
    _write(seed_root / "source_and_identity_audit.json", audit)
    return evaluated, audit


def _aggregate(rows):
    baselines = [row for row in rows if row["variant"] == "B0_H2"]
    chosen = [row for row in rows if row["variant"] != "B0_H2"]
    result = {}
    for name, selected, offset in (("B0_H2", baselines, 0), ("ChosenBounded", chosen, 100)):
        result[name] = {
            metric: _mean_ci([row[metric] for row in selected], 561000 + offset + index)
            for index, metric in enumerate(("mean_power_recall", "min_power_recall", "max_power_fpr"))
        }
        result[name]["ordinary"] = {
            metric: _mean_ci([row["ordinary"][metric] for row in selected], 562000 + offset + index)
            for index, metric in enumerate(("macro_aupr", "macro_f1", "hamming_loss", "saw_negative_fpr", "saw_recall", "saw_aupr"))
        }
        result[name]["by_level"] = {
            level: {
                metric: _mean_ci([row["power"][level][metric] for row in selected], 563000 + offset + int(level) * 10 + index)
                for index, metric in enumerate(("positive_recall", "hard_negative_fpr", "auroc", "aupr"))
            }
            for level in ("40", "45", "50")
        }
    return result


def _gate(rows, audits):
    indexed = {}
    for row in rows: indexed[(row["seed"], "baseline" if row["variant"] == "B0_H2" else "chosen")] = row
    per_seed = []
    for seed in sorted({row["seed"] for row in rows}):
        baseline = indexed[(seed, "baseline")]; chosen = indexed[(seed, "chosen")]
        checks = {
            "mean_power_recall_gain_0p10": chosen["mean_power_recall"] >= baseline["mean_power_recall"] + 0.10,
            "50db_recall_gain_0p15": chosen["power"]["50"]["positive_recall"] >= baseline["power"]["50"]["positive_recall"] + 0.15,
            "power_fpr_0p20": chosen["max_power_fpr"] <= 0.20,
            "ordinary_saw_fpr_0p20": chosen["ordinary"]["saw_negative_fpr"] <= 0.20,
            "macro_aupr_noninferior": chosen["ordinary"]["macro_aupr"] >= baseline["ordinary"]["macro_aupr"] - 0.005,
            "macro_f1_noninferior": chosen["ordinary"]["macro_f1"] >= baseline["ordinary"]["macro_f1"] - 0.02,
            "other_probabilities_unchanged": chosen["other_probability_max_abs_diff"] == 0.0,
            "other_predictions_unchanged": chosen["other_prediction_difference_count"] == 0,
        }
        per_seed.append({"seed": seed, "chosen_variant": chosen["variant"], "checks": checks, "passed": all(checks.values())})
    audit_pass = all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits)
    count = int(sum(item["passed"] for item in per_seed))
    return {"per_seed": per_seed, "passed_seed_count": count, "audit_passed": audit_pass, "passed": count >= 2 and audit_pass}


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels): raise ValueError("identity labels drifted")
    prior_summary = _read(args.prior_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits = [], []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(args, config, labels, identities, int(seed), device, prior_summary)
        rows.extend(seed_rows); audits.append(audit)
    aggregate = _aggregate(rows); gate = _gate(rows, audits)
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {"seeds": list(args.seeds), "levels": list(args.levels), "alphas": list(ALPHAS), "gates": list(GATES), "raw_thresholds": list(RAW_THRESHOLDS), "frozen_sources": True},
        "audits": audits, "per_run": rows, "aggregate": aggregate, "promotion_gate": gate,
        "next_decision": "confirm_bounded_reconnection_on_new_seeds_before_locked_test" if gate["passed"] else "keep_raw_frontend_as_independent_mechanism_do_not_reconnect_h2",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_h2_raw_frontend_bounded_reconnection_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


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
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_h2_raw_frontend_bounded_reconnection")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-positive-fpr", type=float, default=0.20)
    parser.add_argument("--max-positive-fn", type=float, default=0.10)
    parser.add_argument("--min-ordinary-coverage", type=float, default=0.70)
    parser.add_argument("--min-power-coverage", type=float, default=0.40)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
