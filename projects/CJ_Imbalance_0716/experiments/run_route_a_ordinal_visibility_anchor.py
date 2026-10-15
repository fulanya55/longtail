"""Evaluate an explicit same-identity ordinal visibility coordinate for Route A."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.models.evidence_readout import OrdinalVisibilityAnchorCalibrator  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_power_robust_decision import _select_robust_threshold  # noqa: E402
from experiments.run_route_a_sawfm_fixed_fusion_cross_power import _sigmoid  # noqa: E402
from experiments.run_route_a_stable_expert_visibility_calibration import (  # noqa: E402
    _oracle_recall,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


VARIANTS = (
    "A1_AnchorNoOrdinal",
    "A2_OrdinalAnchor",
    "A3_OraclePowerAnchor",
)
ORACLE_BY_LEVEL = {"root": 1.0, "40": 1.0 / 3.0, "45": -1.0 / 3.0, "50": -1.0}
_TAG_QUEUE = []
_DEVELOPMENT = None
_MECHANISM = {}
_PAYLOADS = {}


class _CalibratorCollection(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleDict(models)


def _ordinal_loss(visibility, margin=0.30):
    if visibility.ndim != 2 or visibility.shape[1] != 4:
        raise ValueError("visibility must have shape [N, 4]")
    losses = []
    for earlier in range(3):
        for later in range(earlier + 1, 4):
            losses.append(F.softplus(float(margin) - (visibility[:, earlier] - visibility[:, later])))
    return torch.stack(losses, dim=1).mean()


def _ordinal_metrics(visibility):
    values = np.asarray(visibility, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("visibility must have shape [N, 4]")
    comparisons = []
    for earlier in range(3):
        for later in range(earlier + 1, 4):
            comparisons.append(values[:, earlier] > values[:, later])
    concordance = np.stack(comparisons, axis=1).astype(np.float64)
    pairwise_accuracy = float(concordance.mean())
    kendall = float((2.0 * concordance.sum(axis=1) - 6.0).mean() / 6.0)
    return {
        "pairwise_accuracy": pairwise_accuracy,
        "kendall_tau": kendall,
        "level_means": {
            name: float(values[:, index].mean())
            for index, name in enumerate(("root", "40", "45", "50"))
        },
        "identity_count": int(values.shape[0]),
    }


def _tag_sequence(levels):
    tags = [
        ("ordinary_calibration", "root", "mixed"),
        ("ordinary_threshold", "root", "mixed"),
        ("ordinary_early", "root", "mixed"),
    ]
    for level in levels:
        tags.extend([
            ("development", str(level), "positive"),
            ("development", str(level), "negative"),
        ])
    for role in ("scale", "fit", "threshold"):
        tags.extend([
            (role, "root", "positive"),
            (role, "root", "negative"),
        ])
        for level in levels:
            tags.extend([
                (role, str(level), "positive"),
                (role, str(level), "negative"),
            ])
    return tags


def _wrap_add_c2(original):
    def tagged(group, coefficient, intercept):
        result = original(group, coefficient, intercept)
        if not _TAG_QUEUE:
            raise RuntimeError("Visibility payload tag queue exhausted")
        role, level, target = _TAG_QUEUE.pop(0)
        result["role_hint"] = role
        result["level_hint"] = level
        result["target_hint"] = target
        result["oracle_visibility"] = np.full(
            len(result["c2"]), ORACLE_BY_LEVEL[level], dtype=np.float32
        )
        _PAYLOADS[(role, level, target)] = result
        return result
    return tagged


def _wrap_extract_base(original):
    def indexed(args, config, backbone, base_head, h2_checkpoint, indices, device):
        result = original(args, config, backbone, base_head, h2_checkpoint, indices, device)
        result["source_indices"] = np.asarray(indices, dtype=np.int64)
        return result
    return indexed


def _wrap_development(original):
    def captured(*args, **kwargs):
        global _DEVELOPMENT
        _DEVELOPMENT = original(*args, **kwargs)
        return _DEVELOPMENT
    return captured


def _features(group, mean, std, device):
    normalized = (group["visibility_features"] - mean) / std
    return torch.from_numpy(normalized).to(device=device, dtype=torch.float32)


def _model_probability(model, group, mean, std, device, oracle=False, diagnostics=False):
    feature_tensor = _features(group, mean, std, device)
    c2_tensor = torch.from_numpy(group["c2"]).to(device=device, dtype=torch.float32)
    oracle_tensor = None
    if oracle:
        oracle_tensor = torch.from_numpy(group["oracle_visibility"]).to(
            device=device, dtype=torch.float32
        )
    with torch.no_grad():
        output = model(
            c2_tensor,
            feature_tensor,
            oracle_visibility=oracle_tensor,
            return_diagnostics=diagnostics,
        )
    if diagnostics:
        logits, values = output
        return _sigmoid(logits.cpu().numpy()), values
    return _sigmoid(output.cpu().numpy())


def _threshold_key(model, threshold_groups, mean, std, device, max_fpr, oracle):
    ordinary_group, ordinary_targets = threshold_groups[0]
    ordinary = _model_probability(model, ordinary_group, mean, std, device, oracle)
    power = {}
    for level, positive, negative in threshold_groups[1]:
        power[str(level)] = {
            "positive": _model_probability(model, positive, mean, std, device, oracle),
            "negative": _model_probability(model, negative, mean, std, device, oracle),
        }
    threshold, search = _select_robust_threshold(
        ordinary, ordinary_targets, power, max_fpr
    )
    selected = min(search, key=lambda row: abs(row["threshold"] - threshold))
    norm = float(sum(parameter.detach().pow(2).sum().cpu() for parameter in model.parameters()))
    key = (
        selected["mean_power_recall"],
        selected["min_power_recall"],
        selected["ordinary_f1"],
        -norm,
    )
    return key, threshold, selected, norm


def _train_one(args, variant, fit_groups, threshold_groups, mean, std, seed, device):
    raw_features = np.concatenate([group["visibility_features"] for group, _ in fit_groups])
    c2 = np.concatenate([group["c2"] for group, _ in fit_groups])
    targets = np.concatenate([targets for _, targets in fit_groups]).astype(np.float32)
    domains = np.concatenate([
        np.full(len(group["c2"]), index, dtype=np.int64)
        for index, (group, _) in enumerate(fit_groups)
    ])
    features = (raw_features - mean) / std
    torch.manual_seed(301000 + int(seed) + 100 * VARIANTS.index(variant))
    model = OrdinalVisibilityAnchorCalibrator(feature_dim=features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    feature_tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    c2_tensor = torch.from_numpy(c2).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(targets).to(device)
    domain_tensor = torch.from_numpy(domains).to(device)
    weights = torch.empty_like(target_tensor)
    for domain in np.unique(domains):
        mask = domain_tensor == int(domain)
        weights[mask] = len(targets) / (len(np.unique(domains)) * max(int(mask.sum()), 1))
    positive_groups = [fit_groups[index][0] for index in (1, 3, 5, 7)]
    ordinal_features = torch.stack(
        [_features(group, mean, std, device) for group in positive_groups], dim=1
    )
    best_state = None
    best_key = None
    best_epoch = 0
    stale = 0
    history = []
    oracle = variant == "A3_OraclePowerAnchor"
    ordinal_weight = args.ordinal_weight if variant == "A2_OrdinalAnchor" else 0.0
    oracle_fit = torch.from_numpy(
        np.concatenate([group["oracle_visibility"] for group, _ in fit_groups])
    ).to(device=device, dtype=torch.float32)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            c2_tensor,
            feature_tensor,
            oracle_visibility=oracle_fit if oracle else None,
        )
        classification = (
            F.binary_cross_entropy_with_logits(logits, target_tensor, reduction="none")
            * weights
        ).mean()
        ordinal = classification.new_zeros(())
        if ordinal_weight:
            ordered_scores = []
            for index in range(4):
                ordered_scores.append(model.visibility(ordinal_features[:, index, :]).reshape(-1))
            ordinal = _ordinal_loss(torch.stack(ordered_scores, dim=1), args.ordinal_margin)
        regularization = 1e-4 * sum(parameter.pow(2).sum() for parameter in model.parameters())
        loss = classification + ordinal_weight * ordinal + regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        key, threshold, selected, norm = _threshold_key(
            model, threshold_groups, mean, std, device, args.max_negative_fpr, oracle
        )
        history.append({
            "variant": variant,
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "classification_loss": float(classification.detach().cpu()),
            "ordinal_loss": float(ordinal.detach().cpu()),
            "threshold": float(threshold),
            "mean_power_recall": selected["mean_power_recall"],
            "min_power_recall": selected["min_power_recall"],
            "ordinary_f1": selected["ordinary_f1"],
            "parameter_norm": norm,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def _train_collection(args, fit_groups, threshold_groups, seed, device):
    global _MECHANISM
    raw_features = np.concatenate([group["visibility_features"] for group, _ in fit_groups])
    mean = raw_features.mean(axis=0, keepdims=True)
    std = np.maximum(raw_features.std(axis=0, keepdims=True), 1e-5)
    models = {}
    histories = []
    best_epochs = {}
    for variant in VARIANTS:
        model, best_epoch, history = _train_one(
            args, variant, fit_groups, threshold_groups, mean, std, seed, device
        )
        models[variant] = model
        best_epochs[variant] = int(best_epoch)
        histories.extend(history)
    collection = _CalibratorCollection(models).to(device)
    threshold_positive = [
        _PAYLOADS[("threshold", name, "positive")]
        for name in ("root", "40", "45", "50")
    ]
    for variant, key_name in (
        ("A1_AnchorNoOrdinal", "a1_threshold"),
        ("A2_OrdinalAnchor", "threshold"),
    ):
        learned = []
        for group in threshold_positive:
            _, diagnostics = _model_probability(
                models[variant], group, mean, std, device, False, True
            )
            learned.append(diagnostics["learned_visibility"].cpu().numpy())
        _MECHANISM[key_name] = _ordinal_metrics(np.stack(learned, axis=1))
    _MECHANISM["best_epochs"] = best_epochs
    return collection, mean, std, 0.5, max(best_epochs.values()), histories


def _collection_probability(variant, payload, uniform, isotonic, bundle, device):
    collection, mean, std = bundle
    model = collection.models[variant]
    oracle = variant == "A3_OraclePowerAnchor"
    return _model_probability(model, payload, mean, std, device, oracle)


def _previous_a0(path):
    payload = _read(path)
    return [
        {**row, "variant": "A0_PreviousV2"}
        for row in payload["per_run"]
        if row["variant"] == "V2_VisibilityAffine"
    ]


def _aggregate(rows, levels):
    result = {}
    for variant_index, variant in enumerate(("A0_PreviousV2",) + VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                311000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(
                ("ordinary_saw_recall", "ordinary_saw_fpr", "mean_power_recall", "min_power_recall", "max_power_fpr", "mean_oracle_gap")
            )
        }
        result[variant]["by_level"] = {
            str(level): {
                metric: _mean_ci(
                    [row["power"][str(level)][metric] for row in selected],
                    312000 + variant_index * 1000 + int(level) * 10 + metric_index,
                )
                for metric_index, metric in enumerate(
                    ("positive_recall", "hard_negative_fpr", "oracle_recall_at_fpr", "oracle_gap")
                )
            }
            for level in levels
        }
    return result


def _promotion_gate(rows, aggregate, mechanisms, audits, levels):
    a0 = {row["seed"]: row for row in rows if row["variant"] == "A0_PreviousV2"}
    a1 = {row["seed"]: row for row in rows if row["variant"] == "A1_AnchorNoOrdinal"}
    a2 = {row["seed"]: row for row in rows if row["variant"] == "A2_OrdinalAnchor"}
    per_seed = []
    for seed, row in sorted(a2.items()):
        baseline = a0[seed]
        mechanism = mechanisms[str(seed)]
        mechanism_pass = (
            mechanism["threshold"]["pairwise_accuracy"] >= 0.75
            and mechanism["early"]["pairwise_accuracy"] >= 0.70
            and mechanism["early"]["kendall_tau"] >= 0.40
            and mechanism["early"]["pairwise_accuracy"]
            - mechanism["a1_early"]["pairwise_accuracy"] >= 0.10
        )
        decision_pass = (
            row["mean_power_recall"] >= 0.50
            and row["min_power_recall"] >= 0.35
            and row["max_power_fpr"] <= 0.20 + 1e-12
            and row["ordinary_saw_fpr"] - baseline["ordinary_saw_fpr"] <= 0.02 + 1e-12
            and row["ordinary_saw_recall"] - baseline["ordinary_saw_recall"] >= -0.05 - 1e-12
            and row["mean_oracle_gap"] <= 0.15
            and row["mean_power_recall"] >= baseline["mean_power_recall"]
            and row["mean_power_recall"] >= a1[seed]["mean_power_recall"]
        )
        per_seed.append({
            "seed": int(seed),
            "mechanism_pass": bool(mechanism_pass),
            "decision_pass": bool(decision_pass),
            "passed": bool(mechanism_pass and decision_pass),
        })
    a2_agg = aggregate["A2_OrdinalAnchor"]
    checks = {
        "threshold_pairwise_accuracy_0p75": float(np.mean([mechanisms[str(seed)]["threshold"]["pairwise_accuracy"] for seed in a2])) >= 0.75,
        "early_pairwise_accuracy_0p70": float(np.mean([mechanisms[str(seed)]["early"]["pairwise_accuracy"] for seed in a2])) >= 0.70,
        "early_kendall_tau_0p40": float(np.mean([mechanisms[str(seed)]["early"]["kendall_tau"] for seed in a2])) >= 0.40,
        "a2_minus_a1_pairwise_0p10": float(np.mean([mechanisms[str(seed)]["early"]["pairwise_accuracy"] - mechanisms[str(seed)]["a1_early"]["pairwise_accuracy"] for seed in a2])) >= 0.10,
        "mean_power_recall_0p50": a2_agg["mean_power_recall"]["mean"] >= 0.50,
        "each_level_recall_0p35": all(a2_agg["by_level"][str(level)]["positive_recall"]["mean"] >= 0.35 for level in levels),
        "each_level_fpr_0p20": all(a2_agg["by_level"][str(level)]["hard_negative_fpr"]["mean"] <= 0.20 for level in levels),
        "ordinary_fpr_delta_0p02": a2_agg["ordinary_saw_fpr"]["mean"] - aggregate["A0_PreviousV2"]["ordinary_saw_fpr"]["mean"] <= 0.02,
        "ordinary_recall_delta_minus_0p05": a2_agg["ordinary_saw_recall"]["mean"] - aggregate["A0_PreviousV2"]["ordinary_saw_recall"]["mean"] >= -0.05,
        "two_seed_pass": sum(item["passed"] for item in per_seed) >= 2,
        "mean_oracle_gap_0p15": a2_agg["mean_oracle_gap"]["mean"] <= 0.15,
        "beats_a0_and_a1": a2_agg["mean_power_recall"]["mean"] >= max(aggregate["A0_PreviousV2"]["mean_power_recall"]["mean"], aggregate["A1_AnchorNoOrdinal"]["mean_power_recall"]["mean"]),
        "a3_upper_bound_0p60": aggregate["A3_OraclePowerAnchor"]["mean_power_recall"]["mean"] >= 0.60,
        "identity_checkpoint_locked_test_audit": all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits),
    }
    return {"checks": checks, "per_seed": per_seed, "passed": all(checks.values())}


def run(args):
    global _TAG_QUEUE, _MECHANISM, _PAYLOADS
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base labels and identities are misaligned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous = _previous_a0(args.previous_summary)
    previous_by_seed = {row["seed"]: row for row in previous}
    original_add = stable._add_c2_features
    original_extract = stable._extract_base_indices
    original_development = stable._development_pairs
    original_train = stable._train_visibility
    original_probability = stable._calibrator_probability
    original_variants = stable.VARIANTS
    rows, audits, selections, mechanisms = [], [], [], {}
    try:
        stable._add_c2_features = _wrap_add_c2(original_add)
        stable._extract_base_indices = _wrap_extract_base(original_extract)
        stable._development_pairs = _wrap_development(original_development)
        stable._train_visibility = _train_collection
        stable._calibrator_probability = _collection_probability
        stable.VARIANTS = VARIANTS
        for seed in args.seeds:
            _TAG_QUEUE = _tag_sequence(args.levels)
            _MECHANISM = {}
            _PAYLOADS = {}
            seed_rows, audit, selection = stable._run_seed(
                args, config, labels, identities, int(seed), device
            )
            if _TAG_QUEUE:
                raise RuntimeError("Unused visibility payload tags: {}".format(len(_TAG_QUEUE)))
            checkpoint = torch.load(
                args.output_root / "seed_{}".format(seed) / "best_visibility_calibrator.pth",
                map_location=device,
                weights_only=False,
            )
            collection = _CalibratorCollection({variant: OrdinalVisibilityAnchorCalibrator(10) for variant in VARIANTS}).to(device)
            collection.load_state_dict(checkpoint["model_state_dict"])
            mean = checkpoint["feature_mean"]
            std = checkpoint["feature_std"]
            ordinary_early = _PAYLOADS[("ordinary_early", "root", "mixed")]
            index_to_position = {
                int(value): index
                for index, value in enumerate(ordinary_early["source_indices"])
            }
            root_positions = np.asarray(
                [index_to_position[int(value)] for value in _DEVELOPMENT["positive_base_indices"]],
                dtype=np.int64,
            )
            for variant, key_name in (
                ("A1_AnchorNoOrdinal", "a1_early"),
                ("A2_OrdinalAnchor", "early"),
            ):
                learned = []
                for name in ("root", "40", "45", "50"):
                    group = ordinary_early if name == "root" else _PAYLOADS[("development", name, "positive")]
                    _, diagnostics = _model_probability(
                        collection.models[variant], group, mean, std, device, False, True
                    )
                    values = diagnostics["learned_visibility"].cpu().numpy()
                    learned.append(values[root_positions] if name == "root" else values)
                _MECHANISM[key_name] = _ordinal_metrics(np.stack(learned, axis=1))
            mechanisms[str(seed)] = copy.deepcopy(_MECHANISM)
            rows.extend(seed_rows)
            audits.append(audit)
            selections.append({"seed": int(seed), **selection})
    finally:
        stable._add_c2_features = original_add
        stable._extract_base_indices = original_extract
        stable._development_pairs = original_development
        stable._train_visibility = original_train
        stable._calibrator_probability = original_probability
        stable.VARIANTS = original_variants
    rows = [previous_by_seed[int(seed)] for seed in args.seeds] + rows
    aggregate = _aggregate(rows, args.levels)
    gate = _promotion_gate(rows, aggregate, mechanisms, audits, args.levels)
    payload = {
        "status": "completed",
        "device": str(device),
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds),
            "levels": list(args.levels),
            "variants": ["A0_PreviousV2"] + list(VARIANTS),
            "pairs_per_combo": args.pairs_per_combo,
            "parameters": 16,
            "epochs": args.epochs,
            "ordinal_weight": args.ordinal_weight,
            "ordinal_margin": args.ordinal_margin,
            "max_negative_fpr": args.max_negative_fpr,
        },
        "checkpoint_selections": selections,
        "audits": audits,
        "mechanism": mechanisms,
        "per_run": rows,
        "aggregate": aggregate,
        "promotion_gate": gate,
        "next_decision": "run_last_stage_small_lr_unfreeze" if gate["passed"] else "keep_frozen_and_follow_failure_branch",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_ordinal_visibility_anchor_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--stability-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_s2_initialization_stability")
    parser.add_argument("--previous-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration" / "route_a_stable_expert_visibility_calibration_summary.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_ordinal_visibility_anchor")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ordinal-weight", type=float, default=1.0)
    parser.add_argument("--ordinal-margin", type=float, default=0.30)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
