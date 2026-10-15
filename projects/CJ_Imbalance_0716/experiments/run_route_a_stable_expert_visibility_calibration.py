"""Select the stable 2043 S2 checkpoint and calibrate C2 by evidence visibility."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.models.evidence_readout import VisibilityAffineCalibrator  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_label_local_saw import F2_MODEL_NAME, _load_h2_head  # noqa: E402
from experiments.run_route_a_power_robust_decision import (  # noqa: E402
    _decision_metrics,
    _fit_dual_platt,
    _platt_probability,
    _select_robust_threshold,
)
from experiments.run_route_a_sawfm_expert_routing import (  # noqa: E402
    _expert_scores,
    _extract_base_indices,
    _extract_power_identities,
    _load_expert,
)
from experiments.run_route_a_sawfm_fixed_fusion_cross_power import _sigmoid  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import (  # noqa: E402
    _binary_metrics,
    _development_pairs,
    _level_lookup,
    _model_scores,
    _stratified_triplets,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


SAW_INDEX = 1
SWEPT_INDICES = (3, 4, 5)
FIXED_INDICES = (0, 2)
VARIANTS = ("V0_UniformPlatt", "V1_Isotonic", "V2_VisibilityAffine")


def _split_three_by_combo(selection, labels, seed):
    positive_indices = selection["positive_base_indices"]
    keys = [tuple(np.delete(labels[index], SAW_INDEX).tolist()) for index in positive_indices]
    rng = np.random.default_rng(int(seed))
    roles = [[], [], []]
    for key in sorted(set(keys)):
        positions = np.asarray([index for index, value in enumerate(keys) if value == key], dtype=np.int64)
        rng.shuffle(positions)
        boundaries = np.array_split(positions, 3)
        for role, values in zip(roles, boundaries):
            role.extend(values.tolist())
    return tuple(np.asarray(role, dtype=np.int64) for role in roles)


def _expert_evidence(model, extracted, device, topk=16):
    result = dict(extracted)
    scores, evidence_map = _model_scores(
        model, extracted["features"], device, return_maps=True
    )
    count = min(int(topk), evidence_map.shape[1])
    topk_mean = np.partition(evidence_map, -count, axis=1)[:, -count:].mean(axis=1)
    result.update(
        {
            "expert_raw": scores.astype(np.float32),
            "evidence_topk": topk_mean.astype(np.float32),
            "evidence_max": evidence_map.max(axis=1).astype(np.float32),
            "stage2_abs_mean": np.abs(extracted["features"].astype(np.float32)).mean(axis=(1, 2)),
            "stage2_temporal_std": extracted["features"].astype(np.float32).std(axis=2).mean(axis=1),
        }
    )
    return result


def _fit_expert_scale(groups):
    scores, targets, domains = [], [], []
    for domain_index, (positive, negative) in enumerate(groups):
        scores.extend([positive["expert_raw"], negative["expert_raw"]])
        targets.extend([np.ones(len(positive["expert_raw"])), np.zeros(len(negative["expert_raw"]))])
        domains.extend([
            np.full(len(positive["expert_raw"]), domain_index, dtype=np.int64),
            np.full(len(negative["expert_raw"]), domain_index, dtype=np.int64),
        ])
    score_values = np.concatenate(scores).reshape(-1, 1)
    target_values = np.concatenate(targets).astype(np.uint8)
    domain_values = np.concatenate(domains)
    weights = np.zeros(len(target_values), dtype=np.float64)
    unique_domains = np.unique(domain_values)
    for domain in unique_domains:
        mask = domain_values == domain
        weights[mask] = len(target_values) / (len(unique_domains) * max(int(mask.sum()), 1))
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(score_values, target_values, sample_weight=weights)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _add_c2_features(group, coefficient, intercept):
    result = dict(group)
    expert = coefficient * group["expert_raw"] + intercept
    base = group["base_logits"].astype(np.float32)
    h2 = base[:, SAW_INDEX]
    c2 = 0.20 * h2 + 0.80 * expert
    result["expert_scaled"] = expert.astype(np.float32)
    result["c2"] = c2.astype(np.float32)
    result["visibility_features"] = np.stack(
        [
            c2,
            h2,
            expert,
            np.abs(h2 - expert),
            base[:, list(SWEPT_INDICES)].max(axis=1),
            base[:, list(FIXED_INDICES)].max(axis=1),
            group["evidence_topk"],
            group["evidence_max"],
            group["stage2_abs_mean"],
            group["stage2_temporal_std"],
        ],
        axis=1,
    ).astype(np.float32)
    return result


def _matched_metrics(positive_scores, negative_scores):
    return _binary_metrics(positive_scores, negative_scores)


def _select_2043_checkpoint(args, config, labels, identities, plan, backbone, base_head, h2_checkpoint, power_files, level_info, device):
    selection = _stratified_triplets(plan["train_indices"], labels, identities, 40, 161000 + 2043)
    order = np.random.default_rng(163000 + 2043).permutation(len(selection["positive_identities"]))
    split = int(round(0.8 * len(order)))
    validation_positions = order[split:]
    positive_ids = selection["positive_identities"][validation_positions]
    negative_ids = selection["hard_negative_identities"][validation_positions]
    features = {}
    for level in args.levels:
        features[str(level)] = {
            "positive": _extract_power_identities(
                args, config, backbone, base_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)], positive_ids, device
            )["features"],
            "negative": _extract_power_identities(
                args, config, backbone, base_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)], negative_ids, device
            )["features"],
        }
    candidates = []
    best_key = None
    selected_path = None
    for initialization_seed in args.initialization_seeds:
        path = args.stability_root / "init_{}".format(initialization_seed) / "best_saw_stage2_evidence.pth"
        model, _ = _load_expert(path, device)
        power = {}
        for level in args.levels:
            positive = _model_scores(model, features[str(level)]["positive"], device)
            negative = _model_scores(model, features[str(level)]["negative"], device)
            power[str(level)] = _matched_metrics(positive, negative)
        aurocs = [power[str(level)]["auroc"] for level in args.levels]
        auprs = [power[str(level)]["aupr"] for level in args.levels]
        row = {
            "initialization_seed": int(initialization_seed),
            "checkpoint_path": str(path),
            "checkpoint_sha256": _sha256(path),
            "power": power,
            "worst_auroc": float(min(aurocs)),
            "mean_auroc": float(np.mean(aurocs)),
            "mean_aupr": float(np.mean(auprs)),
        }
        candidates.append(row)
        key = (row["worst_auroc"], row["mean_auroc"], row["mean_aupr"], -int(initialization_seed))
        if best_key is None or key > best_key:
            best_key = key
            selected_path = path
    return selected_path, {
        "selection_role": "original_s2_validation_identities",
        "levels": list(args.levels),
        "validation_pair_count": int(len(validation_positions)),
        "selection_sha256": selection["selection_sha256"],
        "candidates": candidates,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": _sha256(selected_path),
        "early_development_used": False,
        "locked_test_used": False,
    }


def _fit_uniform_platt(groups):
    scores, targets, weights = [], [], []
    for domain_index, (values, target) in enumerate(groups):
        scores.append(values)
        targets.append(target)
        weights.append(np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64))
    score_values = np.concatenate(scores).reshape(-1, 1)
    target_values = np.concatenate(targets).astype(np.uint8)
    sample_weights = np.concatenate(weights)
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(score_values, target_values, sample_weight=sample_weights)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _fit_isotonic(groups):
    scores = np.concatenate([values for values, _ in groups])
    targets = np.concatenate([targets for _, targets in groups])
    weights = np.concatenate([
        np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64)
        for values, _ in groups
    ])
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(scores, targets, sample_weight=weights)
    return model


def _robust_threshold(ordinary_probability, ordinary_targets, power_probability, max_fpr):
    return _select_robust_threshold(
        ordinary_probability, ordinary_targets, power_probability, max_fpr
    )


def _calibrator_probability(variant, payload, uniform, isotonic, visibility_bundle, device):
    if variant == "V0_UniformPlatt":
        return _platt_probability(payload["c2"], uniform[0], uniform[1])
    if variant == "V1_Isotonic":
        return isotonic.predict(payload["c2"]).astype(np.float64)
    model, mean, std = visibility_bundle
    features = (payload["visibility_features"] - mean) / std
    with torch.no_grad():
        logits = model(
            torch.from_numpy(payload["c2"]).to(device=device, dtype=torch.float32),
            torch.from_numpy(features).to(device=device, dtype=torch.float32),
        )
    return _sigmoid(logits.cpu().numpy())


def _oracle_recall(positive_probability, negative_probability, max_fpr):
    best = 0.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        fpr = float((negative_probability >= threshold).mean())
        if fpr <= max_fpr + 1e-12:
            best = max(best, float((positive_probability >= threshold).mean()))
    return best


def _train_visibility(args, fit_groups, threshold_groups, seed, device):
    raw_features = np.concatenate([group["visibility_features"] for group, _ in fit_groups])
    c2 = np.concatenate([group["c2"] for group, _ in fit_groups])
    targets = np.concatenate([targets for _, targets in fit_groups]).astype(np.float32)
    domain = np.concatenate([
        np.full(len(group["c2"]), index, dtype=np.int64)
        for index, (group, _) in enumerate(fit_groups)
    ])
    mean = raw_features.mean(axis=0, keepdims=True)
    std = np.maximum(raw_features.std(axis=0, keepdims=True), 1e-5)
    features = (raw_features - mean) / std
    torch.manual_seed(271000 + int(seed))
    model = VisibilityAffineCalibrator(feature_dim=features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    feature_tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    c2_tensor = torch.from_numpy(c2).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(targets).to(device)
    domain_tensor = torch.from_numpy(domain).to(device)
    weights = torch.empty_like(target_tensor)
    for value in np.unique(domain):
        mask = domain_tensor == int(value)
        weights[mask] = len(targets) / (len(np.unique(domain)) * max(int(mask.sum()), 1))
    best_state = None
    best_threshold = None
    best_key = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(c2_tensor, feature_tensor)
        classification = (F.binary_cross_entropy_with_logits(logits, target_tensor, reduction="none") * weights).mean()
        regularization = 1e-4 * sum(parameter.pow(2).sum() for parameter in model.parameters())
        loss = classification + regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        ordinary_group, ordinary_targets = threshold_groups[0]
        ordinary_probability = _calibrator_probability(
            "V2_VisibilityAffine", ordinary_group, None, None, (model, mean, std), device
        )
        power_probability = {}
        for level, positive, negative in threshold_groups[1]:
            power_probability[str(level)] = {
                "positive": _calibrator_probability("V2_VisibilityAffine", positive, None, None, (model, mean, std), device),
                "negative": _calibrator_probability("V2_VisibilityAffine", negative, None, None, (model, mean, std), device),
            }
        threshold, _ = _robust_threshold(
            ordinary_probability, ordinary_targets, power_probability, args.max_negative_fpr
        )
        recalls = [float((values["positive"] >= threshold).mean()) for values in power_probability.values()]
        ordinary_prediction = ordinary_probability >= threshold
        ordinary_f1 = float(f1_score(ordinary_targets, ordinary_prediction, zero_division=0))
        parameter_norm = float(sum(parameter.detach().pow(2).sum().cpu() for parameter in model.parameters()))
        key = (float(np.mean(recalls)), float(np.min(recalls)), ordinary_f1, -parameter_norm)
        history.append({
            "epoch": epoch, "loss": float(loss.detach().cpu()),
            "classification_loss": float(classification.detach().cpu()),
            "threshold": threshold, "mean_power_recall": float(np.mean(recalls)),
            "min_power_recall": float(np.min(recalls)), "ordinary_f1": ordinary_f1,
            "parameter_norm": parameter_norm,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    model.load_state_dict(best_state)
    return model, mean, std, best_threshold, best_epoch, history


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_seed(args, config, labels, identities, seed, device):
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    receipt = _read(source_dir / "run_receipt.json")
    if receipt.get("locked_test_used") or receipt["split_sha256"] != plan["split_sha256"]:
        raise ValueError("F2 source invalid or split drifted")
    backbone, f2_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    base_head, h2_checkpoint, h2_path, _ = _load_h2_head(seed, args.frozen_head_root, device)
    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    level_info = _level_lookup(power_files, args.levels)
    if seed == 2043:
        expert_path, selection_manifest = _select_2043_checkpoint(
            args, config, labels, identities, plan, backbone, base_head,
            h2_checkpoint, power_files, level_info, device
        )
    else:
        expert_path = args.expert_root / "seed_{}".format(seed) / "S2_TeacherHardNeg" / "best_saw_stage2_evidence.pth"
        selection_manifest = {
            "selection_role": "prior_frozen_expert",
            "selected_checkpoint": str(expert_path),
            "selected_checkpoint_sha256": _sha256(expert_path),
            "early_development_used": False,
            "locked_test_used": False,
        }
    expert, _ = _load_expert(expert_path, device)

    expert_selection = _stratified_triplets(plan["train_indices"], labels, identities, 40, 161000 + seed)
    expert_used = set(expert_selection["positive_base_indices"].tolist() + expert_selection["hard_negative_base_indices"].tolist() + expert_selection["random_negative_base_indices"].tolist())
    remaining_after_expert = np.asarray([index for index in plan["train_indices"] if int(index) not in expert_used], dtype=np.int64)
    route_selection = _stratified_triplets(remaining_after_expert, labels, identities, 20, 201000 + seed)
    route_used = set(route_selection["positive_base_indices"].tolist() + route_selection["hard_negative_base_indices"].tolist() + route_selection["random_negative_base_indices"].tolist())
    remaining = np.asarray([index for index in remaining_after_expert if int(index) not in route_used], dtype=np.int64)
    calibration_selection = _stratified_triplets(remaining, labels, identities, args.pairs_per_combo, 281000 + seed)
    scale_positions, fit_positions, threshold_positions = _split_three_by_combo(calibration_selection, labels, 282000 + seed)
    development = _development_pairs(plan["early_stop_indices"], labels, identities, 162000 + seed)
    all_calibration_ids = set(calibration_selection["positive_identities"].tolist() + calibration_selection["hard_negative_identities"].tolist())
    development_ids = set(development["positive_identities"].tolist() + development["hard_negative_identities"].tolist())
    if all_calibration_ids.intersection(development_ids):
        raise ValueError("Calibration and development identities overlap")
    print("seed={} expert={} extracting visibility roles".format(seed, expert_path.name), flush=True)

    def base_group(indices):
        return _expert_evidence(expert, _extract_base_indices(
            args, config, backbone, base_head, h2_checkpoint, indices, device
        ), device)

    def power_group(identity_values, level):
        return _expert_evidence(expert, _extract_power_identities(
            args, config, backbone, base_head, h2_checkpoint,
            power_files[int(level)], level_info[int(level)], identity_values, device
        ), device)

    role_groups = {}
    for role, positions in (("scale", scale_positions), ("fit", fit_positions), ("threshold", threshold_positions)):
        role_groups[role] = {
            "root_positive": base_group(calibration_selection["positive_base_indices"][positions]),
            "root_negative": base_group(calibration_selection["hard_negative_base_indices"][positions]),
            "power": {
                str(level): {
                    "positive": power_group(calibration_selection["positive_identities"][positions], level),
                    "negative": power_group(calibration_selection["hard_negative_identities"][positions], level),
                }
                for level in args.levels
            },
        }
    ordinary = {
        "calibration": base_group(plan["calibration_indices"]),
        "threshold": base_group(plan["threshold_indices"]),
        "early": base_group(plan["early_stop_indices"]),
    }
    development_power = {
        str(level): {
            "positive": power_group(development["positive_identities"], level),
            "negative": power_group(development["hard_negative_identities"], level),
        }
        for level in args.levels
    }
    scale_groups = [(role_groups["scale"]["root_positive"], role_groups["scale"]["root_negative"])] + [
        (role_groups["scale"]["power"][str(level)]["positive"], role_groups["scale"]["power"][str(level)]["negative"])
        for level in args.levels
    ]
    expert_coefficient, expert_intercept = _fit_expert_scale(scale_groups)
    for payload in list(ordinary.values()) + list(development_power.values()):
        if "positive" in payload:
            payload["positive"] = _add_c2_features(payload["positive"], expert_coefficient, expert_intercept)
            payload["negative"] = _add_c2_features(payload["negative"], expert_coefficient, expert_intercept)
        else:
            updated = _add_c2_features(payload, expert_coefficient, expert_intercept)
            payload.clear(); payload.update(updated)
    for role in role_groups.values():
        role["root_positive"] = _add_c2_features(role["root_positive"], expert_coefficient, expert_intercept)
        role["root_negative"] = _add_c2_features(role["root_negative"], expert_coefficient, expert_intercept)
        for level in args.levels:
            role["power"][str(level)]["positive"] = _add_c2_features(role["power"][str(level)]["positive"], expert_coefficient, expert_intercept)
            role["power"][str(level)]["negative"] = _add_c2_features(role["power"][str(level)]["negative"], expert_coefficient, expert_intercept)

    fit_groups = [(ordinary["calibration"], labels[plan["calibration_indices"], SAW_INDEX].astype(np.uint8))]
    for group in (role_groups["fit"]["root_positive"], role_groups["fit"]["root_negative"]):
        fit_groups.append((group, np.ones(len(group["c2"]), dtype=np.uint8) if group is role_groups["fit"]["root_positive"] else np.zeros(len(group["c2"]), dtype=np.uint8)))
    for level in args.levels:
        fit_groups.extend([
            (role_groups["fit"]["power"][str(level)]["positive"], np.ones(len(fit_positions), dtype=np.uint8)),
            (role_groups["fit"]["power"][str(level)]["negative"], np.zeros(len(fit_positions), dtype=np.uint8)),
        ])
    score_groups = [(group["c2"], targets) for group, targets in fit_groups]
    uniform = _fit_uniform_platt(score_groups)
    isotonic = _fit_isotonic(score_groups)
    threshold_groups = (
        (ordinary["threshold"], labels[plan["threshold_indices"], SAW_INDEX].astype(np.uint8)),
        [(level, role_groups["threshold"]["power"][str(level)]["positive"], role_groups["threshold"]["power"][str(level)]["negative"]) for level in args.levels],
    )
    visibility_model, feature_mean, feature_std, visibility_threshold, best_epoch, history = _train_visibility(
        args, fit_groups, threshold_groups, seed, device
    )
    visibility_bundle = (visibility_model, feature_mean, feature_std)
    seed_root = args.output_root / "seed_{}".format(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    _write(seed_root / "checkpoint_selection.json", selection_manifest)
    _write_history(seed_root / "visibility_history.csv", history)
    visibility_path = seed_root / "best_visibility_calibrator.pth"
    torch.save({
        "model_state_dict": visibility_model.state_dict(), "feature_mean": feature_mean,
        "feature_std": feature_std, "best_epoch": best_epoch,
        "expert_coefficient": expert_coefficient, "expert_intercept": expert_intercept,
        "selected_expert_sha256": _sha256(expert_path),
    }, visibility_path)

    variant_thresholds = {}
    threshold_searches = {}
    for variant in VARIANTS:
        ordinary_probability = _calibrator_probability(variant, ordinary["threshold"], uniform, isotonic, visibility_bundle, device)
        power_probability = {}
        for level in args.levels:
            power_probability[str(level)] = {
                "positive": _calibrator_probability(variant, role_groups["threshold"]["power"][str(level)]["positive"], uniform, isotonic, visibility_bundle, device),
                "negative": _calibrator_probability(variant, role_groups["threshold"]["power"][str(level)]["negative"], uniform, isotonic, visibility_bundle, device),
            }
        threshold, search = _robust_threshold(
            ordinary_probability, labels[plan["threshold_indices"], SAW_INDEX], power_probability, args.max_negative_fpr
        )
        if variant == "V2_VisibilityAffine":
            threshold = visibility_threshold
        variant_thresholds[variant] = float(threshold)
        threshold_searches[variant] = search
    _write(seed_root / "threshold_searches.json", {"thresholds": variant_thresholds, "searches": threshold_searches})

    rows = []
    for variant in VARIANTS:
        threshold = variant_thresholds[variant]
        ordinary_probability = _calibrator_probability(variant, ordinary["early"], uniform, isotonic, visibility_bundle, device)
        ordinary_target = labels[plan["early_stop_indices"], SAW_INDEX].astype(bool)
        power_metrics = {}
        oracle_recalls = []
        for level in args.levels:
            positive_probability = _calibrator_probability(variant, development_power[str(level)]["positive"], uniform, isotonic, visibility_bundle, device)
            negative_probability = _calibrator_probability(variant, development_power[str(level)]["negative"], uniform, isotonic, visibility_bundle, device)
            metrics = _decision_metrics(positive_probability, negative_probability, threshold)
            metrics["oracle_recall_at_fpr"] = _oracle_recall(positive_probability, negative_probability, args.max_negative_fpr)
            metrics["oracle_gap"] = metrics["oracle_recall_at_fpr"] - metrics["positive_recall"]
            power_metrics[str(level)] = metrics
            oracle_recalls.append(metrics["oracle_gap"])
        recalls = [power_metrics[str(level)]["positive_recall"] for level in args.levels]
        fprs = [power_metrics[str(level)]["hard_negative_fpr"] for level in args.levels]
        row = {
            "variant": variant, "seed": int(seed), "threshold": threshold,
            "ordinary_saw_recall": float((ordinary_probability[ordinary_target] >= threshold).mean()),
            "ordinary_saw_fpr": float((ordinary_probability[~ordinary_target] >= threshold).mean()),
            "power": power_metrics, "mean_power_recall": float(np.mean(recalls)),
            "min_power_recall": float(np.min(recalls)), "max_power_fpr": float(np.max(fprs)),
            "mean_oracle_gap": float(np.mean(oracle_recalls)),
            "expert_coefficient": expert_coefficient, "expert_intercept": expert_intercept,
            "selected_expert_sha256": _sha256(expert_path),
            "visibility_calibrator_sha256": _sha256(visibility_path),
            "split_sha256": plan["split_sha256"], "calibration_selection_sha256": calibration_selection["selection_sha256"],
            "identity_overlap": 0, "locked_test_used": False,
        }
        output_dir = seed_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_payload = {"ordinary_probability": ordinary_probability, "ordinary_target": ordinary_target}
        for level in args.levels:
            prediction_payload["positive_{}db".format(level)] = _calibrator_probability(variant, development_power[str(level)]["positive"], uniform, isotonic, visibility_bundle, device)
            prediction_payload["negative_{}db".format(level)] = _calibrator_probability(variant, development_power[str(level)]["negative"], uniform, isotonic, visibility_bundle, device)
        np.savez_compressed(str(output_dir / "predictions.npz"), **prediction_payload)
        _write(output_dir / "metrics.json", row)
        _write(output_dir / "run_receipt.json", {
            "status": "completed", "variant": variant, "seed": int(seed),
            "selected_expert_sha256": _sha256(expert_path), "visibility_calibrator_sha256": _sha256(visibility_path),
            "split_sha256": plan["split_sha256"], "identity_overlap": 0, "locked_test_used": False,
        })
        rows.append(row)
    audit = {
        "seed": int(seed), "scale_pairs": int(len(scale_positions)), "fit_pairs": int(len(fit_positions)),
        "threshold_pairs": int(len(threshold_positions)), "development_pairs": int(len(development["positive_identities"])),
        "selected_expert_path": str(expert_path), "selected_expert_sha256": _sha256(expert_path),
        "f2_checkpoint_sha256": _sha256(f2_path), "h2_checkpoint_sha256": _sha256(h2_path),
        "identity_overlap": 0, "locked_test_used": False,
    }
    _write(seed_root / "identity_and_source_audit.json", audit)
    return rows, audit, selection_manifest


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base labels and identities are misaligned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits, selections = [], [], []
    for seed in args.seeds:
        seed_rows, audit, selection = _run_seed(args, config, labels, identities, int(seed), device)
        rows.extend(seed_rows); audits.append(audit); selections.append({"seed": int(seed), **selection})
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: _mean_ci([row[metric] for row in selected], 291000 + variant_index * 100 + metric_index)
            for metric_index, metric in enumerate(("ordinary_saw_recall", "ordinary_saw_fpr", "mean_power_recall", "min_power_recall", "max_power_fpr", "mean_oracle_gap"))
        }
        aggregate[variant]["by_level"] = {
            str(level): {
                metric: _mean_ci([row["power"][str(level)][metric] for row in selected], 292000 + variant_index * 1000 + int(level) * 10 + metric_index)
                for metric_index, metric in enumerate(("positive_recall", "hard_negative_fpr", "oracle_recall_at_fpr", "oracle_gap"))
            }
            for level in args.levels
        }
    v0 = {row["seed"]: row for row in rows if row["variant"] == "V0_UniformPlatt"}
    v2 = [row for row in rows if row["variant"] == "V2_VisibilityAffine"]
    per_seed = []
    for row in v2:
        baseline = v0[row["seed"]]
        passed = (
            row["mean_power_recall"] >= 0.50 and row["min_power_recall"] >= 0.35
            and row["max_power_fpr"] <= 0.20 + 1e-12
            and row["ordinary_saw_fpr"] - baseline["ordinary_saw_fpr"] <= 0.02 + 1e-12
            and row["ordinary_saw_recall"] - baseline["ordinary_saw_recall"] >= -0.05 - 1e-12
        )
        per_seed.append({"seed": row["seed"], "passed": bool(passed)})
    best_baseline_recall = max(aggregate["V0_UniformPlatt"]["mean_power_recall"]["mean"], aggregate["V1_Isotonic"]["mean_power_recall"]["mean"])
    checks = {
        "mean_power_recall_0p50": aggregate["V2_VisibilityAffine"]["mean_power_recall"]["mean"] >= 0.50,
        "each_level_recall_0p35": all(aggregate["V2_VisibilityAffine"]["by_level"][str(level)]["positive_recall"]["mean"] >= 0.35 for level in args.levels),
        "each_level_fpr_0p20": all(aggregate["V2_VisibilityAffine"]["by_level"][str(level)]["hard_negative_fpr"]["mean"] <= 0.20 for level in args.levels),
        "ordinary_fpr_delta_0p02": aggregate["V2_VisibilityAffine"]["ordinary_saw_fpr"]["mean"] - aggregate["V0_UniformPlatt"]["ordinary_saw_fpr"]["mean"] <= 0.02,
        "ordinary_recall_delta_minus_0p05": aggregate["V2_VisibilityAffine"]["ordinary_saw_recall"]["mean"] - aggregate["V0_UniformPlatt"]["ordinary_saw_recall"]["mean"] >= -0.05,
        "two_seed_pass": sum(row["passed"] for row in per_seed) >= 2,
        "mean_oracle_gap_0p15": aggregate["V2_VisibilityAffine"]["mean_oracle_gap"]["mean"] <= 0.15,
        "beats_best_scalar_baseline": aggregate["V2_VisibilityAffine"]["mean_power_recall"]["mean"] >= best_baseline_recall,
        "identity_checkpoint_locked_test_audit": all(a["identity_overlap"] == 0 and not a["locked_test_used"] for a in audits) and all(not s["early_development_used"] and not s["locked_test_used"] for s in selections),
    }
    gate = {"checks": checks, "per_seed": per_seed, "passed": all(checks.values())}
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {"seeds": list(args.seeds), "levels": list(args.levels), "variants": list(VARIANTS), "pairs_per_combo": args.pairs_per_combo, "visibility_parameters": 22, "epochs": args.epochs, "max_negative_fpr": args.max_negative_fpr},
        "checkpoint_selections": selections, "audits": audits, "per_run": rows,
        "aggregate": aggregate, "promotion_gate": gate,
        "next_decision": "run_last_stage_small_lr_unfreeze" if gate["passed"] else "keep_backbone_frozen_and_revise_visibility_decision",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_stable_expert_visibility_calibration_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--stability-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_s2_initialization_stability")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration")
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
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
