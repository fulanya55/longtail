"""Route the frozen SawFM stage2 expert into H2 without opening locked test."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.evidence_readout import (  # noqa: E402
    SawEvidenceRouter,
    SawStage2LowRankEvidence,
)
from experiments.analyze_route_a_pam_iq_v2_scores import _component_power  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _evaluate_variant  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_label_local_saw import (  # noqa: E402
    F2_MODEL_NAME,
    _load_h2_head,
)
from experiments.run_route_a_sawfm_stage2_power_teacher_student import (  # noqa: E402
    _binary_metrics,
    _development_pairs,
    _extract_level,
    _indices_for_identities,
    _level_lookup,
    _model_scores,
    _sha256_values,
    _stratified_triplets,
)
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


SAW_INDEX = 1
OTHER_INDICES = (0, 2, 3, 4, 5)
SWEPT_INDICES = (3, 4, 5)
FIXED_INDICES = (0, 2)
VARIANTS = ("R0_H2", "R1_ExpertReplace", "R2_FixedBlend", "R3_LowCapacityRouter")


def _concat_extracted(first, second):
    return {
        key: np.concatenate([first[key], second[key]], axis=0)
        for key in first
    }


def _load_expert(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SawStage2LowRankEvidence(channels=64, rank=8, hidden_channels=32, topk=16)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _expert_scores(model, extracted, device):
    result = dict(extracted)
    result["expert"] = _model_scores(model, extracted["features"], device)
    return result


def _extract_base_indices(args, config, backbone, base_head, h2_checkpoint, indices, device):
    return _extract_level(
        backbone,
        _data_path(config),
        np.asarray(indices, dtype=np.int64),
        config["normalization"],
        device,
        args.feature_batch_size,
        base_head=base_head,
        main_mean=h2_checkpoint["main_mean"],
        main_std=h2_checkpoint["main_std"],
    )


def _extract_power_identities(
    args, config, backbone, base_head, h2_checkpoint, power_path, level_info, identities, device
):
    indices = _indices_for_identities(level_info, identities)
    return _extract_level(
        backbone,
        power_path,
        indices,
        config["normalization"],
        device,
        args.feature_batch_size,
        base_head=base_head,
        main_mean=h2_checkpoint["main_mean"],
        main_std=h2_checkpoint["main_std"],
    )


def _fit_platt(expert, targets, domain):
    expert = np.asarray(expert, dtype=np.float64).reshape(-1, 1)
    targets = np.asarray(targets, dtype=np.uint8)
    domain = np.asarray(domain, dtype=np.uint8)
    weights = np.zeros(len(targets), dtype=np.float64)
    for value in (0, 1):
        mask = domain == value
        weights[mask] = len(targets) / (2.0 * max(int(mask.sum()), 1))
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(expert, targets, sample_weight=weights)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _calibrate_expert(values, coefficient, intercept):
    return coefficient * np.asarray(values, dtype=np.float32) + intercept


def _router_features(base_logits, expert_logits):
    base = np.asarray(base_logits, dtype=np.float32)
    h2 = base[:, SAW_INDEX]
    expert = np.asarray(expert_logits, dtype=np.float32).reshape(-1)
    difference = expert - h2
    return np.stack(
        [
            h2,
            expert,
            difference,
            np.abs(difference),
            base[:, list(OTHER_INDICES)].max(axis=1),
            base[:, list(OTHER_INDICES)].mean(axis=1),
            base[:, list(SWEPT_INDICES)].max(axis=1),
            base[:, list(FIXED_INDICES)].max(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def _router_scores(model, base_logits, expert, mean, std, device):
    features = (_router_features(base_logits, expert) - mean) / std
    with torch.no_grad():
        fused, gate = model(
            torch.from_numpy(features).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(base_logits[:, SAW_INDEX])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(expert)).to(device=device, dtype=torch.float32),
            return_gate=True,
        )
    return fused.cpu().numpy(), gate.cpu().numpy()


def _score_candidate(normal_positive, normal_negative, weak_positive, weak_negative):
    normal = _binary_metrics(normal_positive, normal_negative)
    weak = _binary_metrics(weak_positive, weak_negative)
    return normal, weak, 0.5 * normal["aupr"] + 0.5 * weak["auroc"]


def _fit_router(
    args,
    fit_base,
    fit_expert,
    fit_targets,
    fit_domain,
    validation,
    seed,
    device,
):
    raw_features = _router_features(fit_base, fit_expert)
    mean = raw_features.mean(axis=0, keepdims=True)
    std = np.maximum(raw_features.std(axis=0, keepdims=True), 1e-5)
    features = (raw_features - mean) / std
    torch.manual_seed(211000 + int(seed))
    model = SawEvidenceRouter(feature_dim=features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.router_learning_rate, weight_decay=args.router_weight_decay
    )
    feature_tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    h2_tensor = torch.from_numpy(fit_base[:, SAW_INDEX]).to(device=device, dtype=torch.float32)
    expert_tensor = torch.from_numpy(fit_expert).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(fit_targets.astype(np.float32)).to(device)
    domain_tensor = torch.from_numpy(fit_domain.astype(np.int64)).to(device)
    sample_weight = torch.empty_like(target_tensor)
    for value in (0, 1):
        mask = domain_tensor == value
        sample_weight[mask] = len(target_tensor) / (2.0 * max(int(mask.sum()), 1))
    normal_baseline, _, _ = _score_candidate(
        validation["normal_positive_base"][:, SAW_INDEX],
        validation["normal_negative_base"][:, SAW_INDEX],
        validation["weak_positive_base"][:, SAW_INDEX],
        validation["weak_negative_base"][:, SAW_INDEX],
    )
    history = []
    best_state = None
    best_epoch = 0
    best_objective = -float("inf")
    stale = 0
    for epoch in range(1, int(args.router_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fused, gate = model(feature_tensor, h2_tensor, expert_tensor, return_gate=True)
        losses = F.binary_cross_entropy_with_logits(fused, target_tensor, reduction="none")
        classification = (losses * sample_weight).mean()
        normal_penalty = gate[domain_tensor == 0].mean()
        loss = classification + args.normal_gate_penalty * normal_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        normal_positive, normal_positive_gate = _router_scores(
            model, validation["normal_positive_base"], validation["normal_positive_expert"], mean, std, device
        )
        normal_negative, normal_negative_gate = _router_scores(
            model, validation["normal_negative_base"], validation["normal_negative_expert"], mean, std, device
        )
        weak_positive, weak_positive_gate = _router_scores(
            model, validation["weak_positive_base"], validation["weak_positive_expert"], mean, std, device
        )
        weak_negative, weak_negative_gate = _router_scores(
            model, validation["weak_negative_base"], validation["weak_negative_expert"], mean, std, device
        )
        normal, weak, objective = _score_candidate(
            normal_positive, normal_negative, weak_positive, weak_negative
        )
        feasible = (
            normal["auroc"] >= normal_baseline["auroc"] - args.validation_tolerance
            and normal["aupr"] >= normal_baseline["aupr"] - args.validation_tolerance
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "classification_loss": float(classification.detach().cpu()),
                "normal_gate_penalty": float(normal_penalty.detach().cpu()),
                "normal_auroc": normal["auroc"],
                "normal_aupr": normal["aupr"],
                "weak_auroc": weak["auroc"],
                "weak_aupr": weak["aupr"],
                "objective": objective,
                "feasible": bool(feasible),
                "normal_gate_mean": float(np.mean(np.concatenate([normal_positive_gate, normal_negative_gate]))),
                "weak_gate_mean": float(np.mean(np.concatenate([weak_positive_gate, weak_negative_gate]))),
            }
        )
        if feasible and objective > best_objective + 1e-8:
            best_objective = objective
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.router_patience):
                break
    fallback = best_state is None
    if fallback:
        with torch.no_grad():
            model.gate.weight.zero_()
            model.gate.bias.fill_(-20.0)
        best_epoch = 0
    else:
        model.load_state_dict(best_state)
    return model, mean, std, history, best_epoch, fallback


def _select_fixed_alpha(validation, tolerance):
    baseline = _binary_metrics(
        validation["normal_positive_base"][:, SAW_INDEX],
        validation["normal_negative_base"][:, SAW_INDEX],
    )
    rows = []
    best = None
    for alpha in np.arange(0.0, 1.0001, 0.05):
        def blend(prefix):
            return (1.0 - alpha) * validation[prefix + "_base"][:, SAW_INDEX] + alpha * validation[prefix + "_expert"]
        normal, weak, objective = _score_candidate(
            blend("normal_positive"), blend("normal_negative"),
            blend("weak_positive"), blend("weak_negative"),
        )
        feasible = normal["auroc"] >= baseline["auroc"] - tolerance and normal["aupr"] >= baseline["aupr"] - tolerance
        row = {"alpha": float(alpha), "normal": normal, "weak": weak, "objective": objective, "feasible": bool(feasible)}
        rows.append(row)
        if feasible and (best is None or objective > best["objective"] + 1e-8):
            best = row
    return float(best["alpha"] if best is not None else 0.0), rows


def _variant_saw_scores(variant, base_logits, expert, alpha, router_bundle, device):
    if variant == "R0_H2":
        return base_logits[:, SAW_INDEX].copy(), np.zeros(len(base_logits), dtype=np.float32)
    if variant == "R1_ExpertReplace":
        return expert.copy(), np.ones(len(base_logits), dtype=np.float32)
    if variant == "R2_FixedBlend":
        return (1.0 - alpha) * base_logits[:, SAW_INDEX] + alpha * expert, np.full(len(base_logits), alpha, dtype=np.float32)
    model, mean, std = router_bundle
    return _router_scores(model, base_logits, expert, mean, std, device)


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_seed(args, config, labels, weak_mask, identities, seed, device):
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    receipt = _read(source_dir / "run_receipt.json")
    if receipt.get("locked_test_used") or plan["split_sha256"] != receipt["split_sha256"]:
        raise ValueError("F2 receipt invalid or split drifted for seed {}".format(seed))
    backbone, f2_checkpoint = _load_backbone(source_dir, F2_MODEL_NAME, device)
    base_head, h2_checkpoint, h2_checkpoint_path, _ = _load_h2_head(seed, args.frozen_head_root, device)
    expert_path = args.expert_root / "seed_{}".format(seed) / "S2_TeacherHardNeg" / "best_saw_stage2_evidence.pth"
    expert, expert_checkpoint = _load_expert(expert_path, device)
    expert_selection = _stratified_triplets(plan["train_indices"], labels, identities, 40, 161000 + seed)
    excluded = set(
        expert_selection["positive_base_indices"].tolist()
        + expert_selection["hard_negative_base_indices"].tolist()
        + expert_selection["random_negative_base_indices"].tolist()
    )
    remaining_train = np.asarray([value for value in plan["train_indices"] if int(value) not in excluded], dtype=np.int64)
    ordinary_selection = _stratified_triplets(
        remaining_train, labels, identities, args.ordinary_pairs_per_combo, 201000 + seed
    )
    ordinary_order = np.random.default_rng(202000 + seed).permutation(len(ordinary_selection["positive_base_indices"]))
    ordinary_split = int(round(0.8 * len(ordinary_order)))
    ordinary_fit = ordinary_order[:ordinary_split]
    ordinary_validation = ordinary_order[ordinary_split:]
    weak_order = np.random.default_rng(163000 + seed).permutation(len(expert_selection["positive_identities"]))
    weak_split = int(round(0.8 * len(weak_order)))
    weak_fit = weak_order[:weak_split]
    weak_validation = weak_order[weak_split:]
    development = _development_pairs(plan["early_stop_indices"], labels, identities, 162000 + seed)
    expert_train_identities = set(
        expert_selection["positive_identities"].tolist()
        + expert_selection["hard_negative_identities"].tolist()
        + expert_selection["random_negative_identities"].tolist()
    )
    ordinary_train_identities = set(
        ordinary_selection["positive_identities"].tolist()
        + ordinary_selection["hard_negative_identities"].tolist()
        + ordinary_selection["random_negative_identities"].tolist()
    )
    development_identities = set(
        development["positive_identities"].tolist()
        + development["hard_negative_identities"].tolist()
    )
    expert_ordinary_overlap = expert_train_identities.intersection(ordinary_train_identities)
    train_development_overlap = (
        expert_train_identities.union(ordinary_train_identities)
    ).intersection(development_identities)
    if expert_ordinary_overlap or train_development_overlap:
        raise ValueError("Routing identity roles overlap")

    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    level_info = _level_lookup(power_files, (45,))[45]
    print("seed={} extracting frozen H2 and expert scores".format(seed), flush=True)

    ordinary_positive = _expert_scores(expert, _extract_base_indices(
        args, config, backbone, base_head, h2_checkpoint, ordinary_selection["positive_base_indices"], device
    ), device)
    ordinary_negative = _expert_scores(expert, _extract_base_indices(
        args, config, backbone, base_head, h2_checkpoint, ordinary_selection["hard_negative_base_indices"], device
    ), device)
    weak_positive = _expert_scores(expert, _extract_power_identities(
        args, config, backbone, base_head, h2_checkpoint, power_files[45], level_info, expert_selection["positive_identities"], device
    ), device)
    weak_negative = _expert_scores(expert, _extract_power_identities(
        args, config, backbone, base_head, h2_checkpoint, power_files[45], level_info, expert_selection["hard_negative_identities"], device
    ), device)
    weak_development_positive = _expert_scores(expert, _extract_power_identities(
        args, config, backbone, base_head, h2_checkpoint, power_files[45], level_info, development["positive_identities"], device
    ), device)
    weak_development_negative = _expert_scores(expert, _extract_power_identities(
        args, config, backbone, base_head, h2_checkpoint, power_files[45], level_info, development["hard_negative_identities"], device
    ), device)
    roles = {}
    role_indices = {
        "early": np.asarray(plan["early_stop_indices"], dtype=np.int64),
        "calibration": np.asarray(plan["calibration_indices"], dtype=np.int64),
        "threshold": np.asarray(plan["threshold_indices"], dtype=np.int64),
    }
    for role, indices in role_indices.items():
        roles[role] = _expert_scores(expert, _extract_base_indices(
            args, config, backbone, base_head, h2_checkpoint, indices, device
        ), device)

    fit_base = np.concatenate([
        ordinary_positive["base_logits"][ordinary_fit], ordinary_negative["base_logits"][ordinary_fit],
        weak_positive["base_logits"][weak_fit], weak_negative["base_logits"][weak_fit],
    ])
    fit_expert_raw = np.concatenate([
        ordinary_positive["expert"][ordinary_fit], ordinary_negative["expert"][ordinary_fit],
        weak_positive["expert"][weak_fit], weak_negative["expert"][weak_fit],
    ])
    fit_targets = np.concatenate([
        np.ones(len(ordinary_fit)), np.zeros(len(ordinary_fit)),
        np.ones(len(weak_fit)), np.zeros(len(weak_fit)),
    ]).astype(np.uint8)
    fit_domain = np.concatenate([
        np.zeros(2 * len(ordinary_fit)), np.ones(2 * len(weak_fit))
    ]).astype(np.uint8)
    platt_coefficient, platt_intercept = _fit_platt(fit_expert_raw, fit_targets, fit_domain)

    for group in (ordinary_positive, ordinary_negative, weak_positive, weak_negative, weak_development_positive, weak_development_negative, *roles.values()):
        group["expert_cal"] = _calibrate_expert(group["expert"], platt_coefficient, platt_intercept)
    fit_expert = _calibrate_expert(fit_expert_raw, platt_coefficient, platt_intercept)
    validation = {
        "normal_positive_base": ordinary_positive["base_logits"][ordinary_validation],
        "normal_negative_base": ordinary_negative["base_logits"][ordinary_validation],
        "normal_positive_expert": ordinary_positive["expert_cal"][ordinary_validation],
        "normal_negative_expert": ordinary_negative["expert_cal"][ordinary_validation],
        "weak_positive_base": weak_positive["base_logits"][weak_validation],
        "weak_negative_base": weak_negative["base_logits"][weak_validation],
        "weak_positive_expert": weak_positive["expert_cal"][weak_validation],
        "weak_negative_expert": weak_negative["expert_cal"][weak_validation],
    }
    alpha, alpha_rows = _select_fixed_alpha(validation, args.validation_tolerance)
    router, router_mean, router_std, history, best_epoch, fallback = _fit_router(
        args, fit_base, fit_expert, fit_targets, fit_domain, validation, seed, device
    )
    seed_root = args.output_root / "seed_{}".format(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    _write_history(seed_root / "router_history.csv", history)
    _write(seed_root / "fixed_alpha_search.json", {"selected_alpha": alpha, "candidates": alpha_rows})
    router_path = seed_root / "best_router.pth"
    torch.save(
        {
            "model_state_dict": router.state_dict(),
            "feature_mean": router_mean,
            "feature_std": router_std,
            "best_epoch": best_epoch,
            "fallback_to_h2": fallback,
            "platt_coefficient": platt_coefficient,
            "platt_intercept": platt_intercept,
        },
        router_path,
    )
    targets_by_role = {role: labels[indices] for role, indices in role_indices.items()}
    weak_by_role = {role: weak_mask[indices] for role, indices in role_indices.items()}
    router_bundle = (router, router_mean, router_std)
    rows = []
    for variant in VARIANTS:
        logits_by_role = {}
        gate_by_role = {}
        for role, group in roles.items():
            saw, gate = _variant_saw_scores(
                variant, group["base_logits"], group["expert_cal"], alpha, router_bundle, device
            )
            logits = group["base_logits"].copy()
            logits[:, SAW_INDEX] = saw
            logits_by_role[role] = logits
            gate_by_role[role] = gate
        output_dir = seed_root / variant
        ordinary_metrics = _evaluate_variant(
            variant, logits_by_role, targets_by_role, weak_by_role, output_dir, args
        )
        weak_positive_scores, weak_positive_gate = _variant_saw_scores(
            variant, weak_development_positive["base_logits"], weak_development_positive["expert_cal"], alpha, router_bundle, device
        )
        weak_negative_scores, weak_negative_gate = _variant_saw_scores(
            variant, weak_development_negative["base_logits"], weak_development_negative["expert_cal"], alpha, router_bundle, device
        )
        weak_metrics = _binary_metrics(weak_positive_scores, weak_negative_scores)
        row = dict(ordinary_metrics)
        row.update(
            {
                "seed": int(seed),
                "weak45_auroc": weak_metrics["auroc"],
                "weak45_aupr": weak_metrics["aupr"],
                "normal_early_gate_mean": float(np.mean(gate_by_role["early"])),
                "weak45_gate_mean": float(np.mean(np.concatenate([weak_positive_gate, weak_negative_gate]))),
                "gate_domain_gap": float(np.mean(np.concatenate([weak_positive_gate, weak_negative_gate])) - np.mean(gate_by_role["early"])),
                "selected_alpha": alpha if variant == "R2_FixedBlend" else None,
                "router_best_epoch": best_epoch if variant == "R3_LowCapacityRouter" else None,
                "router_fallback": fallback if variant == "R3_LowCapacityRouter" else None,
                "split_sha256": plan["split_sha256"],
                "ordinary_selection_sha256": ordinary_selection["selection_sha256"],
                "expert_selection_sha256": expert_selection["selection_sha256"],
                "weak_development_sha256": development["selection_sha256"],
                "f2_checkpoint_sha256": _sha256(f2_checkpoint),
                "h2_checkpoint_sha256": _sha256(h2_checkpoint_path),
                "expert_checkpoint_sha256": _sha256(expert_path),
                "router_checkpoint_sha256": _sha256(router_path),
                "identity_overlap": int(
                    len(expert_ordinary_overlap) + len(train_development_overlap)
                ),
                "locked_test_used": False,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(output_dir / "weak45_predictions.npz"),
            positive_scores=weak_positive_scores,
            negative_scores=weak_negative_scores,
            positive_gate=weak_positive_gate,
            negative_gate=weak_negative_gate,
        )
        _write(output_dir / "variant_summary.json", row)
        _write(output_dir / "run_receipt.json", {
            "status": "completed", "seed": int(seed), "variant": variant,
            "split_sha256": plan["split_sha256"], "router_checkpoint_sha256": _sha256(router_path),
            "identity_overlap": int(
                len(expert_ordinary_overlap) + len(train_development_overlap)
            ), "locked_test_used": False,
        })
        rows.append(row)
    audit = {
        "seed": int(seed),
        "ordinary_fit_pairs": int(len(ordinary_fit)),
        "ordinary_validation_pairs": int(len(ordinary_validation)),
        "weak_fit_pairs": int(len(weak_fit)),
        "weak_validation_pairs": int(len(weak_validation)),
        "weak_development_positive_count": int(len(development["positive_identities"])),
        "weak_development_negative_count": int(len(development["hard_negative_identities"])),
        "expert_train_ordinary_train_overlap": int(len(expert_ordinary_overlap)),
        "train_development_overlap": int(len(train_development_overlap)),
        "locked_test_used": False,
        "platt_coefficient": platt_coefficient,
        "platt_intercept": platt_intercept,
    }
    _write(seed_root / "sampling_and_routing_audit.json", audit)
    return rows, audit


def _aggregate(rows):
    metrics = ("macro_aupr", "macro_f1", "hamming_loss", "saw_aupr", "saw_negative_fpr", "weak45_auroc", "weak45_aupr", "normal_early_gate_mean", "weak45_gate_mean", "gate_domain_gap")
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            metric: _mean_ci([row[metric] for row in selected], 221000 + variant_index * 100 + metric_index)
            for metric_index, metric in enumerate(metrics)
        }
    return result


def _paired(rows, candidate, baseline):
    metrics = ("macro_aupr", "saw_aupr", "saw_negative_fpr", "weak45_auroc", "weak45_aupr")
    per_seed = []
    for seed in sorted({row["seed"] for row in rows}):
        left = next(row for row in rows if row["seed"] == seed and row["variant"] == candidate)
        right = next(row for row in rows if row["seed"] == seed and row["variant"] == baseline)
        item = {"seed": seed}
        item.update({metric + "_delta": left[metric] - right[metric] for metric in metrics})
        per_seed.append(item)
    return {
        "per_seed": per_seed,
        **{
            metric + "_delta": _mean_ci([row[metric + "_delta"] for row in per_seed], 223000 + index)
            for index, metric in enumerate(metrics)
        },
        "weak45_auroc_positive_seeds": int(sum(row["weak45_auroc_delta"] > 0 for row in per_seed)),
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
        labels, _component_power(parameters, ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"]), args.weak_threshold_db
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    audits = []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(args, config, labels, weak_mask, identities, int(seed), device)
        rows.extend(seed_rows)
        audits.append(audit)
    aggregate = _aggregate(rows)
    comparisons = {
        variant + "_vs_R0_H2": _paired(rows, variant, "R0_H2")
        for variant in VARIANTS[1:]
    }
    router_comparison = comparisons["R3_LowCapacityRouter_vs_R0_H2"]
    checks = {
        "weak45_auroc_0p70": aggregate["R3_LowCapacityRouter"]["weak45_auroc"]["mean"] >= 0.70,
        "weak45_auroc_gain_0p20": router_comparison["weak45_auroc_delta"]["mean"] >= 0.20,
        "weak45_gain_two_seeds": router_comparison["weak45_auroc_positive_seeds"] >= 2,
        "macro_aupr_delta_ge_minus_0p003": router_comparison["macro_aupr_delta"]["mean"] >= -0.003,
        "saw_aupr_delta_ge_minus_0p005": router_comparison["saw_aupr_delta"]["mean"] >= -0.005,
        "saw_fpr_delta_le_0p02": router_comparison["saw_negative_fpr_delta"]["mean"] <= 0.02,
        "gate_domain_gap_0p15": aggregate["R3_LowCapacityRouter"]["gate_domain_gap"]["mean"] >= 0.15,
        "identity_and_locked_test_audit": all(a["expert_train_ordinary_train_overlap"] == 0 and a["train_development_overlap"] == 0 and not a["locked_test_used"] for a in audits),
    }
    gate = {"checks": checks, "passed": all(checks.values())}
    payload = {
        "status": "completed",
        "device": str(device),
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "variants": list(VARIANTS),
            "ordinary_pairs_per_combo": args.ordinary_pairs_per_combo,
            "weak_level_db": 45, "router_parameter_count": 9,
            "router_learning_rate": args.router_learning_rate,
            "router_weight_decay": args.router_weight_decay,
            "router_epochs": args.router_epochs, "normal_gate_penalty": args.normal_gate_penalty,
            "validation_tolerance": args.validation_tolerance,
            "max_negative_fpr": args.max_negative_fpr,
        },
        "audits": audits,
        "per_run": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "promotion_gate": gate,
        "next_decision": "run_last_stage_small_lr_unfreeze" if gate["passed"] else "stop_before_unfreeze_and_revise_router",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_sawfm_expert_routing_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_expert_routing")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--ordinary-pairs-per-combo", type=int, default=20)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--router-learning-rate", type=float, default=1e-2)
    parser.add_argument("--router-weight-decay", type=float, default=1e-3)
    parser.add_argument("--router-epochs", type=int, default=300)
    parser.add_argument("--router-patience", type=int, default=40)
    parser.add_argument("--normal-gate-penalty", type=float, default=0.05)
    parser.add_argument("--validation-tolerance", type=float, default=0.005)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
