"""Five-seed Raw+Platt confirmation and frozen six-label routing audit."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, hamming_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_label_local_saw import _load_h2_head  # noqa: E402
from experiments.run_route_a_raw_frontend_independent_classifier_audit import (  # noqa: E402
    _binary_rank_metrics,
    _calibration_metrics,
    _decision_metrics,
    _fit_platt,
    _platt_probability,
    _raw_logits,
    _search_unified_threshold,
    _subgroup_metrics,
)
from experiments.run_route_a_raw_only_new_split_confirmation import _extract_raw  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import (  # noqa: E402
    _development_pairs,
    _indices_for_identities,
    _level_lookup,
    _stratified_triplets,
)
from experiments.run_route_a_stable_expert_visibility_calibration import (  # noqa: E402
    _split_three_by_combo,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
import experiments.run_route_a_dfms_iq_frozen_heads as frozen_heads  # noqa: E402
import experiments.run_route_a_dfms_iq_structure_screen as structure_screen  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
SAW_INDEX = 1
F2_MODEL_NAME = "F2_DFMS_Full"


def _component_power(values79):
    values = np.asarray(values79, dtype=np.float32)
    return np.stack(
        [values[:, get_parameter_specs(name)[0].column_index] for name in CLASS_NAMES],
        axis=1,
    )


def _ensure_f2_h2(args, config, labels, seeds, device):
    values79, _ = read_parameters(_data_path(config))
    weak_mask = compute_weak_component_mask(
        labels, _component_power(values79), args.weak_threshold_db
    )
    receipts = []
    for seed in seeds:
        f2_dir = args.screen_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
        f2_args = copy.copy(args)
        f2_args.output_root = args.screen_root
        f2_args.epochs = args.f2_epochs
        structure_screen._train(f2_args, f2_dir, F2_MODEL_NAME, int(seed))
        f2_receipt = _read(f2_dir / "run_receipt.json")
        if f2_receipt.get("status") != "completed" or f2_receipt.get("locked_test_used"):
            raise ValueError(f"invalid F2 receipt for seed {seed}")

        h2_dir = args.frozen_head_root / f"seed_{seed}" / "H2_PosBCE"
        original_variants = frozen_heads.VARIANTS
        try:
            frozen_heads.VARIANTS = {
                "H2_PosBCE": original_variants["H2_PosBCE"]
            }
            h2_args = copy.copy(args)
            h2_args.output_root = args.frozen_head_root
            h2_args.epochs = args.h2_epochs
            h2_args.patience = args.h2_patience
            frozen_heads._run_seed(
                h2_args, config, labels, weak_mask, int(seed), device
            )
        finally:
            frozen_heads.VARIANTS = original_variants
        h2_receipt = _read(h2_dir / "run_receipt.json")
        if h2_receipt.get("status") != "completed" or h2_receipt.get("locked_test_used"):
            raise ValueError(f"invalid H2 receipt for seed {seed}")
        receipts.append({
            "seed": int(seed),
            "f2_checkpoint_sha256": _sha256(f2_dir / "best_model.pth"),
            "f2_split_sha256": f2_receipt["split_sha256"],
            "h2_checkpoint_sha256": _sha256(h2_dir / "best_head.pth"),
            "h2_split_sha256": h2_receipt["split_sha256"],
            "locked_test_used": False,
        })
    return receipts


def _frontend_path(args, seed):
    if int(seed) in (2039, 2040):
        return (
            args.new_split_frontend_root / f"seed_{seed}"
            / "R0_FullPhysics" / "best_frontend.pth"
        )
    return (
        args.frontend_root / f"seed_{seed}"
        / "N0_RawOnly" / "best_frontend.pth"
    )


def _load_frontend(args, seed, device):
    path = _frontend_path(args, seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=args.hidden_channels, topk=args.topk,
        dropout=args.dropout, descriptor_mode="full",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


def _merge_raw(extracted, raw):
    result = dict(extracted)
    result.update(raw)
    return result


def _extract_base(args, config, backbone, h2_head, h2_checkpoint, indices, device):
    indices = np.asarray(indices, dtype=np.int64)
    extracted = stable._extract_base_indices(
        args, config, backbone, h2_head, h2_checkpoint, indices, device
    )
    raw = _extract_raw(
        _data_path(config), indices, config["normalization"], args.feature_batch_size
    )
    return _merge_raw(extracted, raw)


def _extract_power(
    args, config, backbone, h2_head, h2_checkpoint,
    power_path, level_info, identities, device,
):
    indices = _indices_for_identities(level_info, identities)
    extracted = stable._extract_power_identities(
        args, config, backbone, h2_head, h2_checkpoint,
        power_path, level_info, identities, device,
    )
    raw = _extract_raw(
        power_path, indices, config["normalization"], args.feature_batch_size
    )
    return _merge_raw(extracted, raw)


def _role_selection(args, labels, identities, plan, seed):
    expert = _stratified_triplets(
        plan["train_indices"], labels, identities, 40, 161000 + int(seed)
    )
    expert_used = set(
        expert["positive_base_indices"].tolist()
        + expert["hard_negative_base_indices"].tolist()
        + expert["random_negative_base_indices"].tolist()
    )
    remaining_after_expert = np.asarray(
        [index for index in plan["train_indices"] if int(index) not in expert_used],
        dtype=np.int64,
    )
    route = _stratified_triplets(
        remaining_after_expert, labels, identities, 20, 201000 + int(seed)
    )
    route_used = set(
        route["positive_base_indices"].tolist()
        + route["hard_negative_base_indices"].tolist()
        + route["random_negative_base_indices"].tolist()
    )
    remaining = np.asarray(
        [index for index in remaining_after_expert if int(index) not in route_used],
        dtype=np.int64,
    )
    calibration = _stratified_triplets(
        remaining, labels, identities, args.pairs_per_combo, 281000 + int(seed)
    )
    _, _, threshold_positions = _split_three_by_combo(
        calibration, labels, 282000 + int(seed)
    )
    development = _development_pairs(
        plan["early_stop_indices"], labels, identities, 162000 + int(seed)
    )
    calibration_ids = set(
        calibration["positive_identities"].tolist()
        + calibration["hard_negative_identities"].tolist()
    )
    development_ids = set(
        development["positive_identities"].tolist()
        + development["hard_negative_identities"].tolist()
    )
    overlap = len(calibration_ids.intersection(development_ids))
    if overlap:
        raise ValueError(f"calibration/development identity overlap for seed {seed}")
    return calibration, threshold_positions, development, overlap


def _power_probabilities(groups, score_fn, calibrate):
    return {
        level: {
            "positive": calibrate(score_fn(values["positive"])),
            "negative": calibrate(score_fn(values["negative"])),
        }
        for level, values in groups.items()
    }


def _multilabel_metrics(probability, targets, thresholds):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.uint8)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    prediction = probability >= thresholds.reshape(1, -1)
    target_bool = targets.astype(bool)
    per_label = {}
    for index, name in enumerate(CLASS_NAMES):
        positive = target_bool[:, index]
        negative = ~positive
        per_label[name] = {
            "aupr": float(average_precision_score(targets[:, index], probability[:, index])),
            "recall": float(prediction[positive, index].mean()),
            "fpr": float(prediction[negative, index].mean()),
            "ece_10bin": _calibration_metrics(probability[:, index], targets[:, index])["ece_10bin"],
        }
    return {
        "macro_aupr": float(average_precision_score(targets, probability, average="macro")),
        "micro_aupr": float(average_precision_score(targets, probability, average="micro")),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, prediction)),
        "exact_match": float(np.all(prediction == target_bool, axis=1).mean()),
        "per_label": per_label,
    }


def _route_metrics(args, seed, raw_probability, raw_threshold, raw_power):
    h2_dir = args.frozen_head_root / f"seed_{seed}" / "H2_PosBCE"
    saved = np.load(h2_dir / "predictions.npz")
    targets = saved["targets"].astype(np.uint8)
    h2_probability = saved["probabilities"].astype(np.float64)
    thresholds_payload = _read(h2_dir / "thresholds.json")
    h2_thresholds = np.asarray(
        thresholds_payload["modes"][thresholds_payload["primary_mode"]],
        dtype=np.float64,
    )
    if len(raw_probability) != len(h2_probability):
        raise ValueError(f"ordinary early length mismatch for seed {seed}")
    routed_probability = h2_probability.copy()
    routed_probability[:, SAW_INDEX] = raw_probability
    routed_thresholds = h2_thresholds.copy()
    routed_thresholds[SAW_INDEX] = float(raw_threshold)
    baseline = _multilabel_metrics(h2_probability, targets, h2_thresholds)
    routed = _multilabel_metrics(routed_probability, targets, routed_thresholds)
    other_fpr_delta = max(
        routed["per_label"][name]["fpr"] - baseline["per_label"][name]["fpr"]
        for index, name in enumerate(CLASS_NAMES) if index != SAW_INDEX
    )
    checks = {
        "macro_aupr_not_down": routed["macro_aupr"] >= baseline["macro_aupr"] - 1e-12,
        "macro_f1_improved": routed["macro_f1"] > baseline["macro_f1"],
        "saw_recall_gain_0p20": routed["per_label"]["SawFM"]["recall"] >= baseline["per_label"]["SawFM"]["recall"] + 0.20,
        "other_label_max_fpr_delta_0p01": other_fpr_delta <= 0.01 + 1e-12,
        "exact_match_delta_minus_0p01": routed["exact_match"] >= baseline["exact_match"] - 0.01 - 1e-12,
    }
    return {
        "seed": int(seed), "baseline": baseline, "routed": routed,
        "deltas": {
            "macro_aupr": routed["macro_aupr"] - baseline["macro_aupr"],
            "macro_f1": routed["macro_f1"] - baseline["macro_f1"],
            "saw_recall": routed["per_label"]["SawFM"]["recall"] - baseline["per_label"]["SawFM"]["recall"],
            "exact_match": routed["exact_match"] - baseline["exact_match"],
            "other_label_max_fpr": float(other_fpr_delta),
        },
        "power": raw_power, "checks": checks, "passed": bool(all(checks.values())),
        "locked_test_used": False,
    }, targets, h2_probability, routed_probability, h2_thresholds, routed_thresholds


def _run_seed(args, config, labels, identities, seed, device):
    plan = build_run_plan(config, "db", int(seed), 0, args.max_eval_samples)
    f2_dir = args.screen_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    receipt = _read(f2_dir / "run_receipt.json")
    if receipt["split_sha256"] != plan["split_sha256"] or receipt.get("locked_test_used"):
        raise ValueError(f"F2 split drift for seed {seed}")
    backbone, f2_path = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    h2_head, h2_checkpoint, h2_path, _ = _load_h2_head(
        int(seed), args.frozen_head_root, device
    )
    frontend, frontend_path = _load_frontend(args, int(seed), device)
    raw_score = lambda group: _raw_logits(frontend, group, device, args.batch_size)
    h2_score = lambda group: np.asarray(group["base_logits"][:, SAW_INDEX], dtype=np.float64)

    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    level_info = _level_lookup(power_files, args.levels)
    calibration_selection, threshold_positions, development, overlap = _role_selection(
        args, labels, identities, plan, int(seed)
    )

    ordinary_calibration = _extract_base(
        args, config, backbone, h2_head, h2_checkpoint,
        plan["calibration_indices"], device,
    )
    ordinary_threshold = _extract_base(
        args, config, backbone, h2_head, h2_checkpoint,
        plan["threshold_indices"], device,
    )
    ordinary_early = _extract_base(
        args, config, backbone, h2_head, h2_checkpoint,
        plan["early_stop_indices"], device,
    )
    threshold_power, development_power = {}, {}
    for level in args.levels:
        name = str(level)
        threshold_power[name] = {
            "positive": _extract_power(
                args, config, backbone, h2_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)],
                calibration_selection["positive_identities"][threshold_positions], device,
            ),
            "negative": _extract_power(
                args, config, backbone, h2_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)],
                calibration_selection["hard_negative_identities"][threshold_positions], device,
            ),
        }
        development_power[name] = {
            "positive": _extract_power(
                args, config, backbone, h2_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)],
                development["positive_identities"], device,
            ),
            "negative": _extract_power(
                args, config, backbone, h2_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)],
                development["hard_negative_identities"], device,
            ),
        }

    calibration_targets = labels[plan["calibration_indices"], SAW_INDEX].astype(np.uint8)
    threshold_targets = labels[plan["threshold_indices"], SAW_INDEX].astype(np.uint8)
    early_indices = np.asarray(plan["early_stop_indices"], dtype=np.int64)
    early_targets = labels[early_indices, SAW_INDEX].astype(np.uint8)
    early_labels = labels[early_indices]
    h2_calibrator = _fit_platt(h2_score(ordinary_calibration), calibration_targets)
    raw_calibrator = _fit_platt(raw_score(ordinary_calibration), calibration_targets)
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    _write(seed_root / "calibration.json", {
        "source_role": "ordinary_calibration",
        "sample_count": int(len(calibration_targets)),
        "positive_count": int(calibration_targets.sum()),
        "negative_count": int((1 - calibration_targets).sum()),
        "h2_platt": {
            "coefficient": float(h2_calibrator.coef_[0, 0]),
            "intercept": float(h2_calibrator.intercept_[0]),
        },
        "raw_platt": {
            "coefficient": float(raw_calibrator.coef_[0, 0]),
            "intercept": float(raw_calibrator.intercept_[0]),
        },
        "locked_test_used": False,
    })
    variants = {
        "I0_H2Platt": (h2_score, lambda values: _platt_probability(h2_calibrator, values)),
        "I1_RawPlatt": (raw_score, lambda values: _platt_probability(raw_calibrator, values)),
    }
    rows, prediction_cache = [], {}
    threshold_payload = {}
    for variant, (score_fn, calibrate) in variants.items():
        threshold_probability = calibrate(score_fn(ordinary_threshold))
        power_threshold_probability = _power_probabilities(
            threshold_power, score_fn, calibrate
        )
        selected, search = _search_unified_threshold(
            threshold_probability, threshold_targets,
            power_threshold_probability, args.grid_step, args.max_fpr,
        )
        if selected is None:
            raise RuntimeError(f"no feasible {variant} threshold for seed {seed}")
        threshold = float(selected["threshold"])
        ordinary_probability = calibrate(score_fn(ordinary_early))
        ordinary = {
            **_binary_rank_metrics(ordinary_probability, early_targets),
            **_decision_metrics(ordinary_probability, early_targets, threshold),
            **_calibration_metrics(ordinary_probability, early_targets),
        }
        ordinary["subgroups"] = _subgroup_metrics(
            ordinary_probability, early_targets, early_labels, threshold
        )
        power_probability = _power_probabilities(
            development_power, score_fn, calibrate
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
            "seed": int(seed), "variant": variant, "selected_threshold": threshold,
            "ordinary": ordinary, "power": power,
            "mean_power_recall": float(np.mean([item["recall"] for item in power.values()])),
            "worst_power_recall": float(np.min([item["recall"] for item in power.values()])),
            "max_power_fpr": float(np.max([item["fpr"] for item in power.values()])),
            "split_sha256": plan["split_sha256"], "identity_overlap": int(overlap),
            "locked_test_used": False,
        }
        rows.append(row)
        prediction_cache[variant] = {
            "ordinary_probability": ordinary_probability,
            "power_probability": power_probability,
        }
        threshold_payload[variant] = {"selected": selected, "rows": search}
        variant_dir = seed_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        _write(variant_dir / "metrics.json", row)
        np.savez_compressed(
            str(variant_dir / "predictions.npz"),
            ordinary_probability=ordinary_probability,
            ordinary_target=early_targets,
            ordinary_source_index=early_indices,
            **{
                f"{target}_{level}db": values[target]
                for level, values in power_probability.items()
                for target in ("positive", "negative")
            },
        )
    _write(seed_root / "threshold_search.json", threshold_payload)

    raw_row = next(row for row in rows if row["variant"] == "I1_RawPlatt")
    raw_prediction = prediction_cache["I1_RawPlatt"]
    route, route_targets, h2_probability, routed_probability, h2_thresholds, routed_thresholds = _route_metrics(
        args, int(seed), raw_prediction["ordinary_probability"],
        raw_row["selected_threshold"], raw_row["power"],
    )
    route_dir = seed_root / "label_routing"
    route_dir.mkdir(parents=True, exist_ok=True)
    _write(route_dir / "metrics.json", route)
    np.savez_compressed(
        str(route_dir / "predictions.npz"), targets=route_targets,
        h2_probability=h2_probability, routed_probability=routed_probability,
        h2_thresholds=h2_thresholds, routed_thresholds=routed_thresholds,
    )
    audit = {
        "seed": int(seed), "split_sha256": plan["split_sha256"],
        "f2_checkpoint": str(f2_path), "f2_checkpoint_sha256": _sha256(f2_path),
        "h2_checkpoint": str(h2_path), "h2_checkpoint_sha256": _sha256(h2_path),
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
        "frontend_protocol": "new_split_R0" if int(seed) in (2039, 2040) else "original_N0",
        "calibration_selection_sha256": calibration_selection["selection_sha256"],
        "development_selection_sha256": development["selection_sha256"],
        "identity_overlap": int(overlap), "locked_test_used": False,
    }
    _write(seed_root / "source_and_identity_audit.json", audit)
    _write(seed_root / "run_receipt.json", {
        "status": "completed", "seed": int(seed),
        "f2_checkpoint_sha256": audit["f2_checkpoint_sha256"],
        "h2_checkpoint_sha256": audit["h2_checkpoint_sha256"],
        "frontend_checkpoint_sha256": audit["frontend_checkpoint_sha256"],
        "split_sha256": plan["split_sha256"], "identity_overlap": int(overlap),
        "locked_test_used": False,
    })
    return rows, route, audit


def _aggregate_confirmation(rows, levels):
    result = {}
    for variant_index, variant in enumerate(("I0_H2Platt", "I1_RawPlatt")):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                711000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate((
                "mean_power_recall", "worst_power_recall", "max_power_fpr",
            ))
        }
        result[variant]["ordinary"] = {
            metric: _mean_ci(
                [row["ordinary"][metric] for row in selected],
                712000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate((
                "auroc", "aupr", "recall", "fpr", "brier", "ece_10bin",
            ))
        }
        result[variant]["ordinary"]["without_swept_recall"] = _mean_ci(
            [row["ordinary"]["subgroups"]["without_swept"]["recall"] for row in selected],
            713000 + variant_index,
        )
        result[variant]["by_level"] = {
            str(level): {
                metric: _mean_ci(
                    [row["power"][str(level)][metric] for row in selected],
                    714000 + variant_index * 1000 + int(level) * 10 + metric_index,
                )
                for metric_index, metric in enumerate(("auroc", "aupr", "recall", "fpr"))
            }
            for level in levels
        }
    return result


def _confirmation_gate(rows, aggregate, levels, audits):
    h2 = {row["seed"]: row for row in rows if row["variant"] == "I0_H2Platt"}
    raw = [row for row in rows if row["variant"] == "I1_RawPlatt"]
    per_seed = []
    for row in raw:
        baseline = h2[row["seed"]]
        checks = {
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
        per_seed.append({"seed": row["seed"], "checks": checks, "passed": bool(all(checks.values()))})
    raw_aggregate = aggregate["I1_RawPlatt"]
    aggregate_checks = {
        "ordinary_aupr_0p90": raw_aggregate["ordinary"]["aupr"]["mean"] >= 0.90,
        "ordinary_recall_0p80": raw_aggregate["ordinary"]["recall"]["mean"] >= 0.80,
        "mean_power_recall_0p90": raw_aggregate["mean_power_recall"]["mean"] >= 0.90,
        "50db_recall_0p85": raw_aggregate["by_level"]["50"]["recall"]["mean"] >= 0.85,
        "without_swept_recall_0p70": raw_aggregate["ordinary"]["without_swept_recall"]["mean"] >= 0.70,
        "four_of_five_seed_pass": sum(item["passed"] for item in per_seed) >= 4,
        "identity_checkpoint_locked_test_audit": all(
            audit["identity_overlap"] == 0 and not audit["locked_test_used"]
            for audit in audits
        ),
    }
    return {
        "per_seed": per_seed, "aggregate_checks": aggregate_checks,
        "passed_seed_count": int(sum(item["passed"] for item in per_seed)),
        "passed": bool(all(aggregate_checks.values())),
    }


def _aggregate_routes(routes):
    result = {}
    for role in ("baseline", "routed"):
        result[role] = {
            metric: _mean_ci(
                [row[role][metric] for row in routes],
                721000 + (0 if role == "baseline" else 100) + metric_index,
            )
            for metric_index, metric in enumerate((
                "macro_aupr", "micro_aupr", "macro_f1", "hamming_loss", "exact_match",
            ))
        }
        result[role]["saw_recall"] = _mean_ci(
            [row[role]["per_label"]["SawFM"]["recall"] for row in routes],
            722000 + (0 if role == "baseline" else 100),
        )
        result[role]["saw_fpr"] = _mean_ci(
            [row[role]["per_label"]["SawFM"]["fpr"] for row in routes],
            723000 + (0 if role == "baseline" else 100),
        )
    result["deltas"] = {
        metric: _mean_ci(
            [row["deltas"][metric] for row in routes], 724000 + index
        )
        for index, metric in enumerate((
            "macro_aupr", "macro_f1", "saw_recall", "exact_match", "other_label_max_fpr",
        ))
    }
    return result


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("base labels and identity labels drifted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completion = _ensure_f2_h2(
        args, config, labels, args.completion_seeds, device
    )
    rows, routes, audits = [], [], []
    for seed in args.seeds:
        seed_rows, route, audit = _run_seed(
            args, config, labels, identities, int(seed), device
        )
        rows.extend(seed_rows); routes.append(route); audits.append(audit)
    confirmation_aggregate = _aggregate_confirmation(rows, args.levels)
    confirmation_gate = _confirmation_gate(
        rows, confirmation_aggregate, args.levels, audits
    )
    routing_aggregate = _aggregate_routes(routes)
    routing_passed_seed_count = int(sum(row["passed"] for row in routes))
    routing_gate = {
        "per_seed": [{"seed": row["seed"], "checks": row["checks"], "passed": row["passed"]} for row in routes],
        "passed_seed_count": routing_passed_seed_count,
        "confirmation_prerequisite": confirmation_gate["passed"],
        "passed": bool(confirmation_gate["passed"] and routing_passed_seed_count >= 4),
    }
    if not confirmation_gate["passed"]:
        next_decision = "stop_at_three_seed_exploratory_raw_result"
    elif routing_gate["passed"]:
        next_decision = "freeze_raw_saw_label_routing_and_prepare_locked_test_request"
    else:
        next_decision = "keep_raw_as_sawfm_auxiliary_reviewer_do_not_replace_h2"
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "completion_seeds": list(args.completion_seeds),
            "levels": list(args.levels), "grid_step": args.grid_step,
            "max_fpr": args.max_fpr, "raw_calibration": "ordinary_platt",
            "label_routing": "replace_sawfm_only_keep_other_h2_labels",
        },
        "baseline_completion": completion, "audits": audits,
        "confirmation": {
            "per_run": rows, "aggregate": confirmation_aggregate,
            "promotion_gate": confirmation_gate,
        },
        "label_routing": {
            "per_run": routes, "aggregate": routing_aggregate,
            "promotion_gate": routing_gate,
        },
        "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "route_a_raw_platt_five_seed_label_routing_summary.json"
    _write(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--frontend-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_trainable_raw_stage1_sawfm_frontend")
    parser.add_argument("--new-split-frontend-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_only_new_split_confirmation")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_platt_five_seed_label_routing")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--completion-seeds", type=int, nargs="+", default=[2039, 2040])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-fpr", type=float, default=0.20)
    parser.add_argument("--f2-epochs", type=int, default=20)
    parser.add_argument("--h2-epochs", type=int, default=60)
    parser.add_argument("--h2-patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--rank-margin", type=float, default=0.20)
    parser.add_argument("--saw-expert-weight", type=float, default=0.20)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
