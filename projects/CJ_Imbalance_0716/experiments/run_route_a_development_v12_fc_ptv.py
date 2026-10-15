"""Preregistered development-v12 evaluation of FC-PTV."""

from __future__ import annotations

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

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.route_a_fc_ptv_core import (  # noqa: E402
    GROUP_NAMES,
    exact_functional_distances,
    functional_gram,
    robust_logit_scales,
    simultaneous_lower_bounds,
    solve_two_block_projection,
)
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    COMMON_INDICES,
    COMMON_NAMES,
    TARGET_NAMES,
    _indices_sha256,
    _read_json,
    _sha256,
    _write_json,
)
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _component_power,
    _predict,
)
from experiments.run_route_a_development_v4_sawfm_specificity import (  # noqa: E402
    MASKED_ENDPOINTS,
    OTHER_MASK_ENDPOINTS,
    _apply_direction,
    _build_roles,
    _direction_norm,
    _evaluate_model_v4,
    _load_model_from_state,
    _metric,
    _save_and_evaluate,
    _state_direction,
    _train_endpoint,
    _variant_map,
)
from experiments.run_route_a_development_v5_layer_alpha import (  # noqa: E402
    _compose_layer_state,
    _direction_add,
    _direction_scale,
    _orthogonal_task_subspace,
)
from experiments.run_route_a_development_v6_multitask_baselines import _train_baseline  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = (
    "C6_Joint",
    "M5_GradientIsolated",
    "FullAlpha1p5",
    "FC_PTV",
    "ParameterNormOnly",
    "ClassifierOnlyAlpha2p0",
    "ModelAverage0p5",
    "CAGrad0p4TwoTask",
    "OrthogonalTaskSubspaceFunctionalMatched",
)


def _safe_mean_ci(values, seed):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "count": 0,
            "bootstrap_95_ci": [float("nan"), float("nan")],
        }
    return _mean_ci(finite, seed)


def _validate_config(config):
    if bool(config.get("allow_locked_test", False)):
        raise ValueError("development-v12 requires allow_locked_test=false")
    if str(config.get("evidence_role")) != "development_v12":
        raise ValueError("evidence_role must be development_v12")
    if bool(config.get("test_v12_generated", True)) or bool(config.get("test_v12_access_authorized", True)) or bool(config.get("test_v12_used", True)):
        raise ValueError("test-v12 must remain absent and unused")
    if not bool(config.get("test_v3_sealed", False)) or bool(config.get("test_v3_used", True)):
        raise ValueError("test-v3 must remain sealed and unused")
    if not bool(config.get("test_v2_consumed", False)):
        raise ValueError("test-v2 must remain consumed")
    if bool(config.get("development_v11_reuse_for_selection", True)):
        raise ValueError("development-v11 reuse for selection must be false")
    data_path = (Path(str(config["data_root"])) / str(config["train_file"])).resolve()
    forbidden = {
        Path(str(config["test_v12_path"])).resolve(),
        Path(str(config["sealed_test_v3_path"])).resolve(),
        Path(str(config["forbidden_test_v2_path"])).resolve(),
    }
    if data_path in forbidden or "test" in data_path.name.lower():
        raise ValueError("development-v12 data resolves to a test role")


def _fit_logits(state, f2_dir, roles, device, args):
    model = _load_model_from_state(f2_dir, device, state)
    logits = _predict(model, roles["fit"]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logits = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("fit-only functional logits are non-finite")
    return logits


def _distance_payload(distances):
    return {
        GROUP_NAMES[group]: {name: float(distances[group, label]) for label, name in enumerate(COMMON_NAMES)}
        for group in range(2)
    }


def _direction_support(direction, tolerance=1e-12):
    support = []
    forbidden = []
    for key, value in direction.items():
        norm = float(torch.linalg.vector_norm(value.double()).cpu())
        if norm <= float(tolerance):
            continue
        support.append({"key": key, "norm": norm})
        if not (key.startswith("temporal_backbone.0.") or key.startswith("classifier.")):
            forbidden.append(key)
    if forbidden:
        raise ValueError(f"paired direction has non-preregistered support: {forbidden}")
    return support


def _paired(seed_rows, candidate, baseline, bootstrap_seed):
    paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    per_seed = []
    for seed_row in seed_rows:
        variants = _variant_map(seed_row)
        left, right = variants[candidate], variants[baseline]
        if left.get("metrics") is None or right.get("metrics") is None:
            continue
        delta = {name: _metric(left, path) - _metric(right, path) for name, path in paths.items()}
        delta["per_label_aupr"] = {
            name: left["metrics"]["per_label_diagnostics"][name]["aupr"] - right["metrics"]["per_label_diagnostics"][name]["aupr"]
            for name in COMMON_NAMES
        }
        delta["per_label_weak_recall"] = {
            name: left["metrics"]["per_label_diagnostics"][name]["weak_recall"] - right["metrics"]["per_label_diagnostics"][name]["weak_recall"]
            for name in TARGET_NAMES
        }
        per_seed.append({"seed": int(seed_row["seed"]), "delta": delta})
    aggregate = {
        name: _safe_mean_ci([row["delta"][name] for row in per_seed], bootstrap_seed + index)
        for index, name in enumerate(paths)
    }
    aggregate["per_label_aupr"] = {
        name: _safe_mean_ci([row["delta"]["per_label_aupr"][name] for row in per_seed], bootstrap_seed + 100 + index)
        for index, name in enumerate(COMMON_NAMES)
    }
    aggregate["per_label_weak_recall"] = {
        name: _safe_mean_ci([row["delta"]["per_label_weak_recall"][name] for row in per_seed], bootstrap_seed + 200 + index)
        for index, name in enumerate(TARGET_NAMES)
    }
    aggregate["positive_target_weak_recall_seeds"] = sum(row["delta"]["target_mean_weak_recall"] > 0 for row in per_seed)
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate, "paired_seed_count": len(per_seed)}


def _aggregate(seed_rows):
    paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    aggregate = {}
    for variant_index, name in enumerate(VARIANTS):
        rows = [_variant_map(seed_row)[name] for seed_row in seed_rows]
        metric_rows = [row for row in rows if row.get("metrics") is not None]
        aggregate[name] = {
            metric: _safe_mean_ci([_metric(row, path) for row in metric_rows], 20261200 + 100 * variant_index + index)
            for index, (metric, path) in enumerate(paths.items())
        }
        aggregate[name]["per_label_aupr"] = {
            label: _safe_mean_ci([row["metrics"]["per_label_diagnostics"][label]["aupr"] for row in metric_rows], 20261300 + 100 * variant_index + index)
            for index, label in enumerate(COMMON_NAMES)
        }
        aggregate[name]["per_label_weak_recall"] = {
            label: _safe_mean_ci([row["metrics"]["per_label_diagnostics"][label]["weak_recall"] for row in metric_rows], 20261400 + 100 * variant_index + index)
            for index, label in enumerate(TARGET_NAMES)
        }
        aggregate[name]["finite_metric_seed_count"] = len(metric_rows)
        aggregate[name]["valid_seed_count"] = sum(row.get("status") == "valid" for row in rows)
    comparisons = {name: _paired(seed_rows, name, "C6_Joint", 20262000 + 500 * index) for index, name in enumerate(VARIANTS) if name != "C6_Joint"}
    return aggregate, comparisons


def _candidate_gate(comparison, absolute):
    delta = comparison["aggregate"]
    checks = {
        "valid_5_of_5": comparison["paired_seed_count"] == 5 and absolute["valid_seed_count"] == 5,
        "macro_aupr_noninferior_minus_0p005": delta["macro_aupr"]["mean"] >= -0.005,
        "macro_f1_gain_0p010": delta["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_0p025": delta["target_mean_weak_recall"]["mean"] >= 0.025,
        "each_target_weak_nonnegative": all(delta["per_label_weak_recall"][name]["mean"] >= 0 for name in TARGET_NAMES),
        "target_fpr_cap_0p20": absolute["target_mean_negative_fpr"]["mean"] <= 0.20,
        "positive_target_weak_4_of_5": delta["positive_target_weak_recall_seeds"] >= 4,
        "sawfm_present_macro_f1_positive": delta["sawfm_present_macro_f1"]["mean"] > 0,
        "sawfm_absent_macro_f1_noninferior_minus_0p005": delta["sawfm_absent_macro_f1"]["mean"] >= -0.005,
        "target_aupr_noninferior_minus_0p010": all(delta["per_label_aupr"][name]["mean"] >= -0.010 for name in TARGET_NAMES),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _improvement_gate(seed_rows, comparisons, aggregate):
    paired = _paired(seed_rows, "FC_PTV", "FullAlpha1p5", 20267000)
    delta = paired["aggregate"]
    fc = comparisons["FC_PTV"]["aggregate"]
    full = comparisons["FullAlpha1p5"]["aggregate"]
    worst_margin = min(fc["per_label_weak_recall"][name]["mean"] for name in TARGET_NAMES) - min(full["per_label_weak_recall"][name]["mean"] for name in TARGET_NAMES)
    label_ci_advantage = any(
        delta["per_label_weak_recall"][name]["mean"] >= 0.005
        and delta["per_label_weak_recall"][name]["bootstrap_95_ci"][0] > 0
        for name in TARGET_NAMES
    )
    main_ci_advantage = (
        delta["target_mean_weak_recall"]["mean"] >= 0.005
        and delta["target_mean_weak_recall"]["bootstrap_95_ci"][0] > 0
    ) or label_ci_advantage
    checks = {
        "worst_target_improves_by_0p005": worst_margin >= 0.005,
        "target_mean_weak_not_lower": delta["target_mean_weak_recall"]["mean"] >= 0,
        "macro_f1_noninferior_minus_0p003": delta["macro_f1"]["mean"] >= -0.003,
        "macro_aupr_noninferior_minus_0p003": delta["macro_aupr"]["mean"] >= -0.003,
        "target_fpr_delta_le_0p005": delta["target_mean_negative_fpr"]["mean"] <= 0.005,
        "one_primary_advantage_positive_ci": main_ci_advantage,
        "weak_effect_std_lower": fc["target_mean_weak_recall"]["std"] < full["target_mean_weak_recall"]["std"],
        "valid_and_finite_5_of_5": aggregate["FC_PTV"]["valid_seed_count"] == 5 and paired["paired_seed_count"] == 5,
    }
    return {"comparison": paired, "worst_target_margin": float(worst_margin), "checks": checks, "passed": all(checks.values())}


def _control_advantage(seed_rows, control, seed):
    paired = _paired(seed_rows, "FC_PTV", control, seed)
    delta = paired["aggregate"]
    worst = min(delta["per_label_weak_recall"][name]["mean"] for name in TARGET_NAMES)
    worst_ci = min(delta["per_label_weak_recall"][name]["bootstrap_95_ci"][0] for name in TARGET_NAMES)
    mean_adv = delta["target_mean_weak_recall"]["mean"] >= 0.005 and delta["target_mean_weak_recall"]["bootstrap_95_ci"][0] > 0
    worst_adv = worst >= 0.005 and worst_ci > 0
    return paired, bool(mean_adv or worst_adv), float(worst), float(worst_ci)


def _mechanism_gate(seed_rows):
    parameter, parameter_adv, parameter_worst, parameter_ci = _control_advantage(seed_rows, "ParameterNormOnly", 20268000)
    classifier, classifier_adv, classifier_worst, classifier_ci = _control_advantage(seed_rows, "ClassifierOnlyAlpha2p0", 20268500)
    orthogonal = _paired(seed_rows, "FC_PTV", "OrthogonalTaskSubspaceFunctionalMatched", 20269000)
    orth_delta = orthogonal["aggregate"]
    q_positive = all(float(row["fc_ptv"]["q_star"][0]) > 0 for row in seed_rows)
    orth_match = all(bool(row["orthogonal_control"]["functional_match_passed"]) for row in seed_rows)
    checks = {
        "beats_parameter_norm_control": parameter_adv,
        "beats_classifier_only_control": classifier_adv and classifier["aggregate"]["macro_aupr"]["mean"] >= -0.003 and classifier["aggregate"]["macro_f1"]["mean"] >= -0.003,
        "beats_orthogonal_functional_control": orth_delta["target_mean_weak_recall"]["mean"] >= 0.005 and orth_delta["macro_f1"]["mean"] >= -0.003,
        "orthogonal_functional_match_5_of_5": orth_match,
        "shared_component_nonzero_5_of_5": q_positive,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "parameter_norm_comparison": parameter,
        "parameter_norm_worst_label_advantage": parameter_worst,
        "parameter_norm_worst_label_ci_lower": parameter_ci,
        "classifier_only_comparison": classifier,
        "classifier_only_worst_label_advantage": classifier_worst,
        "classifier_only_worst_label_ci_lower": classifier_ci,
        "orthogonal_comparison": orthogonal,
    }


def _simultaneous_gate(comparison):
    if comparison["paired_seed_count"] != 5:
        return {
            "labels": list(TARGET_NAMES),
            "bounds": None,
            "checks": {"paired_seed_count_5": False},
            "passed": False,
            "reason": "requires_five_finite_paired_seeds",
        }
    matrix = np.asarray([
        [row["delta"]["per_label_weak_recall"][name] for name in TARGET_NAMES]
        for row in comparison["per_seed"]
    ], dtype=np.float64)
    bounds = simultaneous_lower_bounds(matrix, bootstrap_samples=10000, seed=20261215)
    checks = {
        "all_target_means_positive": all(value > 0 for value in bounds["mean"]),
        "each_target_nonnegative_4_of_5": all(int(np.sum(matrix[:, index] >= 0)) >= 4 for index in range(matrix.shape[1])),
        "all_familywise_lower_bounds_positive": all(value > 0 for value in bounds["familywise_95_lower"]),
    }
    return {"labels": list(TARGET_NAMES), "bounds": bounds, "checks": checks, "passed": all(checks.values())}


def _run_seed(args, config, labels6, weak6, seed, device, data_sha256):
    seed_dir = Path(args.output_root) / f"seed_{seed}"
    summary_path = seed_dir / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return _read_json(summary_path)
    split, manifest, ids, roles = _build_roles(args, config, labels6, seed)
    f2_dir = Path(args.f2_root) / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    template, checkpoint_path = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial = copy.deepcopy({key: value.detach().cpu() for key, value in template.state_dict().items()})
    del template
    states, training, variants = {}, {}, []
    for variant, masked_index in [("C6_Joint", None), *MASKED_ENDPOINTS.items()]:
        model = _load_model_from_state(f2_dir, device, initial)
        model, training[variant] = _train_endpoint(model, variant, masked_index, roles, weak6, device, seed, args, seed_dir / variant)
        states[variant] = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if variant in ("C6_Joint", "M5_GradientIsolated"):
            row = _evaluate_model_v4(variant, model, roles, weak6, seed_dir, device, args)
            row["status"] = "valid"
            variants.append(row)
        del model
    cagrad_model = _load_model_from_state(f2_dir, device, initial)
    cagrad_model, training["CAGrad0p4TwoTask"] = _train_baseline(cagrad_model, "CAGrad0p4TwoTask", roles, weak6, device, seed, args, seed_dir / "CAGrad0p4TwoTask")
    cagrad_row = _evaluate_model_v4("CAGrad0p4TwoTask", cagrad_model, roles, weak6, seed_dir, device, args)
    cagrad_row["status"] = "valid"
    variants.append(cagrad_row)
    del cagrad_model

    base_state = states["C6_Joint"]
    direction = _state_direction(base_state, states["M5_GradientIsolated"])
    support = _direction_support(direction)
    shared_state = _compose_layer_state(base_state, direction, 1.0, 0.0)
    classifier_state = _compose_layer_state(base_state, direction, 0.0, 1.0)
    base_fit = _fit_logits(base_state, f2_dir, roles, device, args)
    shared_fit = _fit_logits(shared_state, f2_dir, roles, device, args)
    classifier_fit = _fit_logits(classifier_state, f2_dir, roles, device, args)
    saw_fit = roles["fit"]["targets6"][:, 1].astype(bool)
    scale_payload = robust_logit_scales(base_fit, saw_fit, min_scale=float(args.functional_min_scale))
    gram = functional_gram(base_fit, shared_fit, classifier_fit, saw_fit, scale_payload["scales"])
    q_star, solver = solve_two_block_projection(gram, q0=(1.5, 1.5), upper=1.5)
    fc_state = _compose_layer_state(base_state, direction, q_star[0], q_star[1])
    fc_fit = _fit_logits(fc_state, f2_dir, roles, device, args)
    fc_distances = exact_functional_distances(base_fit, fc_fit, saw_fit, scale_payload["scales"])
    fc_dmax = float(np.max(fc_distances))
    fc_contract = bool(fc_dmax <= float(args.exact_functional_tolerance))
    fc_diagnostics = {
        "q0": [1.5, 1.5], "q_star": q_star.tolist(), "solver": solver,
        "fit_role_only": True, "fit_sample_count": int(len(base_fit)),
        "robust_medians": scale_payload["medians"].tolist(), "robust_scales": scale_payload["scales"].tolist(),
        "stratum_counts": scale_payload["counts"].tolist(), "gram": gram.tolist(),
        "exact_distances": _distance_payload(fc_distances), "exact_dmax": fc_dmax,
        "exact_tolerance": float(args.exact_functional_tolerance), "functional_contract_passed": fc_contract,
        "direction_support": support, "full_direction_norm": _direction_norm(direction),
    }
    _write_json(seed_dir / "FC_PTV" / "functional_diagnostics.json", fc_diagnostics)

    variants.append(_save_and_evaluate("FullAlpha1p5", _apply_direction(base_state, direction, 1.5), {"source": "frozen_full_alpha", "scale": 1.5}, f2_dir, roles, weak6, seed_dir, device, args))
    fc_row = _save_and_evaluate("FC_PTV", fc_state, fc_diagnostics, f2_dir, roles, weak6, seed_dir, device, args)
    if fc_row.get("status") == "valid" and not fc_contract:
        fc_row["status"] = "invalid_functional_contract"
    variants.append(fc_row)

    shared_direction = _state_direction(base_state, shared_state)
    classifier_direction = _state_direction(base_state, classifier_state)
    fc_update = _direction_add(_direction_scale(shared_direction, q_star[0]), _direction_scale(classifier_direction, q_star[1]))
    parameter_scale = _direction_norm(fc_update) / (_direction_norm(direction) + 1e-24)
    variants.append(_save_and_evaluate("ParameterNormOnly", _apply_direction(base_state, direction, parameter_scale), {"source": "parameter_norm_only", "scale": parameter_scale, "matched_update_norm": _direction_norm(fc_update)}, f2_dir, roles, weak6, seed_dir, device, args))
    variants.append(_save_and_evaluate("ClassifierOnlyAlpha2p0", _compose_layer_state(base_state, direction, 0.0, 2.0), {"source": "frozen_classifier_only", "shared_scale": 0.0, "classifier_scale": 2.0}, f2_dir, roles, weak6, seed_dir, device, args))
    variants.append(_save_and_evaluate("ModelAverage0p5", _apply_direction(base_state, direction, 0.5), {"source": "paired_midpoint", "scale": 0.5}, f2_dir, roles, weak6, seed_dir, device, args))

    other_directions = [_state_direction(base_state, states[name]) for name in OTHER_MASK_ENDPOINTS]
    orthogonal_direction = _orthogonal_task_subspace(direction, other_directions)
    orth_unit_state = _apply_direction(base_state, orthogonal_direction, 1.0)
    orth_unit_fit = _fit_logits(orth_unit_state, f2_dir, roles, device, args)
    orth_unit_distances = exact_functional_distances(base_fit, orth_unit_fit, saw_fit, scale_payload["scales"])
    orth_unit_dmax = float(np.max(orth_unit_distances))
    orth_scale = fc_dmax / (orth_unit_dmax + 1e-24)
    orth_state = _apply_direction(base_state, orthogonal_direction, orth_scale)
    orth_fit = _fit_logits(orth_state, f2_dir, roles, device, args)
    orth_distances = exact_functional_distances(base_fit, orth_fit, saw_fit, scale_payload["scales"])
    orth_dmax = float(np.max(orth_distances))
    orth_relative_error = abs(orth_dmax - fc_dmax) / max(fc_dmax, 1e-12)
    orth_match = bool(np.isfinite(orth_scale) and np.all(np.isfinite(orth_distances)) and orth_relative_error <= 0.10)
    orth_diagnostics = {
        "source": "other_label_orthogonal_task_subspace", "scale": float(orth_scale),
        "unit_dmax": orth_unit_dmax, "target_dmax": fc_dmax, "matched_dmax": orth_dmax,
        "relative_match_error": float(orth_relative_error), "functional_match_passed": orth_match,
        "exact_distances": _distance_payload(orth_distances),
    }
    orth_row = _save_and_evaluate("OrthogonalTaskSubspaceFunctionalMatched", orth_state, orth_diagnostics, f2_dir, roles, weak6, seed_dir, device, args)
    if orth_row.get("status") == "valid" and not orth_match:
        orth_row["status"] = "invalid_functional_match"
    variants.append(orth_row)

    _write_json(seed_dir / "longtail_manifest.json", manifest.to_dict())
    np.savez_compressed(str(seed_dir / "role_indices.npz"), manifest=manifest.selected_indices, **ids)
    payload = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke), "data_sha256": data_sha256,
        "split_sha256": split.sha256, "manifest_sha256": manifest.sha256,
        "role_indices_sha256": _indices_sha256(*[ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")]),
        "f2_checkpoint": str(checkpoint_path), "f2_checkpoint_sha256": _sha256(checkpoint_path),
        "training": training, "variants": variants, "role_counts": {key: len(value) for key, value in ids.items()},
        "fc_ptv": fc_diagnostics, "orthogonal_control": orth_diagnostics,
        "identity_audit_passed": True, "locked_test_used": False, "test_v2_used": False,
        "test_v3_used": False, "test_v12_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    identity = _read_json(args.identity_audit)
    if not identity.get("identity_audit_passed") or identity.get("evidence_role") != "development_v12" or identity.get("test_files_opened"):
        raise ValueError("development-v12 identity audit failed")
    if any(identity.get(name, True) for name in ("locked_test_used", "test_v2_used", "test_v3_used", "test_v12_used")):
        raise ValueError("forbidden role use in development-v12 identity audit")
    data_path = _data_path(config)
    data_sha256 = _sha256(data_path)
    if identity.get("sha256", "").lower() != data_sha256.lower():
        raise ValueError("development-v12 data hash mismatch")
    labels6 = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameters), float(args.weak_threshold_db))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        args.epochs = int(args.smoke_epochs)
    seed_rows = [_run_seed(args, config, labels6, weak6, int(seed), device, data_sha256) for seed in args.seeds]
    aggregate, comparisons = _aggregate(seed_rows)
    if args.smoke:
        candidate_gate = {"passed": False, "checks": {}, "reason": "smoke_only"}
        improvement_gate = {"passed": False, "checks": {}, "reason": "smoke_only"}
        mechanism_gate = {"passed": False, "checks": {}, "reason": "smoke_only"}
        simultaneous_gate = {"passed": False, "checks": {}, "reason": "smoke_only"}
        decision = "smoke_only_no_method_claim"
    else:
        candidate_gate = _candidate_gate(comparisons["FC_PTV"], aggregate["FC_PTV"])
        improvement_gate = _improvement_gate(seed_rows, comparisons, aggregate)
        mechanism_gate = _mechanism_gate(seed_rows)
        simultaneous_gate = _simultaneous_gate(comparisons["FC_PTV"])
        if aggregate["FC_PTV"]["valid_seed_count"] < 5:
            decision = "invalid_and_stop_fc_ptv"
        elif (candidate_gate["passed"] and improvement_gate["passed"]
              and mechanism_gate["passed"] and simultaneous_gate["passed"]):
            decision = "promote_to_full_canonical_baselines"
        else:
            decision = "stop_fc_ptv"
    payload = {
        "status": "completed", "experiment": "route_a_development_v12_fc_ptv",
        "evidence_level": "smoke" if args.smoke else "development", "smoke": bool(args.smoke),
        "device": str(device), "protocol": {
            "seeds": [int(seed) for seed in args.seeds], "variants": list(VARIANTS), "epochs": int(args.epochs),
            "functional_radius": "C6 fit-only per-label/per-SawFM-stratum 1.4826*MAD",
            "q0": [1.5, 1.5], "exact_functional_tolerance": float(args.exact_functional_tolerance),
            "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v12_used": False,
        },
        "data": {"path": str(data_path), "sha256": data_sha256, "sample_count": len(labels6), "identity_audit": identity},
        "per_seed": seed_rows, "aggregate": aggregate, "comparisons_vs_c6": comparisons,
        "candidate_effect_gate": candidate_gate, "improvement_vs_fullalpha_gate": improvement_gate,
        "mechanism_specificity_gate": mechanism_gate, "simultaneous_target_gate": simultaneous_gate,
        "next_decision": decision, "locked_test_used": False, "test_v2_used": False,
        "test_v3_used": False, "test_v12_used": False,
    }
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "route_a_development_v12_fc_ptv_summary.json"
    _write_json(summary_path, payload)
    _write_json(root / "run_receipt.json", {
        "status": "completed", "evidence_level": payload["evidence_level"], "data_sha256": data_sha256,
        "identity_audit_passed": True, "candidate_effect_gate_passed": candidate_gate["passed"],
        "improvement_vs_fullalpha_gate_passed": improvement_gate["passed"],
        "mechanism_specificity_gate_passed": mechanism_gate["passed"],
        "simultaneous_target_gate_passed": simultaneous_gate["passed"], "next_decision": decision,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
        "test_v12_used": False, "test_access_allowed": False,
    })
    print(json.dumps({
        "status": "completed", "evidence_level": payload["evidence_level"],
        "candidate_effect_gate": candidate_gate, "improvement_vs_fullalpha_gate": improvement_gate,
        "mechanism_specificity_gate": mechanism_gate, "simultaneous_target_gate": simultaneous_gate,
        "next_decision": decision,
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v12.yaml")
    parser.add_argument("--identity-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v12_generation/identity_audit.json")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v12_fc_ptv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--representation-lr-scale", type=float, default=0.1)
    parser.add_argument("--saw-loss-weight", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--cagrad-c", type=float, default=0.4)
    parser.add_argument("--cagrad-grid", type=int, default=101)
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-weight-lr", type=float, default=0.025)
    parser.add_argument("--functional-min-scale", type=float, default=1e-6)
    parser.add_argument("--exact-functional-tolerance", type=float, default=1.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=128)
    parser.add_argument("--smoke-eval-samples", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
