"""Development-v5 layer/alpha decomposition of the paired SawFM task direction."""

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
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    COMMON_INDICES,
    COMMON_NAMES,
    TARGET_NAMES,
    _indices_sha256,
    _read_json,
    _sha256,
    _target_weak_rank,
    _write_json,
)
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _component_power,
    _macro_aupr,
    _predict,
)
from experiments.run_route_a_development_v4_sawfm_specificity import (  # noqa: E402
    LABEL_NAMES,
    MASKED_ENDPOINTS,
    _apply_direction,
    _build_roles,
    _direction_cosine,
    _direction_norm,
    _evaluate_model_v4,
    _load_checkpoint_state,
    _load_model_from_state,
    _metric,
    _save_and_evaluate,
    _state_direction,
    _train_endpoint,
    _variant_map,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


FIXED_SPECS = {
    "FullAlpha0p5": (0.5, 0.5),
    "FullAlpha1p5": (1.5, 1.5),
    "FullAlpha2p0": (2.0, 2.0),
    "SharedOnlyAlpha0p5": (0.5, 0.0),
    "SharedOnlyAlpha1p0": (1.0, 0.0),
    "SharedOnlyAlpha1p5": (1.5, 0.0),
    "SharedOnlyAlpha2p0": (2.0, 0.0),
    "ClassifierOnlyAlpha0p5": (0.0, 0.5),
    "ClassifierOnlyAlpha1p0": (0.0, 1.0),
    "ClassifierOnlyAlpha1p5": (0.0, 1.5),
    "ClassifierOnlyAlpha2p0": (0.0, 2.0),
    "SplitShared1p5Classifier0p5": (1.5, 0.5),
    "SplitShared1p5Classifier1p0": (1.5, 1.0),
}
SELECTABLE_VARIANTS = tuple(FIXED_SPECS)
EVALUATED_VARIANTS = (
    "C6_Joint",
    "M5_GradientIsolated",
    *FIXED_SPECS.keys(),
    "AdaptiveLayerAlpha",
    "OrthogonalTaskSubspace1p5",
)


def _validate_v5_config(config):
    if bool(config.get("allow_locked_test", False)):
        raise ValueError("development-v5 requires allow_locked_test=false")
    if str(config.get("evidence_role")) != "development_v5":
        raise ValueError("config evidence_role must be development_v5")
    if bool(config.get("test_v5_generated", True)):
        raise ValueError("test-v5 must not be generated")
    if bool(config.get("test_v5_access_authorized", True)) or bool(config.get("test_v5_used", True)):
        raise ValueError("test-v5 access/use must remain false")
    if not bool(config.get("test_v3_sealed", False)) or bool(config.get("test_v3_used", True)):
        raise ValueError("test-v3 must remain sealed and unused")
    if not bool(config.get("test_v2_consumed", False)):
        raise ValueError("test-v2 must remain declared consumed")
    if bool(config.get("development_v4_reuse_for_selection", True)):
        raise ValueError("development-v4 reuse for selection must be false")
    data_path = (Path(str(config["data_root"])) / str(config["train_file"])).resolve()
    forbidden = {
        Path(str(config["test_v5_path"])).resolve(),
        Path(str(config["sealed_test_v3_path"])).resolve(),
        Path(str(config["forbidden_test_v2_path"])).resolve(),
    }
    if data_path in forbidden or "test" in data_path.name.lower():
        raise ValueError("development-v5 data path resolves to a test role")


def _direction_component(direction, component):
    result = {}
    for key, value in direction.items():
        keep = (
            component == "full"
            or (component == "shared" and key.startswith("temporal_backbone.0."))
            or (component == "classifier" and key.startswith("classifier."))
        )
        result[key] = value.clone() if keep else torch.zeros_like(value)
    return result


def _compose_layer_state(base_state, direction, shared_alpha, classifier_alpha):
    result = {}
    for key, base in base_state.items():
        if not torch.is_floating_point(base):
            result[key] = base.clone()
            continue
        if key.startswith("temporal_backbone.0."):
            scale = float(shared_alpha)
        elif key.startswith("classifier."):
            scale = float(classifier_alpha)
        else:
            scale = 0.0
        result[key] = base + scale * direction[key].to(dtype=base.dtype)
    return result


def _direction_add(first, second):
    return {key: first[key] + second[key] for key in first}


def _direction_scale(direction, scale):
    return {key: value * float(scale) for key, value in direction.items()}


def _direction_dot(first, second):
    return float(sum((first[key].double() * second[key].double()).sum() for key in first))


def _orthogonal_task_subspace(saw_direction, other_directions):
    unit = []
    for direction in other_directions:
        norm = _direction_norm(direction)
        if norm <= 1e-12:
            raise ValueError("other-label direction has zero norm")
        unit.append(_direction_scale(direction, 1.0 / norm))
    combined = {key: sum(direction[key] for direction in unit) / len(unit) for key in saw_direction}
    saw_norm_sq = _direction_dot(saw_direction, saw_direction)
    residual = _direction_add(
        combined,
        _direction_scale(saw_direction, -_direction_dot(combined, saw_direction) / (saw_norm_sq + 1e-24)),
    )
    residual_norm = _direction_norm(residual)
    if residual_norm < 1e-8:
        raise ValueError("orthogonal task-subspace residual is below preregistered tolerance")
    return _direction_scale(residual, _direction_norm(saw_direction) / residual_norm)


def _validation_scores(state, f2_dir, roles, weak6, device, args):
    model = _load_model_from_state(f2_dir, device, state)
    logits = _predict(model, roles["validation"]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES]
    if not np.all(np.isfinite(logits)):
        del model
        return {"valid": False, "macro_aupr": -float("inf"), "target_weak_auroc": -float("inf")}
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    targets = roles["validation"]["targets6"][:, COMMON_INDICES]
    weak = weak6[roles["validation"]["sample_ids"]][:, COMMON_INDICES]
    result = {
        "valid": True,
        "macro_aupr": _macro_aupr(targets, probability),
        "target_weak_auroc": _target_weak_rank(targets, probability, weak),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _choose_adaptive(score_rows, tolerance=0.005):
    valid = [row for row in score_rows if row["valid"] and np.isfinite(row["macro_aupr"])]
    if not valid:
        raise RuntimeError("adaptive layer-alpha selection has no valid state")
    best_macro = max(row["macro_aupr"] for row in valid)
    eligible = [row for row in valid if row["macro_aupr"] >= best_macro - float(tolerance)]
    eligible.sort(key=lambda row: (
        -row["target_weak_auroc"],
        row["total_alpha"],
        row["classifier_alpha"],
        row["variant"],
    ))
    return eligible[0], {"best_macro_aupr": best_macro, "eligible": [row["variant"] for row in eligible]}


def _run_seed(args, config, labels6, weak6, seed, device, data_sha256):
    seed_dir = Path(args.output_root) / "seed_{}".format(seed)
    summary_path = seed_dir / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return _read_json(summary_path)
    split, manifest, ids, roles = _build_roles(args, config, labels6, seed)
    f2_dir = Path(args.f2_root) / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    template, checkpoint_path = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial_state = copy.deepcopy({key: value.detach().cpu() for key, value in template.state_dict().items()})
    del template
    endpoint_states = {}
    training = {}
    variants = []
    for variant, masked_index in [("C6_Joint", None), *MASKED_ENDPOINTS.items()]:
        model = _load_model_from_state(f2_dir, device, initial_state)
        model, training[variant] = _train_endpoint(
            model, variant, masked_index, roles, weak6, device, seed, args, seed_dir / variant
        )
        endpoint_states[variant] = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if variant in ("C6_Joint", "M5_GradientIsolated"):
            row = _evaluate_model_v4(variant, model, roles, weak6, seed_dir, device, args)
            row["status"] = "valid"
            variants.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    c6_state = endpoint_states["C6_Joint"]
    saw_direction = _state_direction(c6_state, endpoint_states["M5_GradientIsolated"])
    shared_direction = _direction_component(saw_direction, "shared")
    classifier_direction = _direction_component(saw_direction, "classifier")
    fixed_states = {
        name: _compose_layer_state(c6_state, saw_direction, shared_alpha, classifier_alpha)
        for name, (shared_alpha, classifier_alpha) in FIXED_SPECS.items()
    }
    score_rows = []
    for name in SELECTABLE_VARIANTS:
        shared_alpha, classifier_alpha = FIXED_SPECS[name]
        scores = _validation_scores(fixed_states[name], f2_dir, roles, weak6, device, args)
        score_rows.append({
            "variant": name,
            "shared_alpha": shared_alpha,
            "classifier_alpha": classifier_alpha,
            "total_alpha": shared_alpha + classifier_alpha,
            **scores,
        })
    selected, selection_context = _choose_adaptive(score_rows)

    direction_diagnostics = {
        "full_norm": _direction_norm(saw_direction),
        "shared_norm": _direction_norm(shared_direction),
        "classifier_norm": _direction_norm(classifier_direction),
        "shared_classifier_cosine": _direction_cosine(shared_direction, classifier_direction),
        "nonzero_support": {
            "shared": int(sum(_direction_norm({key: value}) > 0 for key, value in shared_direction.items())),
            "classifier": int(sum(_direction_norm({key: value}) > 0 for key, value in classifier_direction.items())),
        },
    }
    for name in SELECTABLE_VARIANTS:
        shared_alpha, classifier_alpha = FIXED_SPECS[name]
        diagnostics = {
            "source": "paired_sawfm_layer_alpha_grid",
            "shared_alpha": shared_alpha,
            "classifier_alpha": classifier_alpha,
            **direction_diagnostics,
        }
        variants.append(_save_and_evaluate(
            name, fixed_states[name], diagnostics, f2_dir, roles, weak6, seed_dir, device, args
        ))

    adaptive_diagnostics = {
        "source": "internal_validation_layer_alpha_selection",
        "selected_fixed_variant": selected["variant"],
        "selection_scores": score_rows,
        "selection_context": selection_context,
        "shared_alpha": selected["shared_alpha"],
        "classifier_alpha": selected["classifier_alpha"],
        **direction_diagnostics,
    }
    variants.append(_save_and_evaluate(
        "AdaptiveLayerAlpha",
        fixed_states[selected["variant"]],
        adaptive_diagnostics,
        f2_dir,
        roles,
        weak6,
        seed_dir,
        device,
        args,
    ))

    other_directions = [
        _state_direction(c6_state, endpoint_states[name]) for name in MASKED_ENDPOINTS if name != "M5_GradientIsolated"
    ]
    orthogonal = _orthogonal_task_subspace(saw_direction, other_directions)
    orth_diagnostics = {
        "source": "orthogonal_other_label_task_subspace",
        "scale": 1.5,
        "direction_norm": _direction_norm(orthogonal),
        "cosine_vs_sawfm": _direction_cosine(orthogonal, saw_direction),
    }
    variants.append(_save_and_evaluate(
        "OrthogonalTaskSubspace1p5",
        _apply_direction(c6_state, orthogonal, 1.5),
        orth_diagnostics,
        f2_dir,
        roles,
        weak6,
        seed_dir,
        device,
        args,
    ))

    _write_json(seed_dir / "longtail_manifest.json", manifest.to_dict())
    np.savez_compressed(str(seed_dir / "role_indices.npz"), manifest=manifest.selected_indices, **ids)
    payload = {
        "seed": int(seed),
        "status": "completed",
        "smoke": bool(args.smoke),
        "data_sha256": data_sha256,
        "split_sha256": split.sha256,
        "manifest_sha256": manifest.sha256,
        "role_indices_sha256": _indices_sha256(*[ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")]),
        "f2_checkpoint": str(checkpoint_path),
        "f2_checkpoint_sha256": _sha256(checkpoint_path),
        "training": training,
        "direction_diagnostics": direction_diagnostics,
        "adaptive_selection": adaptive_diagnostics,
        "variants": variants,
        "role_counts": {role: int(len(value)) for role, value in ids.items()},
        "identity_audit_passed": True,
        "invalid_control_count": int(sum(row.get("metrics") is None for row in variants)),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
        "test_v5_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def _aggregate(seed_rows):
    paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "hamming_loss": ("metrics", "hamming_loss"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    result = {}
    for variant_index, variant in enumerate(EVALUATED_VARIANTS):
        all_rows = [_variant_map(seed_row)[variant] for seed_row in seed_rows]
        rows = [row for row in all_rows if row.get("metrics") is not None]
        result[variant] = {
            name: _mean_ci([_metric(row, path) for row in rows], 20262000 + 100 * variant_index + offset)
            for offset, (name, path) in enumerate(paths.items())
        } if rows else {}
        result[variant]["per_label_aupr"] = {
            name: _mean_ci([row["metrics"]["per_label_diagnostics"][name]["aupr"] for row in rows], 20262100 + 100 * variant_index + index)
            for index, name in enumerate(COMMON_NAMES)
        } if rows else {}
        result[variant]["per_label_weak_recall"] = {
            name: _mean_ci([row["metrics"]["per_label_diagnostics"][name]["weak_recall"] for row in rows], 20262200 + 100 * variant_index + index)
            for index, name in enumerate(TARGET_NAMES)
        } if rows else {}
        result[variant]["valid_seed_count"] = len(rows)
        result[variant]["invalid_seed_count"] = len(all_rows) - len(rows)
    return result


def _comparison(seed_rows, candidate, baseline="C6_Joint"):
    valid = []
    invalid = []
    for seed_row in seed_rows:
        variants = _variant_map(seed_row)
        if variants[candidate].get("metrics") is None or variants[baseline].get("metrics") is None:
            invalid.append(int(seed_row["seed"]))
        else:
            valid.append(seed_row)
    metric_paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    per_seed = []
    for seed_row in valid:
        variants = _variant_map(seed_row)
        c, b = variants[candidate], variants[baseline]
        delta = {name: _metric(c, path) - _metric(b, path) for name, path in metric_paths.items()}
        delta["per_label_aupr"] = {
            name: c["metrics"]["per_label_diagnostics"][name]["aupr"] - b["metrics"]["per_label_diagnostics"][name]["aupr"]
            for name in COMMON_NAMES
        }
        delta["per_label_weak_recall"] = {
            name: c["metrics"]["per_label_diagnostics"][name]["weak_recall"] - b["metrics"]["per_label_diagnostics"][name]["weak_recall"]
            for name in TARGET_NAMES
        }
        per_seed.append({"seed": int(seed_row["seed"]), "delta": delta})
    aggregate = {
        name: _mean_ci([row["delta"][name] for row in per_seed], 20262300 + offset)
        for offset, name in enumerate(metric_paths)
    } if per_seed else {}
    aggregate["per_label_aupr"] = {
        name: _mean_ci([row["delta"]["per_label_aupr"][name] for row in per_seed], 20262400 + index)
        for index, name in enumerate(COMMON_NAMES)
    } if per_seed else {}
    aggregate["per_label_weak_recall"] = {
        name: _mean_ci([row["delta"]["per_label_weak_recall"][name] for row in per_seed], 20262500 + index)
        for index, name in enumerate(TARGET_NAMES)
    } if per_seed else {}
    aggregate["positive_target_weak_recall_seeds"] = int(sum(row["delta"]["target_mean_weak_recall"] > 0 for row in per_seed))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "valid_seed_count": len(valid),
        "invalid_seed_count": len(invalid),
        "invalid_seeds": invalid,
    }


def _candidate_eligible(name, comparison, aggregate):
    if comparison["valid_seed_count"] != 5:
        return False, {"valid_5_of_5": False}
    delta = comparison["aggregate"]
    checks = {
        "valid_5_of_5": True,
        "macro_aupr_noninferior": delta["macro_aupr"]["mean"] >= -0.005,
        "macro_f1_gain": delta["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain": delta["target_mean_weak_recall"]["mean"] >= 0.020,
        "each_target_weak_recall_nonnegative": all(delta["per_label_weak_recall"][target]["mean"] >= 0 for target in TARGET_NAMES),
        "target_fpr_cap": aggregate[name]["target_mean_negative_fpr"]["mean"] <= 0.20,
        "positive_weak_recall_4_of_5": delta["positive_target_weak_recall_seeds"] >= 4,
        "saw_present_positive": delta["sawfm_present_macro_f1"]["mean"] > 0,
        "saw_absent_noninferior": delta["sawfm_absent_macro_f1"]["mean"] >= -0.005,
        "target_aupr_noninferior": all(delta["per_label_aupr"][target]["mean"] >= -0.010 for target in TARGET_NAMES),
    }
    return bool(all(checks.values())), checks


def _selection_metadata(name):
    if name == "AdaptiveLayerAlpha":
        return float("inf"), float("inf")
    shared, classifier = FIXED_SPECS[name]
    return shared + classifier, classifier


def _select_candidate(comparisons, aggregate):
    candidates = [*SELECTABLE_VARIANTS, "AdaptiveLayerAlpha"]
    audit = {}
    eligible = []
    for name in candidates:
        passed, checks = _candidate_eligible(name, comparisons[name], aggregate)
        audit[name] = {"passed": passed, "checks": checks}
        if passed:
            delta = comparisons[name]["aggregate"]
            worst = min(delta["per_label_weak_recall"][target]["mean"] for target in TARGET_NAMES)
            total_alpha, classifier_alpha = _selection_metadata(name)
            eligible.append((name, worst, delta["target_mean_weak_recall"]["mean"], delta["macro_f1"]["mean"], total_alpha, classifier_alpha))
    eligible.sort(key=lambda row: (-row[1], -row[2], -row[3], row[4], row[5], row[0]))
    return (eligible[0][0] if eligible else None), audit, eligible


def _promotion_gate(selected, comparisons, aggregate):
    if selected is None:
        return {"passed": False, "checks": {}, "reason": "no_eligible_candidate"}
    delta = comparisons[selected]["aggregate"]
    reference = comparisons["FullAlpha1p5"]["aggregate"]
    orth = comparisons["OrthogonalTaskSubspace1p5"]
    selected_worst = min(delta["per_label_weak_recall"][target]["mean"] for target in TARGET_NAMES)
    reference_worst = min(reference["per_label_weak_recall"][target]["mean"] for target in TARGET_NAMES)
    checks = {
        "target_weak_recall_gain_0p025": delta["target_mean_weak_recall"]["mean"] >= 0.025,
        "worst_target_weak_recall_nonnegative": selected_worst >= 0.0,
        "worst_target_improves_full1p5_by_0p005": selected_worst - reference_worst >= 0.005,
        "macro_f1_gain_0p010": delta["macro_f1"]["mean"] >= 0.010,
        "macro_aupr_noninferior": delta["macro_aupr"]["mean"] >= -0.005,
        "target_fpr_cap": aggregate[selected]["target_mean_negative_fpr"]["mean"] <= 0.20,
        "orthogonal_control_finite": orth["valid_seed_count"] == 5,
        "beats_orthogonal_weak_recall": orth["valid_seed_count"] == 5 and delta["target_mean_weak_recall"]["mean"] > orth["aggregate"]["target_mean_weak_recall"]["mean"],
        "beats_orthogonal_macro_f1": orth["valid_seed_count"] == 5 and delta["macro_f1"]["mean"] > orth["aggregate"]["macro_f1"]["mean"],
    }
    return {
        "selected_candidate": selected,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "selected_worst_target_weak_recall_delta": selected_worst,
        "full1p5_worst_target_weak_recall_delta": reference_worst,
    }


def run(args):
    config = load_config(args.config)
    _validate_v5_config(config)
    identity = _read_json(args.identity_audit)
    if not identity.get("identity_audit_passed") or identity.get("test_files_opened"):
        raise ValueError("development-v5 identity audit failed")
    if any(identity.get(name, True) for name in ("locked_test_used", "test_v2_used", "test_v3_used", "test_v4_used", "test_v5_used")):
        raise ValueError("identity audit reports forbidden role use")
    data_path = _data_path(config)
    data_sha256 = _sha256(data_path)
    if identity.get("sha256", "").lower() != data_sha256.lower():
        raise ValueError("development-v5 data hash differs from identity audit")
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameter_values), float(args.weak_threshold_db))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.epochs = int(args.smoke_epochs) if args.smoke else int(args.epochs)
    seed_rows = [_run_seed(args, config, labels6, weak6, int(seed), device, data_sha256) for seed in args.seeds]
    aggregate = _aggregate(seed_rows)
    comparisons = {variant: _comparison(seed_rows, variant) for variant in EVALUATED_VARIANTS if variant != "C6_Joint"}
    if args.smoke:
        selected = None
        eligibility = {}
        eligible_ranking = []
        promotion = {"passed": False, "reason": "smoke_only", "checks": {}}
        decision = "smoke_only_no_method_claim"
    else:
        selected, eligibility, eligible_ranking = _select_candidate(comparisons, aggregate)
        promotion = _promotion_gate(selected, comparisons, aggregate)
        if promotion["passed"]:
            decision = "promote_to_general_multitask_baseline_stage"
        elif selected is not None:
            decision = "revise_and_stop_current_alpha_grid"
        else:
            decision = "stop_parameter_extrapolation_improvement_route"
    payload = {
        "status": "completed",
        "experiment": "route_a_development_v5_layer_alpha",
        "evidence_level": "smoke" if args.smoke else "development",
        "smoke": bool(args.smoke),
        "device": str(device),
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "fixed_specs": {name: {"shared_alpha": value[0], "classifier_alpha": value[1]} for name, value in FIXED_SPECS.items()},
            "evaluated_variants": list(EVALUATED_VARIANTS),
            "epochs": int(args.epochs),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
            "test_v4_used": False,
            "test_v5_used": False,
        },
        "data": {"path": str(data_path), "sha256": data_sha256, "sample_count": int(len(labels6)), "identity_audit": identity},
        "per_seed": seed_rows,
        "aggregate": aggregate,
        "comparisons_vs_c6": comparisons,
        "candidate_eligibility": eligibility,
        "eligible_ranking": eligible_ranking,
        "selected_candidate": selected,
        "promotion_gate": promotion,
        "next_decision": decision,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
        "test_v5_used": False,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "route_a_development_v5_layer_alpha_summary.json", payload)
    _write_json(output_root / "run_receipt.json", {
        "status": "completed",
        "evidence_level": payload["evidence_level"],
        "data_sha256": data_sha256,
        "identity_audit_passed": True,
        "selected_candidate": selected,
        "promotion_gate_passed": bool(promotion["passed"]),
        "next_decision": decision,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
        "test_v5_used": False,
        "test_access_allowed": False,
    })
    print(json.dumps({
        "status": payload["status"],
        "evidence_level": payload["evidence_level"],
        "selected_candidate": selected,
        "promotion_gate": promotion,
        "next_decision": decision,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
        "test_v5_used": False,
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v5.yaml")
    parser.add_argument("--identity-audit", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v5_generation" / "identity_audit.json")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v5_layer_alpha")
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=128)
    parser.add_argument("--smoke-eval-samples", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

