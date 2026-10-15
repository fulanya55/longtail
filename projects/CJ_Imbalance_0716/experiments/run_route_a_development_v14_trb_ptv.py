"""Preregistered, train-only development-v14 runner for TRB-PTV.

No test role is accepted.  The only coefficient selection is the frozen
fit-only constrained solve in :mod:`route_a_trb_ptv_core`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments import run_route_a_development_v12_fc_ptv as fc  # noqa: E402
from experiments.route_a_trb_ptv_core import activation_gate, solve_target_risk_projection  # noqa: E402
from experiments.route_a_v14_identity import audit_train  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    COMMON_INDICES, TARGET_NAMES, _indices_sha256, _read_json, _sha256, _write_json,
)
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power, _predict  # noqa: E402
from experiments.run_route_a_development_v4_sawfm_specificity import (  # noqa: E402
    MASKED_ENDPOINTS, OTHER_MASK_ENDPOINTS, _apply_direction, _build_roles, _direction_norm,
    _evaluate_model_v4, _load_model_from_state,
    _save_and_evaluate, _state_direction, _train_endpoint,
)
from experiments.run_route_a_development_v5_layer_alpha import (  # noqa: E402
    _compose_layer_state, _direction_add, _direction_scale, _orthogonal_task_subspace,
)
from experiments.run_route_a_development_v6_multitask_baselines import _train_baseline  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = (
    "C6_Joint", "M5_GradientIsolated", "FullAlpha1p5", "TRB_PTV",
    "ParameterNormOnly", "ClassifierOnlyAlpha2p0", "ModelAverage0p5",
    "CAGrad0p4TwoTask", "OrthogonalTaskSubspaceFunctionalMatched",
)


def _validate_config(config):
    if str(config.get("evidence_role")) != "development_v14":
        raise ValueError("TRB-PTV requires development_v14")
    if bool(config.get("allow_locked_test", False)) or bool(config.get("test_v14_access_authorized", True)):
        raise ValueError("TRB-PTV forbids test access")
    if bool(config.get("test_v14_generated", True)) or bool(config.get("test_v14_used", True)):
        raise ValueError("development-v14 must generate and use train only")
    if not bool(config.get("test_v3_sealed", False)) or bool(config.get("test_v3_used", True)):
        raise ValueError("test-v3 must remain sealed")
    if not bool(config.get("test_v2_consumed", False)):
        raise ValueError("test-v2 must remain consumed")
    data_path = _data_path(config)
    if "test" in data_path.name.lower():
        raise ValueError("development-v14 config resolves to a test path")


def _direction_support(direction, tolerance=1e-12):
    support = []
    for key, value in direction.items():
        norm = float(torch.linalg.vector_norm(value.double()).cpu())
        if norm <= tolerance:
            continue
        if not (key.startswith("temporal_backbone.0.") or key.startswith("classifier.")):
            raise ValueError(f"paired direction has forbidden support: {key}")
        support.append({"key": key, "norm": norm})
    return support


def _fit_logits_for_state(state, f2_dir, roles, device, args):
    model = _load_model_from_state(f2_dir, device, state)
    logits = np.asarray(_predict(model, roles["fit"]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES], dtype=np.float64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("fit logits are non-finite")
    return logits


def _target_risk_matched_orthogonal_state(base_state, orth_direction, desired_update_norm, f2_dir, roles, device, args, targets, weak):
    """One-dimensional, fit-only risk-safe counterpart to TRB's orthogonal control."""
    def fit_logits(scale):
        return _fit_logits_for_state(_apply_direction(base_state, orth_direction, float(scale[0])), f2_dir, roles, device, args)

    reference = max(float(desired_update_norm) / (_direction_norm(orth_direction) + 1e-24), 0.0)
    # Reuse the frozen solver by mapping scalar a to (0,a); its q_shared<=q_classifier
    # constraint remains valid and it minimizes distance to the fixed full reference only
    # through the surrounding scale definition, never through a held-out metric.
    q_star, diagnostics = solve_target_risk_projection(
        lambda q: fit_logits(np.asarray([q[1], q[1]])), targets, weak,
        q0=(min(reference, 1.5), min(reference, 1.5)), upper=1.5,
        maxiter=args.trb_maxiter, ftol=args.trb_ftol, constraint_tolerance=args.trb_constraint_tolerance,
    )
    scale = float(q_star[1])
    state = _apply_direction(base_state, orth_direction, scale)
    diagnostics["source"] = "other_label_orthogonal_task_subspace_target_risk_matched"
    diagnostics["scale"] = scale
    diagnostics["desired_update_norm"] = float(desired_update_norm)
    return state, diagnostics


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
            row["status"] = "valid"; variants.append(row)
        del model
    cagrad_model = _load_model_from_state(f2_dir, device, initial)
    cagrad_model, training["CAGrad0p4TwoTask"] = _train_baseline(cagrad_model, "CAGrad0p4TwoTask", roles, weak6, device, seed, args, seed_dir / "CAGrad0p4TwoTask")
    cagrad_row = _evaluate_model_v4("CAGrad0p4TwoTask", cagrad_model, roles, weak6, seed_dir, device, args)
    cagrad_row["status"] = "valid"; variants.append(cagrad_row); del cagrad_model

    base_state = states["C6_Joint"]
    direction = _state_direction(base_state, states["M5_GradientIsolated"])
    support = _direction_support(direction)
    fit_targets = roles["fit"]["targets6"][:, COMMON_INDICES]
    fit_weak = weak6[roles["fit"]["sample_ids"]][:, COMMON_INDICES]

    def trb_fit_logits(q):
        return _fit_logits_for_state(_compose_layer_state(base_state, direction, float(q[0]), float(q[1])), f2_dir, roles, device, args)

    q_star, trb_solver = solve_target_risk_projection(
        trb_fit_logits, fit_targets, fit_weak, maxiter=args.trb_maxiter, ftol=args.trb_ftol,
        constraint_tolerance=args.trb_constraint_tolerance,
    )
    trb_state = _compose_layer_state(base_state, direction, q_star[0], q_star[1])
    trb_diagnostics = {
        "q0": [1.5, 1.5], "q_star": q_star.tolist(), "solver": trb_solver,
        "fit_role_only": True, "fit_sample_count": int(len(fit_targets)),
        "target_labels": list(TARGET_NAMES), "fixed_label_weights": [1.0, 1.0, 1.0],
        "direction_support": support, "full_direction_norm": _direction_norm(direction),
    }
    _write_json(seed_dir / "TRB_PTV" / "target_risk_diagnostics.json", trb_diagnostics)

    variants.append(_save_and_evaluate("FullAlpha1p5", _apply_direction(base_state, direction, 1.5), {"source": "frozen_full_alpha", "scale": 1.5}, f2_dir, roles, weak6, seed_dir, device, args))
    trb_row = _save_and_evaluate("TRB_PTV", trb_state, trb_diagnostics, f2_dir, roles, weak6, seed_dir, device, args)
    if trb_row.get("status") == "valid" and (q_star[0] <= 0 or not trb_solver["finite"]):
        trb_row["status"] = "invalid_trb_contract"
    variants.append(trb_row)

    shared_state = _compose_layer_state(base_state, direction, 1.0, 0.0)
    classifier_state = _compose_layer_state(base_state, direction, 0.0, 1.0)
    trb_update = _direction_add(_direction_scale(_state_direction(base_state, shared_state), q_star[0]), _direction_scale(_state_direction(base_state, classifier_state), q_star[1]))
    parameter_scale = _direction_norm(trb_update) / (_direction_norm(direction) + 1e-24)
    variants.append(_save_and_evaluate("ParameterNormOnly", _apply_direction(base_state, direction, parameter_scale), {"source": "parameter_norm_only", "scale": parameter_scale, "matched_update_norm": _direction_norm(trb_update)}, f2_dir, roles, weak6, seed_dir, device, args))
    variants.append(_save_and_evaluate("ClassifierOnlyAlpha2p0", _compose_layer_state(base_state, direction, 0.0, 2.0), {"source": "frozen_classifier_only", "shared_scale": 0.0, "classifier_scale": 2.0}, f2_dir, roles, weak6, seed_dir, device, args))
    variants.append(_save_and_evaluate("ModelAverage0p5", _apply_direction(base_state, direction, 0.5), {"source": "paired_midpoint", "scale": 0.5}, f2_dir, roles, weak6, seed_dir, device, args))

    other_directions = [_state_direction(base_state, states[name]) for name in OTHER_MASK_ENDPOINTS]
    orth_direction = _orthogonal_task_subspace(direction, other_directions)
    orth_state, orth_diagnostics = _target_risk_matched_orthogonal_state(base_state, orth_direction, _direction_norm(trb_update), f2_dir, roles, device, args, fit_targets, fit_weak)
    orth_row = _save_and_evaluate("OrthogonalTaskSubspaceFunctionalMatched", orth_state, orth_diagnostics, f2_dir, roles, weak6, seed_dir, device, args)
    variants.append(orth_row)

    _write_json(seed_dir / "longtail_manifest.json", manifest.to_dict())
    np.savez_compressed(str(seed_dir / "role_indices.npz"), manifest=manifest.selected_indices, **ids)
    payload = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke), "data_sha256": data_sha256,
        "split_sha256": split.sha256, "manifest_sha256": manifest.sha256,
        "role_indices_sha256": _indices_sha256(*[ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")]),
        "f2_checkpoint": str(checkpoint_path), "f2_checkpoint_sha256": _sha256(checkpoint_path),
        "training": training, "variants": variants, "role_counts": {key: len(value) for key, value in ids.items()},
        "trb_ptv": trb_diagnostics, "orthogonal_control": orth_diagnostics,
        "identity_audit_passed": True, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v14_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def _improvement_gate(seed_rows, comparisons, aggregate):
    paired = fc._paired(seed_rows, "TRB_PTV", "FullAlpha1p5", 20267000)
    delta, trb, full = paired["aggregate"], comparisons["TRB_PTV"]["aggregate"], comparisons["FullAlpha1p5"]["aggregate"]
    worst = min(trb["per_label_weak_recall"][name]["mean"] for name in TARGET_NAMES) - min(full["per_label_weak_recall"][name]["mean"] for name in TARGET_NAMES)
    primary_ci = delta["target_mean_weak_recall"]["mean"] >= 0.005 and delta["target_mean_weak_recall"]["bootstrap_95_ci"][0] > 0
    label_ci = any(delta["per_label_weak_recall"][name]["mean"] >= 0.005 and delta["per_label_weak_recall"][name]["bootstrap_95_ci"][0] > 0 for name in TARGET_NAMES)
    checks = {"worst_target_improves_by_0p005": worst >= 0.005, "target_mean_weak_not_lower": delta["target_mean_weak_recall"]["mean"] >= 0, "macro_f1_noninferior_minus_0p003": delta["macro_f1"]["mean"] >= -0.003, "macro_aupr_noninferior_minus_0p003": delta["macro_aupr"]["mean"] >= -0.003, "target_fpr_delta_le_0p005": delta["target_mean_negative_fpr"]["mean"] <= 0.005, "one_primary_advantage_positive_ci": primary_ci or label_ci, "weak_effect_std_lower": trb["target_mean_weak_recall"]["std"] < full["target_mean_weak_recall"]["std"], "valid_and_finite_5_of_5": aggregate["TRB_PTV"]["valid_seed_count"] == 5 and paired["paired_seed_count"] == 5}
    return {"comparison": paired, "worst_target_margin": float(worst), "checks": checks, "passed": all(checks.values())}


def _mechanism_gate(seed_rows):
    def advantage(control, seed):
        paired = fc._paired(seed_rows, "TRB_PTV", control, seed); delta = paired["aggregate"]
        label = any(delta["per_label_weak_recall"][name]["mean"] >= 0.005 and delta["per_label_weak_recall"][name]["bootstrap_95_ci"][0] > 0 for name in TARGET_NAMES)
        mean = delta["target_mean_weak_recall"]["mean"] >= 0.005 and delta["target_mean_weak_recall"]["bootstrap_95_ci"][0] > 0
        return paired, bool(label or mean)
    parameter, parameter_ok = advantage("ParameterNormOnly", 20268000)
    classifier, classifier_ok = advantage("ClassifierOnlyAlpha2p0", 20268500)
    orthogonal, orthogonal_ok = advantage("OrthogonalTaskSubspaceFunctionalMatched", 20269000)
    checks = {"beats_parameter_norm_control": parameter_ok, "beats_classifier_only_control": classifier_ok and classifier["aggregate"]["macro_aupr"]["mean"] >= -0.003 and classifier["aggregate"]["macro_f1"]["mean"] >= -0.003, "beats_orthogonal_functional_control": orthogonal_ok and orthogonal["aggregate"]["macro_aupr"]["mean"] >= -0.003 and orthogonal["aggregate"]["macro_f1"]["mean"] >= -0.003, "shared_component_nonzero_5_of_5": all(float(row["trb_ptv"]["q_star"][0]) > 0 for row in seed_rows)}
    return {"checks": checks, "passed": all(checks.values()), "parameter_norm_comparison": parameter, "classifier_only_comparison": classifier, "orthogonal_comparison": orthogonal}


def run(args):
    config = load_config(args.config); _validate_config(config)
    identity = _read_json(args.identity_audit)
    if not identity.get("identity_audit_passed") or identity.get("evidence_role") != "development_v14" or identity.get("test_files_opened"):
        raise ValueError("development-v14 identity audit failed")
    if any(identity.get(name, True) for name in ("locked_test_used", "test_v2_used", "test_v3_used", "test_v14_used")):
        raise ValueError("forbidden role usage recorded")
    data_path = _data_path(config); data_sha256 = _sha256(data_path)
    if identity.get("sha256", "").lower() != data_sha256.lower(): raise ValueError("development-v14 train hash mismatch")
    labels6 = read_labels(data_path); parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameters), float(args.weak_threshold_db))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke: args.epochs = int(args.smoke_epochs)
    seed_rows = [_run_seed(args, config, labels6, weak6, int(seed), device, data_sha256) for seed in args.seeds]
    old = fc.VARIANTS; fc.VARIANTS = VARIANTS
    try: aggregate, comparisons = fc._aggregate(seed_rows)
    finally: fc.VARIANTS = old
    activation = activation_gate([np.asarray(row["trb_ptv"]["q_star"], dtype=np.float64) for row in seed_rows])
    if args.smoke:
        candidate = improvement = mechanism = simultaneous = {"passed": False, "reason": "smoke_only"}; decision = "smoke_only_no_method_claim"
    else:
        candidate = fc._candidate_gate(comparisons["TRB_PTV"], aggregate["TRB_PTV"])
        improvement = _improvement_gate(seed_rows, comparisons, aggregate); mechanism = _mechanism_gate(seed_rows); simultaneous = fc._simultaneous_gate(comparisons["TRB_PTV"])
        if aggregate["TRB_PTV"]["valid_seed_count"] < 5 or not activation["passed"]: decision = "stop_trb_ptv"
        elif candidate["passed"] and improvement["passed"] and mechanism["passed"] and simultaneous["passed"]: decision = "promote_to_full_canonical_baselines"
        else: decision = "stop_trb_ptv"
    payload = {"status": "completed", "experiment": "route_a_development_v14_trb_ptv", "evidence_level": "smoke" if args.smoke else "development", "smoke": bool(args.smoke), "device": str(device), "protocol": {"seeds": [int(seed) for seed in args.seeds], "variants": list(VARIANTS), "epochs": int(args.epochs), "q0": [1.5, 1.5], "solver": "SLSQP", "maxiter": int(args.trb_maxiter), "ftol": float(args.trb_ftol), "constraint_tolerance": float(args.trb_constraint_tolerance), "activation_norm_min": 0.05, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v14_used": False}, "data": {"path": str(data_path), "sha256": data_sha256, "sample_count": len(labels6), "identity_audit": identity}, "per_seed": seed_rows, "aggregate": aggregate, "comparisons_vs_c6": comparisons, "candidate_effect_gate": candidate, "improvement_vs_fullalpha_gate": improvement, "mechanism_specificity_gate": mechanism, "simultaneous_target_gate": simultaneous, "q_activation_gate": activation, "next_decision": decision, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v14_used": False}
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "route_a_development_v14_trb_ptv_summary.json", payload)
    _write_json(root / "run_receipt.json", {"status": "completed", "evidence_level": payload["evidence_level"], "data_sha256": data_sha256, "identity_audit_passed": True, "candidate_effect_gate_passed": candidate["passed"], "improvement_vs_fullalpha_gate_passed": improvement["passed"], "mechanism_specificity_gate_passed": mechanism["passed"], "simultaneous_target_gate_passed": simultaneous["passed"], "q_activation_gate_passed": activation["passed"], "next_decision": decision, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v14_used": False, "test_access_allowed": False})
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v14.yaml")
    parser.add_argument("--identity-audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v14_generation/identity_audit.json")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v14_trb_ptv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--io-batch-size", type=int, default=128); parser.add_argument("--eval-batch-size", type=int, default=128); parser.add_argument("--learning-rate", type=float, default=5e-4); parser.add_argument("--representation-lr-scale", type=float, default=0.1); parser.add_argument("--saw-loss-weight", type=float, default=0.2); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip", type=float, default=5.0); parser.add_argument("--max-negative-fpr", type=float, default=0.20); parser.add_argument("--weak-threshold-db", type=float, default=10.0); parser.add_argument("--cagrad-c", type=float, default=0.4); parser.add_argument("--cagrad-grid", type=int, default=101); parser.add_argument("--gradnorm-alpha", type=float, default=1.5); parser.add_argument("--gradnorm-weight-lr", type=float, default=0.025)
    parser.add_argument("--trb-maxiter", type=int, default=200); parser.add_argument("--trb-ftol", type=float, default=1e-12); parser.add_argument("--trb-constraint-tolerance", type=float, default=1e-7)
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2); parser.add_argument("--smoke-train-samples", type=int, default=128); parser.add_argument("--smoke-eval-samples", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps({key: value for key, value in run(parse_args()).items() if key in ("experiment", "evidence_level", "next_decision")}, ensure_ascii=False, indent=2))
