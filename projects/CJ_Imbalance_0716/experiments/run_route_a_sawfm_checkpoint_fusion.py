"""Validate logit and parameter interpolation of early/late Base-DB checkpoints."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.locked_test import sha256_file  # noqa: E402
from cjlt.engine.trainer import predict_loader  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.student_a import StudentAModel  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


TAIL_NAME = "SawFM"
TAIL_INDEX = 1
CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
METHODS = ("logit", "weight")
METRICS = ("macro_aupr", "tail_aupr", "macro_f1", "hamming_loss", "tail_f1")


def _sigmoid(values):
    logits = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _endpoint_dir(root, variant, seed):
    return Path(root) / "runs" / TAIL_NAME / variant / "seed_{}".format(seed)


def _scientific_arguments(directory):
    payload = _read(Path(directory) / "scientific_config.json")
    return payload["scientific_config"]["arguments"]


def _build_model(arguments, num_classes):
    backbone = build_backbone(
        arguments["backbone"],
        num_classes=int(num_classes),
        width=int(arguments["width"]),
        dropout=float(arguments["dropout"]),
    )
    return StudentAModel(
        backbone,
        relation_dim=int(arguments["relation_dim"]),
        relation_dropout=float(arguments["relation_dropout"]),
        conditioning_mode=str(arguments["conditioning_mode"]),
        enable_parameter_heads=False,
    )


def _load_state(path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing model_state_dict: {}".format(path))
    return checkpoint["model_state_dict"]


def _interpolate_state(late_state, early_state, alpha):
    output = {}
    weight = float(alpha)
    for key in late_state:
        late = late_state[key]
        early = early_state[key]
        if late.shape != early.shape or late.dtype != early.dtype:
            raise ValueError("Endpoint state mismatch at {}".format(key))
        if torch.is_floating_point(late) or torch.is_complex(late):
            output[key] = late.mul(1.0 - weight).add(early, alpha=weight)
        else:
            output[key] = early.clone() if weight >= 0.5 else late.clone()
    return output


def _metrics(logits, targets, temperature=1.0):
    calibrated = apply_temperature(np.asarray(logits, dtype=np.float32), temperature)
    probability = _sigmoid(calibrated)
    prediction = (probability >= 0.5).astype(np.uint8)
    result = compute_multilabel_metrics(
        targets,
        prediction,
        probabilities=probability,
        class_names=CLASS_NAMES,
        tail_indices=[TAIL_INDEX],
    )
    tail = result["per_label"][TAIL_INDEX]
    return {
        "macro_aupr": result["macro_aupr"],
        "tail_aupr": tail["aupr"],
        "macro_f1": result["macro_f1"],
        "hamming_loss": result["hamming_loss"],
        "tail_f1": tail["f1"],
    }


def _delta(candidate, baseline):
    return {"delta_{}".format(key): candidate[key] - baseline[key] for key in METRICS}


def _predict(model, loader, device, use_amp):
    output = predict_loader(model, loader, device, use_amp=use_amp)
    return {
        "logits": output.logits,
        "targets": output.targets,
        "sample_ids": output.sample_ids,
    }


def _assert_outputs_match(left, right, role, seed):
    if not np.array_equal(left["sample_ids"], right["sample_ids"]):
        raise ValueError("Sample-id mismatch role={} seed={}".format(role, seed))
    if not np.array_equal(left["targets"], right["targets"]):
        raise ValueError("Target mismatch role={} seed={}".format(role, seed))


def _collect_seed(args, seed, config, device):
    late_dir = _endpoint_dir(args.db_macro_root, "BaseDB", seed)
    early_dir = _endpoint_dir(args.db_tail_root, "DB-Tail", seed)
    late_receipt = _read(late_dir / "run_receipt.json")
    early_receipt = _read(early_dir / "run_receipt.json")
    if late_receipt["locked_test_used"] or early_receipt["locked_test_used"]:
        raise ValueError("Endpoint used locked test")
    if late_receipt["split_sha256"] != early_receipt["split_sha256"]:
        raise ValueError("Endpoint split mismatch seed={}".format(seed))
    if late_receipt["manifest_sha256"] != early_receipt["manifest_sha256"]:
        raise ValueError("Endpoint manifest mismatch seed={}".format(seed))

    late_args = _scientific_arguments(late_dir)
    early_args = _scientific_arguments(early_dir)
    architecture_keys = (
        "backbone", "width", "dropout", "relation_dim",
        "relation_dropout", "conditioning_mode", "variant",
    )
    for key in architecture_keys:
        if late_args[key] != early_args[key]:
            raise ValueError("Endpoint architecture mismatch at {}".format(key))

    plan = build_run_plan(
        config,
        loss_name="db",
        seed=int(seed),
        max_train_samples=0,
        max_eval_samples=int(args.max_eval_samples),
    )
    if plan["allow_locked_test"]:
        raise ValueError("Configuration authorizes locked test")
    if plan["split_sha256"] != late_receipt["split_sha256"]:
        raise ValueError("Rebuilt split does not match endpoint")
    common = {
        "path": _data_path(config),
        "class_mode": plan["class_mode"],
        "normalization": config["normalization"],
        "include_parameters": False,
    }
    calibration_dataset = V73SignalDataset(
        indices=plan["calibration_indices"], **common
    )
    validation_dataset = V73SignalDataset(
        indices=plan["threshold_indices"], **common
    )
    loader_args = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.workers),
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        "calibration": DataLoader(calibration_dataset, shuffle=False, **loader_args),
        "validation": DataLoader(validation_dataset, shuffle=False, **loader_args),
    }
    late_path = late_dir / "best_model.pth"
    early_path = early_dir / "best_model.pth"
    late_state = _load_state(late_path)
    early_state = _load_state(early_path)
    model = _build_model(late_args, plan["num_classes"]).to(device)
    use_amp = bool(args.use_amp and device.type == "cuda")
    endpoint_outputs = {"late": {}, "early": {}}
    try:
        for endpoint, state in (("late", late_state), ("early", early_state)):
            model.load_state_dict(state, strict=True)
            for role, loader in loaders.items():
                endpoint_outputs[endpoint][role] = _predict(model, loader, device, use_amp)
        for role in loaders:
            _assert_outputs_match(
                endpoint_outputs["late"][role], endpoint_outputs["early"][role], role, seed
            )
        calibration_ids = endpoint_outputs["late"]["calibration"]["sample_ids"]
        validation_ids = endpoint_outputs["late"]["validation"]["sample_ids"]
        role_overlap = np.intersect1d(calibration_ids, validation_ids)
        if role_overlap.size:
            raise ValueError("Calibration/fusion-validation leakage seed={}".format(seed))
        if len(calibration_ids) != int(args.max_eval_samples):
            raise ValueError("Unexpected calibration sample count")
        if len(validation_ids) != int(args.max_eval_samples):
            raise ValueError("Unexpected fusion-validation sample count")

        logit_grid = []
        weight_grid = []
        weight_outputs = {}
        for alpha in args.alphas:
            alpha_value = float(alpha)
            logit_calibration = (
                (1.0 - alpha_value) * endpoint_outputs["late"]["calibration"]["logits"]
                + alpha_value * endpoint_outputs["early"]["calibration"]["logits"]
            )
            logit_validation = (
                (1.0 - alpha_value) * endpoint_outputs["late"]["validation"]["logits"]
                + alpha_value * endpoint_outputs["early"]["validation"]["logits"]
            )
            logit_grid.append(
                {
                    "seed": int(seed),
                    "alpha": alpha_value,
                    "calibration": _metrics(
                        logit_calibration,
                        endpoint_outputs["late"]["calibration"]["targets"],
                    ),
                }
            )
            weight_state = _interpolate_state(late_state, early_state, alpha_value)
            model.load_state_dict(weight_state, strict=True)
            weight_calibration = _predict(model, loaders["calibration"], device, use_amp)
            weight_validation = _predict(model, loaders["validation"], device, use_amp)
            _assert_outputs_match(
                endpoint_outputs["late"]["calibration"], weight_calibration,
                "weight_calibration", seed,
            )
            _assert_outputs_match(
                endpoint_outputs["late"]["validation"], weight_validation,
                "weight_validation", seed,
            )
            weight_grid.append(
                {
                    "seed": int(seed),
                    "alpha": alpha_value,
                    "calibration": _metrics(
                        weight_calibration["logits"], weight_calibration["targets"]
                    ),
                }
            )
            weight_outputs[alpha_value] = {
                "calibration": weight_calibration["logits"],
                "validation": weight_validation["logits"],
            }
            if alpha_value == 0.0:
                maximum = max(
                    float(np.max(np.abs(weight_calibration["logits"] - endpoint_outputs["late"]["calibration"]["logits"]))),
                    float(np.max(np.abs(weight_validation["logits"] - endpoint_outputs["late"]["validation"]["logits"]))),
                )
                if maximum > 1e-6:
                    raise ValueError("Weight alpha=0 does not reproduce late endpoint")
            if alpha_value == 1.0:
                maximum = max(
                    float(np.max(np.abs(weight_calibration["logits"] - endpoint_outputs["early"]["calibration"]["logits"]))),
                    float(np.max(np.abs(weight_validation["logits"] - endpoint_outputs["early"]["validation"]["logits"]))),
                )
                if maximum > 1e-6:
                    raise ValueError("Weight alpha=1 does not reproduce early endpoint")

        return {
            "seed": int(seed),
            "split_sha256": late_receipt["split_sha256"],
            "manifest_sha256": late_receipt["manifest_sha256"],
            "late_checkpoint": str(late_path.resolve()),
            "late_checkpoint_sha256": sha256_file(late_path),
            "early_checkpoint": str(early_path.resolve()),
            "early_checkpoint_sha256": sha256_file(early_path),
            "late_best_epoch": int(late_receipt["best_epoch"]),
            "early_best_epoch": int(early_receipt["best_epoch"]),
            "calibration_sample_count": int(len(calibration_ids)),
            "validation_sample_count": int(len(validation_ids)),
            "calibration_validation_overlap": int(role_overlap.size),
            "endpoint_outputs": endpoint_outputs,
            "logit_grid": logit_grid,
            "weight_grid": weight_grid,
            "weight_outputs": weight_outputs,
            "late_state": late_state,
            "early_state": early_state,
        }
    finally:
        calibration_dataset.close()
        validation_dataset.close()


def _select_alpha(seed_results, method, alphas):
    rows = []
    grid_key = "{}_grid".format(method)
    required_seeds = max(1, int(np.ceil(2.0 * len(seed_results) / 3.0)))
    for alpha in alphas:
        alpha_value = float(alpha)
        per_seed = []
        for result in seed_results:
            selected = next(
                row for row in result[grid_key] if row["alpha"] == alpha_value
            )
            baseline = next(
                row for row in result[grid_key] if row["alpha"] == 0.0
            )
            per_seed.append(
                {
                    "seed": result["seed"],
                    **_delta(selected["calibration"], baseline["calibration"]),
                }
            )
        macro_values = [row["delta_macro_aupr"] for row in per_seed]
        tail_values = [row["delta_tail_aupr"] for row in per_seed]
        rows.append(
            {
                "method": method,
                "alpha": alpha_value,
                "eligible": (
                    float(np.mean(macro_values)) >= -0.005
                    and sum(value >= -0.005 for value in macro_values)
                    >= required_seeds
                ),
                "delta_macro_aupr": _mean_ci(macro_values, 21000 + int(alpha_value * 100)),
                "delta_tail_aupr": _mean_ci(tail_values, 22000 + int(alpha_value * 100)),
                "per_seed": per_seed,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        selected = next(row for row in rows if row["alpha"] == 0.0)
    else:
        selected = max(
            eligible,
            key=lambda row: (
                row["delta_tail_aupr"]["mean"],
                row["delta_macro_aupr"]["mean"],
                -row["alpha"],
            ),
        )
    return rows, selected


def _validation_logits(result, method, alpha):
    if method == "logit":
        late = result["endpoint_outputs"]["late"]
        early = result["endpoint_outputs"]["early"]
        return {
            "calibration": (
                (1.0 - alpha) * late["calibration"]["logits"]
                + alpha * early["calibration"]["logits"]
            ),
            "validation": (
                (1.0 - alpha) * late["validation"]["logits"]
                + alpha * early["validation"]["logits"]
            ),
        }
    return result["weight_outputs"][float(alpha)]


def _validate_method(seed_results, method, selected_alpha, output_root):
    rows = []
    required_seeds = max(1, int(np.ceil(2.0 * len(seed_results) / 3.0)))
    for result in seed_results:
        alpha_logits = _validation_logits(result, method, selected_alpha)
        late = result["endpoint_outputs"]["late"]
        temperature = fit_temperature(
            alpha_logits["calibration"],
            late["calibration"]["targets"],
            source_role="calibration",
        )
        baseline_temperature = fit_temperature(
            late["calibration"]["logits"],
            late["calibration"]["targets"],
            source_role="calibration",
        )
        candidate_metrics = _metrics(
            alpha_logits["validation"],
            late["validation"]["targets"],
            temperature.temperature,
        )
        baseline_metrics = _metrics(
            late["validation"]["logits"],
            late["validation"]["targets"],
            baseline_temperature.temperature,
        )
        delta = _delta(candidate_metrics, baseline_metrics)
        row = {
            "method": method,
            "seed": result["seed"],
            "alpha": float(selected_alpha),
            "temperature": temperature.temperature,
            "baseline_temperature": baseline_temperature.temperature,
            "candidate": candidate_metrics,
            "baseline": baseline_metrics,
            "delta": delta,
        }
        seed_dir = Path(output_root) / "runs" / "seed_{}".format(result["seed"])
        seed_dir.mkdir(parents=True, exist_ok=True)
        if method == "weight":
            artifact_path = seed_dir / "weight_fusion_model.pth"
            torch.save(
                {
                    "model_state_dict": _interpolate_state(
                        result["late_state"], result["early_state"], selected_alpha
                    ),
                    "alpha": float(selected_alpha),
                    "late_checkpoint": result["late_checkpoint"],
                    "late_checkpoint_sha256": result["late_checkpoint_sha256"],
                    "early_checkpoint": result["early_checkpoint"],
                    "early_checkpoint_sha256": result["early_checkpoint_sha256"],
                },
                str(artifact_path),
            )
            row["artifact"] = {
                "path": str(artifact_path.resolve()),
                "sha256": sha256_file(artifact_path),
                "deployment_models": 1,
            }
        else:
            artifact_path = seed_dir / "logit_fusion_manifest.json"
            _write(
                artifact_path,
                {
                    "alpha": float(selected_alpha),
                    "temperature": temperature.temperature,
                    "late_checkpoint": result["late_checkpoint"],
                    "late_checkpoint_sha256": result["late_checkpoint_sha256"],
                    "early_checkpoint": result["early_checkpoint"],
                    "early_checkpoint_sha256": result["early_checkpoint_sha256"],
                    "deployment_models": 2,
                },
            )
            row["artifact"] = {
                "path": str(artifact_path.resolve()),
                "sha256": sha256_file(artifact_path),
                "deployment_models": 2,
            }
        rows.append(row)
        np.savez_compressed(
            str(seed_dir / "{}_fusion_predictions.npz".format(method)),
            validation_sample_ids=late["validation"]["sample_ids"],
            validation_targets=late["validation"]["targets"],
            baseline_logits=late["validation"]["logits"],
            fused_logits=alpha_logits["validation"],
            alpha=np.asarray([selected_alpha], dtype=np.float32),
        )
    aggregate = {}
    for index, metric in enumerate(METRICS):
        aggregate["delta_{}".format(metric)] = _mean_ci(
            [row["delta"]["delta_{}".format(metric)] for row in rows],
            23000 + index,
        )
    aggregate["macro_noninferior_seeds"] = sum(
        row["delta"]["delta_macro_aupr"] >= -0.005 for row in rows
    )
    aggregate["tail_positive_seeds"] = sum(
        row["delta"]["delta_tail_aupr"] > 0 for row in rows
    )
    aggregate["passed"] = (
        aggregate["delta_macro_aupr"]["mean"] >= -0.005
        and aggregate["delta_tail_aupr"]["mean"] >= 0.005
        and aggregate["macro_noninferior_seeds"] >= required_seeds
        and aggregate["tail_positive_seeds"] >= required_seeds
    )
    return rows, aggregate


def run(args):
    config = load_config(args.config)
    if bool(config.get("allow_locked_test", False)):
        raise ValueError("Locked test is authorized in config; fusion run refused")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_results = [
        _collect_seed(args, seed, config, device) for seed in args.seeds
    ]
    selection = {}
    validation = {}
    gate = {}
    for method in METHODS:
        grid, selected_by_grid = _select_alpha(seed_results, method, args.alphas)
        fixed_alpha = (
            args.fixed_logit_alpha if method == "logit" else args.fixed_weight_alpha
        )
        if fixed_alpha is None:
            selected = selected_by_grid
            selection_mode = "calibration_grid"
        else:
            selected = next(
                row for row in grid
                if abs(row["alpha"] - float(fixed_alpha)) < 1e-12
            )
            selection_mode = "fixed_from_prior_discovery"
        rows, aggregate = _validate_method(
            seed_results, method, selected["alpha"], args.output_root
        )
        selection[method] = {
            "selected_alpha": selected["alpha"],
            "selection_mode": selection_mode,
            "selected_calibration_effect": selected,
            "grid": grid,
        }
        validation[method] = {"per_seed": rows, "aggregate": aggregate}
        gate[method] = bool(aggregate["passed"])
    gate["passed"] = any(gate.values())
    gate["decision"] = (
        "advance_static_checkpoint_fusion"
        if gate["passed"]
        else "static_fusion_failed_move_to_training_time_retention"
    )
    endpoint_audit = []
    for result in seed_results:
        endpoint_audit.append(
            {
                key: value for key, value in result.items()
                if key not in (
                    "endpoint_outputs", "logit_grid", "weight_grid", "weight_outputs",
                    "late_state", "early_state",
                )
            }
        )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds),
            "alphas": [float(value) for value in args.alphas],
            "alpha_selection_role": (
                "fixed_from_prior_discovery"
                if args.fixed_logit_alpha is not None
                and args.fixed_weight_alpha is not None
                else "calibration"
            ),
            "validation_role": "threshold_selection_repurposed_as_fusion_validation",
            "threshold_selection_performed": False,
            "fixed_threshold": 0.5,
            "max_eval_samples": int(args.max_eval_samples),
        },
        "gate": gate,
        "selection": selection,
        "validation": validation,
        "endpoint_audit": endpoint_audit,
    }
    output = Path(args.output_root) / "route_a_sawfm_checkpoint_fusion_summary.json"
    _write(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--db-macro-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_six_class_head_db_screen",
    )
    parser.add_argument(
        "--db-tail-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_loss_selection_factorial",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_checkpoint_fusion",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fixed-logit-alpha", type=float, default=None)
    parser.add_argument("--fixed-weight-alpha", type=float, default=None)
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
