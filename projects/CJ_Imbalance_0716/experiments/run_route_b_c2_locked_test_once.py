"""One-shot locked-test evaluation for the frozen Route-B C2 safe router.

This runner never fits, selects, calibrates, or changes a threshold.  It only
replays the six frozen C2 checkpoints, local experts, and serialized routers.
Once the access marker is written, the output directory cannot be reused.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import get_class_names, select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.locked_test import sha256_file, validate_locked_test_access  # noqa: E402
from cjlt.engine.supervised_joint_trainer import compute_parameter_metrics  # noqa: E402
from cjlt.engine.thresholding import apply_thresholds  # noqa: E402
from cjlt.losses.parameter import select_parameter_targets  # noqa: E402
from cjlt.models.supervised_joint import temporal_local_distribution  # noqa: E402
from experiments.run_route_b_c2_frozen_local_replacement import (  # noqa: E402
    PROMOTED_TARGETS,
    _normalizers,
)
from experiments.run_route_b_c2_local_parameter_probe import (  # noqa: E402
    _load_model,
    _sha256,
    _target_info,
    _write_json,
)
from experiments.run_route_b_c2_visibility_safe_routing import (  # noqa: E402
    _gate_features,
    _load_expert,
)


ACCESS_MARKER = "locked_test_access_started.json"
FINAL_RECEIPT = "run_receipt.json"


def _resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _stable_sigmoid(values):
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _predict_serialized_router(state, features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("router features must be two-dimensional")
    if values.shape[1] != len(state.get("feature_names", [])):
        raise ValueError("router feature count differs from frozen state")
    if state["type"] == "constant":
        return np.full(
            len(values), float(state["constant_probability"]), dtype=np.float64
        )
    if state["type"] != "standardized_logistic_regression":
        raise ValueError("unsupported frozen router type: {}".format(state["type"]))
    mean = np.asarray(state["scaler_mean"], dtype=np.float64)
    scale = np.asarray(state["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(state["coefficient"], dtype=np.float64)
    if mean.shape != scale.shape or mean.shape != coefficient.shape:
        raise ValueError("serialized router arrays do not match")
    standardized = (values - mean[None, :]) / scale[None, :]
    logits = standardized @ coefficient + float(state["intercept"])
    return _stable_sigmoid(logits)


def _verify_frozen_source(path, expected_sha256):
    resolved = _resolve_project_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError("frozen source missing: {}".format(resolved))
    actual = sha256_file(resolved)
    if actual != str(expected_sha256).lower():
        raise ValueError(
            "frozen source hash mismatch for {}: expected {}, got {}".format(
                resolved, expected_sha256, actual
            )
        )
    return resolved


def _validate_freeze(payload):
    if payload.get("status") != "locked_test_configuration_frozen":
        raise ValueError("freeze status is not locked_test_configuration_frozen")
    if payload.get("locked_test_authorized") is not True:
        raise ValueError("freeze does not record locked-test authorization")
    protocol = payload["protocol"]
    if protocol.get("scope") != "closedset_6class_test.mat only":
        raise ValueError("locked-test scope drifted")
    if protocol.get("fit_or_selection_on_locked_test") is not False:
        raise ValueError("locked-test fitting must be disabled")
    if protocol.get("identity_consensus_beta") != 0:
        raise ValueError("official frozen route requires identity consensus beta=0")
    if protocol.get("predicted_context_safety") is not False:
        raise ValueError("official frozen route requires context safety off")
    seeds = [int(row["seed"]) for row in payload["seeds"]]
    if seeds != [2041, 2042, 2043, 2044, 2045, 2046]:
        raise ValueError("frozen seed list drifted")
    if not payload.get("implementation_sources"):
        raise ValueError("implementation source hashes are missing")


def _load_seed_sources(seed_row):
    sources = seed_row["sources"]
    return {
        key: _verify_frozen_source(row["path"], row["sha256"])
        for key, row in sources.items()
    }


def _validate_router_receipt(receipt, seed, checkpoint_path, expert_paths):
    if receipt.get("status") != "completed" or receipt.get("locked_test_used"):
        raise ValueError("router receipt is not a clean development receipt")
    if int(receipt["seed"]) != int(seed):
        raise ValueError("router receipt seed mismatch")
    if receipt["c2_checkpoint_sha256"] != _sha256(checkpoint_path):
        raise ValueError("router receipt C2 checkpoint mismatch")
    expected_experts = {
        "FMB.interference_bandwidth": _sha256(expert_paths[0]),
        "SinFM.frequency_deviation": _sha256(expert_paths[1]),
    }
    if receipt["expert_sha256"] != expected_experts:
        raise ValueError("router receipt expert hash mismatch")
    if receipt.get("classification_outputs_modified") is not False:
        raise ValueError("frozen router must not modify classification outputs")
    for target in receipt["targets"]:
        if float(target.get("chosen_identity_consensus_beta", 0.0)) != 0.0:
            raise ValueError("identity consensus is not part of the frozen route")
        if bool(target.get("chosen_predicted_context_safety", False)):
            raise ValueError("context safety is not part of the frozen route")


def _extract_predictions(
    model,
    experts,
    expert_checkpoints,
    test_path,
    sample_count,
    normalization,
    normalizer_mean,
    normalizer_scale,
    device,
    batch_size,
):
    indices = np.arange(int(sample_count), dtype=np.int64)
    dataset = V73SignalDataset(
        test_path,
        indices,
        class_mode="six",
        normalization=normalization,
        include_parameters=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    parts = {
        key: []
        for key in (
            "sample_ids",
            "raw_rms",
            "logits",
            "class_probabilities",
            "c2_predictions",
            "local_predictions",
            "local_z_rms",
            "local_z_max",
        )
    }
    mean_tensor = torch.as_tensor(
        normalizer_mean, device=device, dtype=torch.float32
    )
    scale_tensor = torch.as_tensor(
        normalizer_scale, device=device, dtype=torch.float32
    )
    class_names = get_class_names("six")
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                temporal = model.backbone.forward_temporal_features(signal)
                pooled = model.backbone.global_pool(temporal).squeeze(-1)
                logits = model.backbone.forward_classifier(pooled)
                embeddings = model.label_feature_bank(pooled)
                normalized_parameters = model.parameter_heads(embeddings)
                raw_parameters = (
                    normalized_parameters * scale_tensor[None] + mean_tensor[None]
                )
                local_features = temporal_local_distribution(temporal).float()
                local_predictions = []
                local_z_rms = []
                local_z_max = []
                for class_name, parameter_name in PROMOTED_TARGETS:
                    key = (class_name, parameter_name)
                    checkpoint = expert_checkpoints[key]
                    feature_mean = checkpoint["feature_mean"].to(
                        device=device, dtype=local_features.dtype
                    )
                    feature_std = checkpoint["feature_std"].to(
                        device=device, dtype=local_features.dtype
                    )
                    z = (local_features - feature_mean) / feature_std
                    prediction = experts[key](z) * float(
                        checkpoint["target_scale"]
                    ) + float(checkpoint["target_mean"])
                    local_predictions.append(prediction)
                    local_z_rms.append(z.square().mean(dim=1).sqrt())
                    local_z_max.append(z.abs().amax(dim=1))
                parts["sample_ids"].append(
                    torch.as_tensor(batch["sample_index"]).cpu().numpy()
                )
                parts["raw_rms"].append(np.asarray(batch["raw_rms"], dtype=np.float32))
                parts["logits"].append(logits.float().cpu().numpy())
                parts["class_probabilities"].append(logits.sigmoid().float().cpu().numpy())
                parts["c2_predictions"].append(raw_parameters.float().cpu().numpy())
                parts["local_predictions"].append(
                    torch.stack(local_predictions, dim=1).float().cpu().numpy()
                )
                parts["local_z_rms"].append(
                    torch.stack(local_z_rms, dim=1).float().cpu().numpy()
                )
                parts["local_z_max"].append(
                    torch.stack(local_z_max, dim=1).float().cpu().numpy()
                )
    finally:
        dataset.close()
    output = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    if not np.array_equal(output["sample_ids"], indices):
        raise ValueError("locked-test prediction order drifted")
    return output


def _target_metrics(c2, local, routed, expected, valid, scale, decision, labels):
    c2_error = np.abs(c2[valid] - expected[valid]) / float(scale)
    local_error = np.abs(local[valid] - expected[valid]) / float(scale)
    routed_error = np.abs(routed[valid] - expected[valid]) / float(scale)
    combinations = np.asarray(
        ["".join(str(int(value)) for value in row) for row in labels[valid]]
    )
    by_combination = {}
    for key in sorted(np.unique(combinations)):
        selected = combinations == key
        base = float(c2_error[selected].mean())
        final = float(routed_error[selected].mean())
        by_combination[str(key)] = {
            "count": int(selected.sum()),
            "c2_nmae": base,
            "routed_nmae": final,
            "relative_reduction": float((base - final) / max(base, 1.0e-12)),
            "route_coverage": float(decision[valid][selected].mean()),
        }
    c2_mean = float(c2_error.mean())
    local_mean = float(local_error.mean())
    routed_mean = float(routed_error.mean())
    return {
        "count": int(valid.sum()),
        "c2_nmae": c2_mean,
        "always_local_nmae": local_mean,
        "routed_nmae": routed_mean,
        "always_local_relative_reduction": float(
            (c2_mean - local_mean) / max(c2_mean, 1.0e-12)
        ),
        "routed_relative_reduction": float(
            (c2_mean - routed_mean) / max(c2_mean, 1.0e-12)
        ),
        "route_coverage": float(decision[valid].mean()),
        "oracle_local_better_rate": float((local_error < c2_error).mean()),
        "negative_combination_count": int(
            sum(row["relative_reduction"] < 0.0 for row in by_combination.values())
        ),
        "minimum_combination_reduction": float(
            min(row["relative_reduction"] for row in by_combination.values())
        ),
        "by_combination": by_combination,
    }


def _run_seed(seed_row, test_path, labels, targets, masks, device, batch_size, output_root):
    seed = int(seed_row["seed"])
    sources = _load_seed_sources(seed_row)
    expert_paths = [sources["fmb_expert"], sources["sinfm_expert"]]
    receipt = json.loads(sources["router_receipt"].read_text(encoding="utf-8"))
    _validate_router_receipt(receipt, seed, sources["c2_checkpoint"], expert_paths)
    run_dir = sources["c2_checkpoint"].parent
    model, checkpoint_path = _load_model(run_dir, device)
    if checkpoint_path.resolve() != sources["c2_checkpoint"]:
        raise ValueError("C2 checkpoint path drifted")
    _, normalizer_mean, normalizer_scale = _normalizers(run_dir)
    experts = {}
    expert_checkpoints = {}
    for (class_name, parameter_name), path in zip(PROMOTED_TARGETS, expert_paths):
        key = (class_name, parameter_name)
        experts[key], expert_checkpoints[key] = _load_expert(path, device)
    extracted = _extract_predictions(
        model,
        experts,
        expert_checkpoints,
        test_path,
        len(labels),
        "complex_rms",
        normalizer_mean,
        normalizer_scale,
        device,
        batch_size,
    )
    calibration = json.loads(sources["calibration"].read_text(encoding="utf-8"))
    calibrated_logits = apply_temperature(
        extracted["logits"], float(calibration["temperature"])
    )
    probabilities = _stable_sigmoid(calibrated_logits).astype(np.float32)
    class_predictions = apply_thresholds(probabilities, calibration["thresholds"])
    classification = compute_multilabel_metrics(
        labels,
        class_predictions,
        probabilities=probabilities,
        class_names=get_class_names("six"),
    )
    routed_predictions = extracted["c2_predictions"].copy()
    target_rows = []
    route_arrays = {}
    class_names = get_class_names("six")
    for target_offset, (class_name, parameter_name) in enumerate(PROMOTED_TARGETS):
        frozen = next(
            row
            for row in receipt["targets"]
            if row["class"] == class_name and row["parameter"] == parameter_name
        )
        class_index, parameter_index, _ = _target_info(
            class_names, class_name, parameter_name
        )
        payload = {
            "c2_predictions": np.stack(
                [
                    extracted["c2_predictions"][:, _target_info(class_names, c, p)[0], _target_info(class_names, c, p)[1]]
                    for c, p in PROMOTED_TARGETS
                ],
                axis=1,
            ),
            "local_predictions": extracted["local_predictions"],
            "local_z_rms": extracted["local_z_rms"],
            "local_z_max": extracted["local_z_max"],
            "class_probabilities": extracted["class_probabilities"],
            "raw_rms": extracted["raw_rms"],
        }
        variant = frozen["chosen_variant"]
        features = _gate_features(
            payload,
            target_offset,
            class_index,
            normalizer_mean[class_index, parameter_index],
            normalizer_scale[class_index, parameter_index],
            variant,
        )
        router_state = frozen["chosen_router_state"]
        if features.shape[1] != len(router_state["feature_names"]):
            raise ValueError("frozen feature variant does not match router state")
        routing_probability = _predict_serialized_router(router_state, features)
        decision = routing_probability >= float(frozen["chosen_threshold"])
        c2 = extracted["c2_predictions"][:, class_index, parameter_index]
        local = extracted["local_predictions"][:, target_offset]
        blended = c2 + float(frozen["chosen_local_weight"]) * (local - c2)
        routed = np.where(decision, blended, c2).astype(np.float32)
        routed_predictions[:, class_index, parameter_index] = routed
        valid = (
            (labels[:, class_index] == 1)
            & masks[:, class_index, parameter_index]
            & np.isfinite(targets[:, class_index, parameter_index])
        )
        metrics = _target_metrics(
            c2,
            local,
            routed,
            targets[:, class_index, parameter_index],
            valid,
            normalizer_scale[class_index, parameter_index],
            decision,
            labels,
        )
        target_rows.append(
            {
                "class": class_name,
                "parameter": parameter_name,
                "variant": variant,
                "threshold": float(frozen["chosen_threshold"]),
                "local_weight": float(frozen["chosen_local_weight"]),
                **metrics,
            }
        )
        prefix = "{}_{}".format(class_name, parameter_name)
        route_arrays[prefix + "_probability"] = routing_probability.astype(np.float32)
        route_arrays[prefix + "_decision"] = decision.astype(np.uint8)
        route_arrays[prefix + "_local_prediction"] = local.astype(np.float32)
    normalizer = SimpleNamespace(
        mean=torch.from_numpy(normalizer_mean.astype(np.float32)),
        scale=torch.from_numpy(normalizer_scale.astype(np.float32)),
    )
    c2_output = SimpleNamespace(
        parameter_predictions=extracted["c2_predictions"],
        parameter_targets=targets,
        parameter_masks=masks,
        targets=labels,
    )
    routed_output = SimpleNamespace(
        parameter_predictions=routed_predictions,
        parameter_targets=targets,
        parameter_masks=masks,
        targets=labels,
    )
    parameter_specs = {
        name: get_parameter_specs(name) for name in class_names
    }
    c2_metrics = compute_parameter_metrics(
        c2_output,
        class_names,
        parameter_specs,
        normalizer,
        class_predictions=class_predictions,
    )
    routed_metrics = compute_parameter_metrics(
        routed_output,
        class_names,
        parameter_specs,
        normalizer,
        class_predictions=class_predictions,
    )
    seed_root = output_root / "seed_{}".format(seed)
    seed_root.mkdir(parents=True, exist_ok=False)
    prediction_path = seed_root / "locked_test_predictions.npz"
    np.savez_compressed(
        str(prediction_path),
        sample_ids=extracted["sample_ids"],
        logits=extracted["logits"],
        probabilities=probabilities,
        class_predictions=class_predictions,
        labels=labels,
        c2_parameter_predictions=extracted["c2_predictions"],
        routed_parameter_predictions=routed_predictions,
        parameter_targets=targets,
        parameter_masks=masks,
        raw_rms=extracted["raw_rms"],
        **route_arrays,
    )
    result = {
        "status": "completed",
        "seed": seed,
        "locked_test_used": True,
        "classification_outputs_modified": False,
        "classification": classification,
        "c2_parameters": c2_metrics,
        "routed_parameters": routed_metrics,
        "overall_relative_macro_nmae_reduction": float(
            1.0
            - routed_metrics["oracle_macro_target_nmae"]
            / max(c2_metrics["oracle_macro_target_nmae"], 1.0e-12)
        ),
        "targets": target_rows,
        "source_hashes": {
            key: _sha256(path) for key, path in sources.items()
        },
        "predictions_sha256": _sha256(prediction_path),
    }
    _write_json(seed_root / "run_receipt.json", result)
    del model, experts, expert_checkpoints
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std()), "values": array.tolist()}


def _aggregate(rows):
    target_rows = []
    per_seed_combined = []
    for class_name, parameter_name in PROMOTED_TARGETS:
        selected = [
            next(
                target
                for target in row["targets"]
                if target["class"] == class_name and target["parameter"] == parameter_name
            )
            for row in rows
        ]
        c2 = np.asarray([row["c2_nmae"] for row in selected], dtype=np.float64)
        routed = np.asarray([row["routed_nmae"] for row in selected], dtype=np.float64)
        reductions = 1.0 - routed / c2
        target_rows.append(
            {
                "class": class_name,
                "parameter": parameter_name,
                "c2_nmae": _mean_std(c2),
                "routed_nmae": _mean_std(routed),
                "relative_reduction": _mean_std(reductions),
                "positive_seed_count": int((reductions > 0.0).sum()),
                "route_coverage": _mean_std([row["route_coverage"] for row in selected]),
            }
        )
    for row in rows:
        c2 = np.mean([target["c2_nmae"] for target in row["targets"]])
        routed = np.mean([target["routed_nmae"] for target in row["targets"]])
        per_seed_combined.append(1.0 - routed / c2)
    c2_macro = [row["c2_parameters"]["oracle_macro_target_nmae"] for row in rows]
    routed_macro = [row["routed_parameters"]["oracle_macro_target_nmae"] for row in rows]
    all_target_reduction = 1.0 - np.asarray(routed_macro) / np.asarray(c2_macro)
    return {
        "seed_count": len(rows),
        "classification_macro_aupr": _mean_std(
            [row["classification"]["macro_aupr"] for row in rows]
        ),
        "classification_macro_f1": _mean_std(
            [row["classification"]["macro_f1"] for row in rows]
        ),
        "targets": target_rows,
        "two_target_combined_relative_reduction": _mean_std(per_seed_combined),
        "two_target_positive_seed_count": int(
            (np.asarray(per_seed_combined) > 0.0).sum()
        ),
        "all_12_target_c2_macro_nmae": _mean_std(c2_macro),
        "all_12_target_routed_macro_nmae": _mean_std(routed_macro),
        "all_12_target_relative_reduction": _mean_std(all_target_reduction),
        "all_12_target_positive_seed_count": int((all_target_reduction > 0.0).sum()),
    }


def run(args):
    output_root = Path(args.output_root)
    marker_path = output_root / ACCESS_MARKER
    final_path = output_root / FINAL_RECEIPT
    if marker_path.exists() or final_path.exists() or output_root.exists():
        raise RuntimeError(
            "one-shot output already exists; locked test will not be accessed again: {}".format(
                output_root
            )
        )
    freeze_path = validate_locked_test_access(
        args.authorize_locked_test,
        args.configuration_freeze,
        args.expected_freeze_sha256,
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _validate_freeze(freeze)
    for source in freeze["implementation_sources"]:
        _verify_frozen_source(source["path"], source["sha256"])
    for seed_row in freeze["seeds"]:
        _load_seed_sources(seed_row)
    test_path = Path(freeze["locked_test"]["path"]).resolve()
    if test_path.name != "closedset_6class_test.mat" or "power_sweep" in str(test_path).lower():
        raise ValueError("only the base closedset_6class_test.mat is authorized")
    if not test_path.is_file():
        raise FileNotFoundError("locked test missing: {}".format(test_path))
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(
        marker_path,
        {
            "status": "locked_test_access_started",
            "authorization": "user replied ok to explicit locked-test authorization request",
            "configuration_freeze": str(freeze_path),
            "configuration_freeze_sha256": sha256_file(freeze_path),
            "locked_test_path": str(test_path),
            "scope": "closedset_6class_test.mat only",
            "rerun_permitted": False,
        },
    )
    locked_test_sha256 = sha256_file(test_path)
    labels = select_labels(read_labels(test_path), "six")
    expected_count = int(freeze["locked_test"]["expected_sample_count"])
    if len(labels) != expected_count:
        raise ValueError(
            "locked-test sample count mismatch: expected {}, got {}".format(
                expected_count, len(labels)
            )
        )
    values79, mask79 = read_parameters(test_path)
    target_tensor, mask_tensor = select_parameter_targets(
        values79, mask79, get_class_names("six")
    )
    targets = target_tensor.numpy().astype(np.float32)
    masks = mask_tensor.numpy().astype(bool)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for seed_row in freeze["seeds"]:
        print("locked test seed={}".format(seed_row["seed"]), flush=True)
        rows.append(
            _run_seed(
                seed_row,
                test_path,
                labels,
                targets,
                masks,
                device,
                args.batch_size,
                output_root,
            )
        )
    payload = {
        "status": "completed",
        "locked_test_used": True,
        "one_shot": True,
        "rerun_permitted": False,
        "scope": "closedset_6class_test.mat only",
        "locked_test_path": str(test_path),
        "locked_test_sha256": locked_test_sha256,
        "configuration_freeze": str(freeze_path),
        "configuration_freeze_sha256": sha256_file(freeze_path),
        "device": str(device),
        "sample_count": int(len(labels)),
        "fit_or_selection_on_locked_test": False,
        "classification_outputs_modified": False,
        "aggregate": _aggregate(rows),
        "runs": rows,
    }
    _write_json(final_path, payload)
    print(json.dumps({"aggregate": payload["aggregate"]}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--configuration-freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/student_b/route_b_c2_locked_test_once_20260726",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
