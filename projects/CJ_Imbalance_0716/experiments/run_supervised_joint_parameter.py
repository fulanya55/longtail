"""Run supervised joint classification/parameter experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import get_class_names, select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.run_manager import (  # noqa: E402
    configuration_sha256,
    decide_run,
    stable_run_arguments,
)
from cjlt.engine.supervised_joint_trainer import (  # noqa: E402
    compute_parameter_metrics,
    predict_joint,
    train_supervised_joint,
)
from cjlt.engine.thresholding import (  # noqa: E402
    apply_thresholds,
    select_per_label_f1_thresholds,
)
from cjlt.losses.classification import build_classification_loss  # noqa: E402
from cjlt.losses.parameter import ParameterNormalizer, select_parameter_targets  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone, count_trainable_parameters  # noqa: E402
from cjlt.models.supervised_joint import SupervisedJointParameterModel  # noqa: E402
from experiments.run_shared import (  # noqa: E402
    _data_path,
    _sigmoid,
    _write_history,
    _write_json,
    build_run_plan,
    load_config,
)


VARIANTS = {
    "C0": {
        "parameter_heads": False,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
    },
    "C1": {
        "parameter_heads": True,
        "detach_parameter_backbone": True,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
    },
    "C2": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
        "gradient_strategy": "standard",
        "blocked_parameter_targets": [],
    },
    "C2_SD": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
        "gradient_strategy": "selective",
        "blocked_parameter_targets": [["MFT", "num_targets"], ["VFT", "Doppler"]],
    },
    "C2_GP": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
        "gradient_strategy": "target_pcgrad",
        "blocked_parameter_targets": [],
    },
    "C2_IGP": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
        "gradient_strategy": "target_pcgrad",
        "blocked_parameter_targets": [["MFT", "num_targets"], ["VFT", "Doppler"]],
    },
    "C2_LFMB": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
        "gradient_strategy": "standard",
        "blocked_parameter_targets": [],
        "local_parameter_targets": [[0, 1]],
    },
    "C2_LSIN": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
        "gradient_strategy": "standard",
        "blocked_parameter_targets": [],
        "local_parameter_targets": [[2, 1]],
    },
    "C2_LFS": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
        "gradient_strategy": "standard",
        "blocked_parameter_targets": [],
        "local_parameter_targets": [[0, 1], [2, 1]],
    },
    "C3": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.2,
        "ramp_end_fraction": 0.4,
        "mft_count_classifier": False,
        "vft_spectral_head": False,
    },
    "C2_MFT_CE": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": True,
        "vft_spectral_head": False,
    },
    "C2_VFT_FFT": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": False,
        "vft_spectral_head": True,
    },
    "C2_TYPED_FFT": {
        "parameter_heads": True,
        "detach_parameter_backbone": False,
        "warmup_fraction": 0.0,
        "ramp_end_fraction": 0.0,
        "mft_count_classifier": True,
        "vft_spectral_head": True,
    },
}


def _indices_sha256(indices):
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _parameter_transfer_configuration(class_names, blocked_targets):
    names = []
    mask = []
    blocked = {(str(class_name), str(parameter_name)) for class_name, parameter_name in blocked_targets}
    for class_name in class_names:
        class_mask = []
        for spec in get_parameter_specs(class_name):
            names.append("{}_{}".format(class_name, spec.name))
            class_mask.append((str(class_name), str(spec.name)) not in blocked)
        mask.append(class_mask)
    return names, mask


def _gradient_summary(history):
    cosine_values = {}
    negative_fractions = []
    projected_counts = []
    for row in history:
        if "negative_gradient_fraction" in row:
            negative_fractions.append(float(row["negative_gradient_fraction"]))
        if "projected_target_count" in row:
            projected_counts.append(float(row["projected_target_count"]))
        for key, value in row.items():
            if key.startswith("grad_cos_") and np.isfinite(float(value)):
                cosine_values.setdefault(key[len("grad_cos_"):], []).append(float(value))
    return {
        "per_target_mean_cosine": {
            key: float(np.mean(values)) for key, values in sorted(cosine_values.items())
        },
        "mean_negative_gradient_fraction": (
            float(np.mean(negative_fractions)) if negative_fractions else None
        ),
        "mean_projected_target_count": (
            float(np.mean(projected_counts)) if projected_counts else None
        ),
    }


def _stratified_limit(indices, labels, maximum, seed):
    values = np.asarray(indices, dtype=np.int64)
    count = int(maximum)
    if count <= 0 or len(values) <= count:
        return values.copy()
    target = np.asarray(labels, dtype=np.uint8)
    powers = np.left_shift(1, np.arange(target.shape[1], dtype=np.int64))
    codes = (target[values].astype(np.int64) * powers[None, :]).sum(axis=1)
    rng = np.random.RandomState(int(seed))
    groups = []
    for code in sorted(np.unique(codes)):
        group = values[codes == code].copy()
        rng.shuffle(group)
        groups.append(group)
    offsets = np.zeros(len(groups), dtype=np.int64)
    selected = []
    while len(selected) < count:
        progressed = False
        for group_index, group in enumerate(groups):
            if offsets[group_index] < len(group):
                selected.append(int(group[offsets[group_index]]))
                offsets[group_index] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _parameter_normalizer(data_path, indices, class_names, class_mode):
    values79, mask79 = read_parameters(data_path)
    targets, masks = select_parameter_targets(
        values79[indices], mask79[indices], class_names
    )
    labels6 = read_labels(data_path)[indices]
    labels = labels6 if str(class_mode) == "six" else labels6[:, [0, 2, 3, 5]]
    labels = torch.from_numpy(labels).float()
    valid = torch.logical_and(masks, labels[:, :, None] > 0.5)
    return ParameterNormalizer().fit(targets, valid)


def _normalizer_payload(normalizer, class_names):
    rows = []
    for class_index, class_name in enumerate(class_names):
        for parameter_index, spec in enumerate(get_parameter_specs(class_name)):
            rows.append(
                {
                    "class": class_name,
                    "parameter": spec.name,
                    "unit": spec.unit,
                    "column_index": spec.column_index,
                    "mean": float(normalizer.mean[class_index, parameter_index]),
                    "scale": float(normalizer.scale[class_index, parameter_index]),
                }
            )
    return rows


def _build_loader(data_path, config, indices, batch_size, workers, shuffle, seed):
    dataset = V73SignalDataset(
        path=data_path,
        indices=indices,
        class_mode=str(config["class_mode"]),
        normalization=str(config["normalization"]),
        include_parameters=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=(torch.Generator().manual_seed(int(seed)) if shuffle else None),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def _prediction_metrics(output, temperature, thresholds, class_names):
    calibrated_logits = apply_temperature(output.logits, float(temperature))
    probabilities = _sigmoid(calibrated_logits)
    predictions = apply_thresholds(probabilities, thresholds)
    metrics = compute_multilabel_metrics(
        output.targets,
        predictions,
        probabilities=probabilities,
        class_names=class_names,
    )
    return probabilities, predictions, metrics


def run_one(args, variant_name, seed):
    config = load_config(args.config)
    variant = VARIANTS[variant_name]
    plan = build_run_plan(
        config,
        loss_name=args.loss,
        seed=int(seed),
        max_train_samples=0,
        max_eval_samples=0,
    )
    data_path = _data_path(config)
    labels = select_labels(read_labels(data_path), str(config["class_mode"]))
    for offset, (key, count_key, maximum) in enumerate(
        (
            ("train_indices", "train_samples", args.max_train_samples),
            ("early_stop_indices", "early_stop_samples", args.max_eval_samples),
            ("calibration_indices", "calibration_samples", args.max_eval_samples),
            ("threshold_indices", "threshold_samples", args.max_eval_samples),
        )
    ):
        plan[key] = _stratified_limit(
            plan[key], labels, maximum, int(seed) + 1001 + offset
        )
        plan[count_key] = int(len(plan[key]))
        plan[key.replace("_indices", "_sample_sha256")] = _indices_sha256(plan[key])
    printable_plan = {
        key: value for key, value in plan.items() if not key.endswith("_indices")
    }
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else PROJECT_ROOT / "outputs" / "student_b" / "supervised_joint_parameter"
    )
    output_dir = output_root / variant_name / "seed_{}".format(seed)
    stable_arguments = dict(vars(args))
    stable_arguments.update({"variant": variant_name, "seed": int(seed)})
    config_hash = configuration_sha256(
        {
            "config": config,
            "arguments": stable_run_arguments(stable_arguments),
            "plan": printable_plan,
            "variant": variant,
        }
    )
    decision = decide_run(output_dir, config_hash, resume=args.resume)
    if decision.action == "skip":
        print("Skipping completed run: {}".format(output_dir))
        return decision.receipt
    if args.dry_run:
        payload = {
            "variant": variant_name,
            "seed": int(seed),
            "variant_flags": variant,
            "plan": printable_plan,
            "output_dir": str(output_dir),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    class_names = get_class_names(str(config["class_mode"]))
    train_dataset, train_loader = _build_loader(
        data_path,
        config,
        plan["train_indices"],
        args.batch_size,
        args.workers,
        True,
        int(seed) + 101,
    )
    early_dataset, early_loader = _build_loader(
        data_path,
        config,
        plan["early_stop_indices"],
        args.batch_size,
        args.workers,
        False,
        seed,
    )
    calibration_dataset, calibration_loader = _build_loader(
        data_path,
        config,
        plan["calibration_indices"],
        args.batch_size,
        args.workers,
        False,
        seed,
    )
    threshold_dataset, threshold_loader = _build_loader(
        data_path,
        config,
        plan["threshold_indices"],
        args.batch_size,
        args.workers,
        False,
        seed,
    )
    normalizer = _parameter_normalizer(
        data_path,
        plan["train_indices"],
        class_names,
        plan["class_mode"],
    )
    positive_counts = train_dataset.labels6[plan["train_indices"]][
        :, train_dataset.selected_indices
    ].sum(axis=0)
    classification_loss = build_classification_loss(
        args.loss,
        positive_counts=positive_counts,
        sample_count=len(plan["train_indices"]),
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.negative_clip,
        beta=args.effective_beta,
    )
    backbone = build_backbone(
        args.backbone,
        num_classes=plan["num_classes"],
        width=args.width,
        dropout=args.dropout,
    )
    model = SupervisedJointParameterModel(
        backbone,
        relation_dim=args.relation_dim,
        parameters_per_label=2,
        enable_parameter_heads=variant["parameter_heads"],
        relation_dropout=0.0,
        use_mft_count_classifier=variant["mft_count_classifier"],
        use_vft_spectral_head=variant["vft_spectral_head"],
        local_parameter_targets=variant.get("local_parameter_targets", []),
    )
    # Reset stochastic training state after optional branch initialization so C0/C1
    # share the exact classifier dropout stream when the parameter branch is detached.
    torch.manual_seed(int(seed) + 909)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 909)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    parameter_target_names, parameter_transfer_mask = _parameter_transfer_configuration(
        class_names, variant.get("blocked_parameter_targets", [])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_supervised_joint(
        model=model,
        train_loader=train_loader,
        validation_loader=early_loader,
        classification_loss=classification_loss,
        parameter_normalizer=normalizer,
        class_names=class_names,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=output_dir / "best_model.pth",
        use_parameter_heads=variant["parameter_heads"],
        detach_parameter_backbone=variant["detach_parameter_backbone"],
        maximum_parameter_weight=args.parameter_weight,
        warmup_fraction=variant["warmup_fraction"],
        ramp_end_fraction=variant["ramp_end_fraction"],
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
        gradient_strategy=variant.get("gradient_strategy", "standard"),
        parameter_transfer_mask=parameter_transfer_mask,
        parameter_target_names=parameter_target_names,
        collect_gradient_diagnostics=args.gradient_diagnostics,
    )

    early_output = predict_joint(
        model,
        early_loader,
        device,
        class_names,
        normalizer if variant["parameter_heads"] else None,
        use_amp=args.use_amp,
    )
    calibration_output = predict_joint(
        model,
        calibration_loader,
        device,
        class_names,
        normalizer if variant["parameter_heads"] else None,
        use_amp=args.use_amp,
    )
    threshold_output = predict_joint(
        model,
        threshold_loader,
        device,
        class_names,
        normalizer if variant["parameter_heads"] else None,
        use_amp=args.use_amp,
    )
    temperature = fit_temperature(
        calibration_output.logits,
        calibration_output.targets,
        source_role="calibration",
        max_iter=20,
    )
    threshold_probabilities = _sigmoid(
        apply_temperature(threshold_output.logits, temperature.temperature)
    )
    threshold_selection = select_per_label_f1_thresholds(
        threshold_output.targets,
        threshold_probabilities,
        source_role="threshold_selection",
    )
    early_probabilities, early_predictions, early_metrics = _prediction_metrics(
        early_output,
        temperature.temperature,
        threshold_selection.thresholds,
        class_names,
    )
    parameter_specs = {
        class_name: get_parameter_specs(class_name) for class_name in class_names
    }
    parameter_metrics = compute_parameter_metrics(
        early_output,
        class_names,
        parameter_specs,
        normalizer,
        class_predictions=early_predictions,
    )
    _write_history(output_dir / "history.csv", result.history)
    _write_json(output_dir / "classification_metrics.json", early_metrics)
    _write_json(output_dir / "parameter_metrics.json", parameter_metrics)
    _write_json(output_dir / "parameter_normalizer.json", _normalizer_payload(normalizer, class_names))
    _write_json(
        output_dir / "calibration_and_thresholds.json",
        {
            "temperature": temperature.temperature,
            "calibration_loss_before": temperature.loss_before,
            "calibration_loss_after": temperature.loss_after,
            "thresholds": threshold_selection.thresholds,
            "threshold_mode": threshold_selection.mode,
            "calibration_source_role": temperature.source_role,
            "threshold_source_role": threshold_selection.source_role,
        },
    )
    np.savez_compressed(
        str(output_dir / "predictions.npz"),
        logits=early_output.logits,
        probabilities=early_probabilities,
        predictions=early_predictions,
        targets=early_output.targets,
        sample_ids=early_output.sample_ids,
        parameter_predictions=early_output.parameter_predictions,
        parameter_targets=early_output.parameter_targets,
        parameter_masks=early_output.parameter_masks,
    )
    receipt = {
        "status": "completed",
        "variant": variant_name,
        "seed": int(seed),
        "config_sha256": config_hash,
        "split_sha256": plan["split_sha256"],
        "train_samples": plan["train_samples"],
        "early_stop_samples": plan["early_stop_samples"],
        "calibration_samples": plan["calibration_samples"],
        "threshold_samples": plan["threshold_samples"],
        "best_epoch": result.best_epoch,
        "epochs_ran": result.epochs_ran,
        "best_macro_aupr": result.best_macro_aupr,
        "best_hamming_loss": result.best_hamming_loss,
        "calibrated_macro_aupr": early_metrics["macro_aupr"],
        "calibrated_macro_f1": early_metrics["macro_f1"],
        "calibrated_hamming_loss": early_metrics["hamming_loss"],
        "oracle_macro_target_nmae": (
            None
            if parameter_metrics is None
            else parameter_metrics["oracle_macro_target_nmae"]
        ),
        "system_parameter_coverage": (
            None
            if parameter_metrics is None
            else parameter_metrics["system_parameter_coverage"]
        ),
        "mft_count_rounded_mae": None,
        "mft_count_rounded_accuracy": None,
        "vft_doppler_mae": None,
        "train_seconds": result.train_seconds,
        "training_parameters": count_trainable_parameters(model),
        "gradient_strategy": variant.get("gradient_strategy", "standard"),
        "blocked_parameter_targets": variant.get("blocked_parameter_targets", []),
        "local_parameter_targets": variant.get("local_parameter_targets", []),
        "gradient_diagnostics": _gradient_summary(result.history),
        "device": str(device),
        "locked_test_used": False,
        "output_dir": str(output_dir),
    }
    if parameter_metrics is not None:
        target_lookup = {
            (row["class"], row["parameter"]): row
            for row in parameter_metrics["per_target"]
        }
        mft_count = target_lookup.get(("MFT", "num_targets"))
        vft_doppler = target_lookup.get(("VFT", "Doppler"))
        if mft_count is not None:
            receipt["mft_count_rounded_mae"] = mft_count.get("rounded_mae")
            receipt["mft_count_rounded_accuracy"] = mft_count.get(
                "rounded_accuracy"
            )
        if vft_doppler is not None:
            receipt["vft_doppler_mae"] = vft_doppler["mae"]
    _write_json(output_dir / "run_receipt.json", receipt)
    for dataset in (
        train_dataset,
        early_dataset,
        calibration_dataset,
        threshold_dataset,
    ):
        dataset.close()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def _aggregate(receipts):
    rows = []
    for variant_name in VARIANTS:
        selected = [row for row in receipts if row.get("variant") == variant_name and row.get("status") == "completed"]
        if not selected:
            continue
        entry = {"variant": variant_name, "seed_count": len(selected)}
        for key in (
            "calibrated_macro_aupr",
            "calibrated_macro_f1",
            "calibrated_hamming_loss",
            "oracle_macro_target_nmae",
            "system_parameter_coverage",
            "mft_count_rounded_mae",
            "mft_count_rounded_accuracy",
            "vft_doppler_mae",
            "train_seconds",
        ):
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            entry[key] = (
                {"mean": float(np.mean(values)), "std": float(np.std(values))}
                if values
                else None
            )
        rows.append(entry)
    by_variant_seed = {
        (row["variant"], int(row["seed"])): row
        for row in receipts
        if row.get("status") == "completed"
    }
    gates = []
    gate_candidates = (
        ("C2", "C0"),
        ("C3", "C0"),
        ("C2_SD", "C2"),
        ("C2_GP", "C2"),
        ("C2_IGP", "C2"),
    )
    for candidate, baseline_name in gate_candidates:
        differences = []
        directions = []
        parameter_differences = []
        for seed in sorted(
            {key[1] for key in by_variant_seed if key[0] == baseline_name}
        ):
            baseline = by_variant_seed.get((baseline_name, seed))
            selected = by_variant_seed.get((candidate, seed))
            if baseline is None or selected is None:
                continue
            difference = float(selected["calibrated_macro_aupr"]) - float(
                baseline["calibrated_macro_aupr"]
            )
            differences.append(difference)
            directions.append(difference >= 0.0)
            if (
                baseline.get("oracle_macro_target_nmae") is not None
                and selected.get("oracle_macro_target_nmae") is not None
            ):
                parameter_differences.append(
                    float(baseline["oracle_macro_target_nmae"])
                    - float(selected["oracle_macro_target_nmae"])
                )
        gates.append(
            {
                "variant": candidate,
                "baseline": baseline_name,
                "macro_aupr_differences": differences,
                "mean_macro_aupr_difference": (
                    float(np.mean(differences)) if differences else None
                ),
                "parameter_nmae_reductions": parameter_differences,
                "mean_parameter_nmae_reduction": (
                    float(np.mean(parameter_differences))
                    if parameter_differences
                    else None
                ),
                "nonnegative_seed_count": int(sum(directions)),
                "classification_noninferiority_passed": bool(
                    differences
                    and float(np.mean(differences)) >= -0.005
                    and sum(directions) >= 2
                ),
            }
        )
    return {"aggregate": rows, "gates": gates, "runs": receipts}


def run_experiment(args):
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    receipts = []
    for seed in args.seeds:
        for variant_name in variants:
            receipts.append(run_one(args, variant_name, int(seed)))
    if args.dry_run:
        return receipts
    summary = _aggregate(receipts)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else PROJECT_ROOT / "outputs" / "student_b" / "supervised_joint_parameter"
    )
    summary.update(
        {
            "status": "completed",
            "locked_test_used": False,
            "protocol": {
                "variants": variants,
                "seeds": [int(value) for value in args.seeds],
                "config": str(args.config),
                "epochs": int(args.epochs),
            "parameter_weight": float(args.parameter_weight),
                "gradient_diagnostics": bool(args.gradient_diagnostics),
                "max_train_samples": int(args.max_train_samples),
                "max_eval_samples": int(args.max_eval_samples),
            },
        }
    )
    _write_json(output_root / "supervised_joint_parameter_summary.json", summary)
    print(json.dumps({"aggregate": summary["aggregate"], "gates": summary["gates"]}, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_b" / "supervised_joint_six_class.yaml",
    )
    parser.add_argument("--variant", choices=["all"] + list(VARIANTS), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--loss", choices=["bce", "focal", "asl", "cb_bce", "cb_asl"], default="bce")
    parser.add_argument("--backbone", default="complex_1d_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--gradient-diagnostics",
        action="store_true",
        help="record one 12-target gradient audit batch per epoch",
    )
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--parameter-weight", type=float, default=0.1)
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=3.0)
    parser.add_argument("--negative-clip", type=float, default=0.05)
    parser.add_argument("--effective-beta", type=float, default=0.99)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
