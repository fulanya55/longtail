"""TAPC-CJNet T0-T5 runner with multi-axis manifests and audit artifacts."""

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

from cjlt.data.longtail_manifest import (  # noqa: E402
    build_combo_longtail,
    build_label_longtail,
    build_mixed_longtail,
    build_power_longtail,
    compute_weak_component_mask,
    profile_multi_axis_imbalance,
)
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.power_pair_dataset import PowerPairDataset  # noqa: E402
from cjlt.data.power_pairs import build_role_bound_power_pair_index  # noqa: E402
from cjlt.data.v73_dataset import (  # noqa: E402
    V73SignalDataset,
    read_labels,
    read_parameters,
)
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.experiment_record import ExperimentRecord, write_json  # noqa: E402
from cjlt.engine.locked_test import sha256_file  # noqa: E402
from cjlt.engine.run_manager import (  # noqa: E402
    configuration_sha256,
    decide_run,
    stable_run_arguments,
)
from cjlt.engine.student_a_trainer import train_student_a  # noqa: E402
from cjlt.engine.thresholding import (  # noqa: E402
    apply_thresholds,
    select_global_threshold,
    select_per_label_fbeta_thresholds,
    select_per_label_f1_thresholds,
    select_per_label_youden_thresholds,
    select_shrinkage_per_label_thresholds,
)
from cjlt.engine.trainer import predict_loader  # noqa: E402
from cjlt.losses.classification import (  # noqa: E402
    build_classification_loss,
    effective_number_weights,
)
from cjlt.losses.parameter import (  # noqa: E402
    ParameterNormalizer,
    select_parameter_targets,
)
from cjlt.models.backbone_adapter import build_backbone, count_trainable_parameters  # noqa: E402
from cjlt.models.student_a import (  # noqa: E402
    StudentAModel,
    configure_classifier_only_training,
    configure_last_block_classifier_training,
    configure_last_two_blocks_classifier_training,
    count_deployment_parameters,
)
from experiments.run_shared import (  # noqa: E402
    _data_path,
    _write_history,
    build_run_plan,
    load_config,
)


def tapc_variant_matrix():
    return {
        "T0": {"pair_views": False, "prototype": False, "distillation": False, "ordering": False, "stability": False, "parameter": False},
        "T0P": {"pair_views": True, "prototype": False, "distillation": False, "ordering": False, "stability": False, "parameter": False},
        "T1": {"pair_views": False, "prototype": True, "distillation": False, "ordering": False, "stability": False, "parameter": False},
        "T2": {"pair_views": True, "prototype": False, "distillation": True, "ordering": True, "stability": False, "parameter": False},
        "T2S": {"pair_views": True, "prototype": False, "distillation": True, "ordering": True, "stability": True, "parameter": False},
        "T2O_V2": {"pair_views": True, "prototype": False, "distillation": False, "ordering": True, "stability": False, "parameter": False},
        "T2D": {"pair_views": True, "prototype": False, "distillation": True, "ordering": False, "stability": False, "parameter": False},
        "T2OD_V2": {"pair_views": True, "prototype": False, "distillation": True, "ordering": True, "stability": False, "parameter": False},
        "T3": {"pair_views": True, "prototype": True, "distillation": True, "ordering": True, "stability": False, "parameter": False},
        "T4": {"pair_views": True, "prototype": True, "distillation": True, "ordering": True, "stability": True, "parameter": False},
        "T5": {"pair_views": True, "prototype": True, "distillation": True, "ordering": True, "stability": True, "parameter": True},
    }


def _component_power(values79, class_names):
    values = np.asarray(values79, dtype=np.float32)
    return np.stack(
        [values[:, get_parameter_specs(name)[0].column_index] for name in class_names],
        axis=1,
    )


def _target_label_counts(
    labels,
    indices,
    imbalance_factor,
    rank_order=None,
    tail_count_anchor=None,
):
    counts = labels[indices].sum(axis=0).astype(np.int64)
    if rank_order is None:
        order = np.argsort(-counts, kind="stable")
    else:
        order = np.asarray(rank_order, dtype=np.int64)
        if sorted(order.tolist()) != list(range(labels.shape[1])):
            raise ValueError(
                "label_rank_order must be a permutation of all label indices"
            )
    if tail_count_anchor is None:
        maximum = int(counts[order[0]])
    else:
        tail_count = int(tail_count_anchor)
        if tail_count <= 0:
            raise ValueError("label_tail_count_anchor must be positive")
        maximum = int(round(tail_count * float(imbalance_factor)))
    target = counts.copy()
    denominator = max(len(order) - 1, 1)
    for rank, label_index in enumerate(order):
        desired = int(
            round(
                maximum
                * float(imbalance_factor) ** (-float(rank) / float(denominator))
            )
        )
        target[label_index] = max(1, desired)
    if tail_count_anchor is None:
        target = np.minimum(target, counts)
    elif np.any(target > counts):
        bad = np.flatnonzero(target > counts).tolist()
        raise ValueError(
            "Controlled label target exceeds available positives for labels {}: "
            "target={}, available={}".format(
                bad, target.tolist(), counts.tolist()
            )
        )
    return target


def _build_manifest(args, labels, plan, component_power):
    indices = plan["train_indices"]
    common = {
        "labels": labels,
        "eligible_indices": indices,
        "seed": args.seed,
        "source_split_sha256": plan["split_sha256"],
    }
    if args.protocol == "balanced":
        profile = profile_multi_axis_imbalance(
            labels,
            indices,
            component_power,
            args.weak_threshold_db,
            class_names=plan["class_names"],
        )
        return None, indices.copy(), profile
    if args.protocol == "combo_lt":
        manifest = build_combo_longtail(
            **common, imbalance_factor=args.combo_imbalance_factor
        )
    elif args.protocol == "label_lt":
        manifest = build_label_longtail(
            **common,
            target_label_counts=_target_label_counts(
                labels,
                indices,
                args.label_imbalance_factor,
                rank_order=args.label_rank_order,
                tail_count_anchor=args.label_tail_count_anchor,
            ),
            tolerance=args.label_count_tolerance,
            target_sample_count=(
                None
                if int(args.label_target_samples) <= 0
                else int(args.label_target_samples)
            ),
            sample_count_tolerance=args.label_sample_count_tolerance,
        )
    elif args.protocol == "power_lt":
        manifest = build_power_longtail(
            **common,
            component_power_db=component_power,
            weak_threshold_db=args.weak_threshold_db,
        )
    elif args.protocol == "mixed_lt":
        manifest = build_mixed_longtail(
            **common,
            component_power_db=component_power,
            label_imbalance_factor=args.label_imbalance_factor,
            combo_imbalance_factor=args.combo_imbalance_factor,
            weak_threshold_db=args.weak_threshold_db,
        )
    else:
        raise ValueError("Unknown protocol: {}".format(args.protocol))
    profile = profile_multi_axis_imbalance(
        labels,
        manifest.selected_indices,
        component_power,
        args.weak_threshold_db,
        class_names=plan["class_names"],
    )
    return manifest, manifest.selected_indices, profile


def _named_combo_metadata(profile, class_names):
    def named(bits):
        return "+".join(
            class_names[index] for index, value in enumerate(str(bits)) if value == "1"
        )

    counts = {
        named(key): int(value)
        for key, value in profile["combination_counts"].items()
    }
    tiers = {
        named(key): str(value) for key, value in profile["combo_tiers"].items()
    }
    return counts, tiers


def _sigmoid(logits):
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _parameter_metrics(model, loader, device, normalizer, class_names, use_amp):
    predictions = []
    targets = []
    masks = []
    labels = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(batch["signal"].to(device), return_aux=True)
            target, mask = select_parameter_targets(
                batch["parameter_values"], batch["parameter_mask"], class_names
            )
            predictions.append(
                normalizer.inverse_transform(
                    output["parameter_predictions"].detach().cpu()
                ).numpy()
            )
            targets.append(target.numpy())
            masks.append(mask.numpy())
            labels.append(batch["label"].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    valid = np.logical_and(np.concatenate(masks), np.concatenate(labels)[:, :, None] > 0.5)
    rows = []
    for class_index, class_name in enumerate(class_names):
        for parameter_index, spec in enumerate(get_parameter_specs(class_name)):
            selected = valid[:, class_index, parameter_index]
            rows.append(
                {
                    "label": class_name,
                    "parameter": spec.name,
                    "unit": spec.unit,
                    "count": int(selected.sum()),
                    "mae": (
                        float(
                            np.abs(
                                prediction[selected, class_index, parameter_index]
                                - target[selected, class_index, parameter_index]
                            ).mean()
                        )
                        if selected.any()
                        else float("nan")
                    ),
                }
            )
    return {"per_parameter": rows}


def run_experiment(args):
    variants = tapc_variant_matrix()
    variant = variants[args.variant]
    config = load_config(args.config)
    plan = build_run_plan(
        config,
        loss_name=args.loss,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    data_path = _data_path(config)
    labels6 = read_labels(data_path)
    labels = select_labels(labels6, plan["class_mode"])
    values79, mask79 = read_parameters(data_path)
    component_power = _component_power(values79, plan["class_names"])
    manifest, train_indices, profile = _build_manifest(
        args, labels, plan, component_power
    )
    plan = dict(plan)
    plan["train_indices"] = train_indices
    plan["train_samples"] = int(len(train_indices))
    plan["protocol"] = args.protocol
    plan["manifest_sha256"] = None if manifest is None else manifest.sha256
    printable_plan = {
        key: value for key, value in plan.items() if not key.endswith("_indices")
    }
    scientific_config = {
        "config": config,
        "arguments": stable_run_arguments(vars(args)),
        "plan": printable_plan,
        "variant_flags": variant,
        "multi_axis_profile": profile,
    }
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "outputs"
        / "tapc"
        / str(plan["experiment_name"])
        / args.protocol
        / args.variant
        / "seed_{}".format(args.seed)
    )
    config_hash = configuration_sha256(scientific_config)
    if args.dry_run:
        print(json.dumps({"config_sha256": config_hash, **printable_plan, **variant}, ensure_ascii=False, indent=2))
        return printable_plan
    decision = decide_run(output_dir, config_hash, resume=args.resume)
    if decision.action == "skip":
        return decision.receipt
    record = ExperimentRecord(output_dir, scientific_config)
    if record.config_sha256 != config_hash:
        raise RuntimeError("experiment record config hash mismatch")

    datasets = []
    try:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        record.event("data_manifest_ready", {"profile": profile})
        write_json(output_dir / "multi_axis_profile.json", profile)
        if manifest is not None:
            write_json(output_dir / "longtail_manifest.json", manifest.to_dict())

        common = {
            "path": data_path,
            "class_mode": plan["class_mode"],
            "normalization": config["normalization"],
        }
        early_dataset = V73SignalDataset(
            indices=plan["early_stop_indices"],
            include_parameters=variant["parameter"],
            **common,
        )
        calibration_dataset = V73SignalDataset(
            indices=plan["calibration_indices"], **common
        )
        threshold_dataset = V73SignalDataset(
            indices=plan["threshold_indices"], **common
        )
        datasets.extend([early_dataset, calibration_dataset, threshold_dataset])
        pair_index = None
        if variant["pair_views"]:
            power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
            pair_index = build_role_bound_power_pair_index(
                data_path,
                power_root,
                base_indices=train_indices,
                levels=args.power_levels,
                reference_level=args.reference_level,
            )
            train_dataset = PowerPairDataset(
                power_root,
                pair_index.records,
                class_mode=plan["class_mode"],
                normalization=config["normalization"],
                include_parameters=variant["parameter"],
            )
        else:
            train_dataset = V73SignalDataset(
                indices=train_indices,
                include_parameters=variant["parameter"],
                **common,
            )
        datasets.append(train_dataset)
        loader_args = {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.workers),
            "pin_memory": torch.cuda.is_available(),
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
        early_loader = DataLoader(early_dataset, shuffle=False, **loader_args)
        calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_args)
        threshold_loader = DataLoader(threshold_dataset, shuffle=False, **loader_args)

        positive_counts = labels[train_indices].sum(axis=0)
        classification_loss = build_classification_loss(
            args.loss,
            positive_counts=positive_counts,
            sample_count=len(train_indices),
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            clip=args.negative_clip,
            beta=args.effective_beta,
            db_mix_weight=args.db_mix_weight,
            tail_label_index=args.tail_label_index,
            tail_loss_weight=args.tail_loss_weight,
        )
        relation_weights = effective_number_weights(
            positive_counts, beta=args.effective_beta
        )
        parameter_normalizer = None
        if variant["parameter"]:
            selected_target, selected_mask = select_parameter_targets(
                values79[train_indices], mask79[train_indices], plan["class_names"]
            )
            selected_mask = torch.logical_and(
                selected_mask,
                torch.as_tensor(labels[train_indices]).bool()[:, :, None],
            )
            parameter_normalizer = ParameterNormalizer().fit(
                selected_target, selected_mask
            )
            write_json(
                output_dir / "parameter_normalizer.json",
                {
                    "mean": parameter_normalizer.mean,
                    "scale": parameter_normalizer.scale,
                    "fit_role": "optimization",
                },
            )
        backbone = build_backbone(
            args.backbone,
            num_classes=plan["num_classes"],
            width=args.width,
            dropout=args.dropout,
        )
        model = StudentAModel(
            backbone,
            relation_dim=args.relation_dim,
            relation_dropout=args.relation_dropout,
            conditioning_mode=args.conditioning_mode,
            enable_parameter_heads=variant["parameter"],
        )
        initial_checkpoint_sha256 = None
        if args.initial_checkpoint is not None:
            initial_path = Path(args.initial_checkpoint)
            if not initial_path.exists():
                raise FileNotFoundError(
                    "Initial checkpoint does not exist: {}".format(initial_path)
                )
            checkpoint = torch.load(
                str(initial_path), map_location="cpu", weights_only=False
            )
            if "model_state_dict" not in checkpoint:
                raise ValueError("Initial checkpoint is missing model_state_dict")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            initial_checkpoint_sha256 = sha256_file(initial_path)
            record.event(
                "initial_checkpoint_loaded",
                {
                    "path": str(initial_path),
                    "sha256": initial_checkpoint_sha256,
                },
            )
        if args.trainable_scope == "classifier":
            if args.initial_checkpoint is None:
                raise ValueError(
                    "classifier-only tuning requires --initial-checkpoint"
                )
            trainable_classifier_parameters = configure_classifier_only_training(model)
            record.event(
                "trainable_scope_configured",
                {
                    "scope": "classifier",
                    "trainable_parameters": trainable_classifier_parameters,
                    "frozen_representation": True,
                },
            )
        elif args.trainable_scope == "last_block_classifier":
            if args.initial_checkpoint is None:
                raise ValueError(
                    "last-block adaptation requires --initial-checkpoint"
                )
            trainable_groups = configure_last_block_classifier_training(model)
            record.event(
                "trainable_scope_configured",
                {
                    "scope": "last_block_classifier",
                    "trainable_parameters": trainable_groups["total"],
                    "final_temporal_block_parameters": trainable_groups[
                        "final_temporal_block"
                    ],
                    "classifier_parameters": trainable_groups["classifier"],
                    "representation_learning_rate_scale": (
                        args.representation_learning_rate_scale
                    ),
                    "starting_point_weight": args.starting_point_weight,
                    "frozen_representation": False,
                    "partially_adapted_representation": True,
                },
            )
        elif args.trainable_scope == "last_two_blocks_classifier":
            if args.initial_checkpoint is None:
                raise ValueError(
                    "last-two-block adaptation requires --initial-checkpoint"
                )
            trainable_groups = configure_last_two_blocks_classifier_training(model)
            record.event(
                "trainable_scope_configured",
                {
                    "scope": "last_two_blocks_classifier",
                    "trainable_parameters": trainable_groups["total"],
                    "penultimate_temporal_block_parameters": trainable_groups[
                        "penultimate_temporal_block"
                    ],
                    "final_temporal_block_parameters": trainable_groups[
                        "final_temporal_block"
                    ],
                    "classifier_parameters": trainable_groups["classifier"],
                    "penultimate_representation_learning_rate_scale": (
                        args.penultimate_representation_learning_rate_scale
                    ),
                    "representation_learning_rate_scale": (
                        args.representation_learning_rate_scale
                    ),
                    "frozen_representation": False,
                    "partially_adapted_representation": True,
                },
            )
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        record.event("training_started", {"device": str(device)})
        result = train_student_a(
            model=model,
            train_loader=train_loader,
            validation_loader=early_loader,
            classification_loss=classification_loss,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            checkpoint_path=output_dir / "best_model.pth",
            use_prototype=variant["prototype"],
            use_power_consistency=False,
            use_pair_views=variant["pair_views"],
            use_power_distillation=variant["distillation"],
            use_directional_ordering=variant["ordering"],
            use_stability=variant["stability"],
            use_parameter_heads=variant["parameter"],
            prototype_weight=args.prototype_weight,
            power_weight=args.power_weight,
            stability_weight=args.stability_weight,
            parameter_weight=args.parameter_weight,
            ordering_margin=args.ordering_margin,
            ordering_mode=args.ordering_mode,
            ordering_gradient_mode=args.ordering_gradient_mode,
            ordering_start_fraction=args.ordering_start_fraction,
            stability_tolerance=args.stability_tolerance,
            prototype_temperature=args.prototype_temperature,
            prototype_momentum=args.prototype_momentum,
            label_weights=relation_weights,
            warmup_fraction=args.warmup_fraction,
            ramp_end_fraction=args.ramp_end_fraction,
            use_amp=args.use_amp,
            gradient_clip=args.gradient_clip,
            class_names=plan["class_names"],
            parameter_normalizer=parameter_normalizer,
            parameter_stop_gradient=args.parameter_stop_gradient,
            selection_metric=args.selection_metric,
            freeze_representation=args.trainable_scope == "classifier",
            trainable_scope=args.trainable_scope,
            representation_learning_rate_scale=(
                args.representation_learning_rate_scale
            ),
            starting_point_weight=args.starting_point_weight,
            selection_tail_index=args.selection_tail_index,
            penultimate_representation_learning_rate_scale=(
                args.penultimate_representation_learning_rate_scale
            ),
        )
        _write_history(output_dir / "history.csv", result.history)
        best_epoch_diagnostics = next(
            row for row in result.history if int(row["epoch"]) == result.best_epoch
        )
        write_json(
            output_dir / "best_epoch_diagnostics.json", best_epoch_diagnostics
        )
        write_json(output_dir / "prototype_diagnostics.json", result.prototype_diagnostics)

        early_output = predict_loader(model, early_loader, device, use_amp=args.use_amp)
        calibration_output = predict_loader(model, calibration_loader, device, use_amp=args.use_amp)
        threshold_output = predict_loader(model, threshold_loader, device, use_amp=args.use_amp)
        temperature = fit_temperature(
            calibration_output.logits,
            calibration_output.targets,
            source_role="calibration",
        )
        threshold_probability = _sigmoid(
            apply_temperature(threshold_output.logits, temperature.temperature)
        )
        global_thresholds = select_global_threshold(
            threshold_output.targets,
            threshold_probability,
            source_role="threshold_selection",
        )
        per_label_f1_thresholds = select_per_label_f1_thresholds(
            threshold_output.targets,
            threshold_probability,
            source_role="threshold_selection",
        )
        per_label_f2_thresholds = select_per_label_fbeta_thresholds(
            threshold_output.targets,
            threshold_probability,
            beta=2.0,
            source_role="threshold_selection",
        )
        thresholds = select_shrinkage_per_label_thresholds(
            threshold_output.targets,
            threshold_probability,
            positive_counts=positive_counts,
            beta=100.0,
            source_role="threshold_selection",
        )
        per_label_youden_thresholds = select_per_label_youden_thresholds(
            threshold_output.targets,
            threshold_probability,
            source_role="threshold_selection",
        )
        early_probability = _sigmoid(
            apply_temperature(early_output.logits, temperature.temperature)
        )
        threshold_modes = {
            "fixed_0p5": [0.5] * early_probability.shape[1],
            "global": global_thresholds.thresholds,
            "per_label_f1": per_label_f1_thresholds.thresholds,
            "per_label_f2": per_label_f2_thresholds.thresholds,
            "shrinkage_per_label_f1": thresholds.thresholds,
            "per_label_youden": per_label_youden_thresholds.thresholds,
        }
        predictions_by_mode = {
            name: apply_thresholds(early_probability, values)
            for name, values in threshold_modes.items()
        }
        early_prediction = predictions_by_mode["shrinkage_per_label_f1"]
        fixed_prediction = predictions_by_mode["fixed_0p5"]
        train_combo_counts, combo_tiers = _named_combo_metadata(
            profile, plan["class_names"]
        )
        tail_indices = [
            index
            for index, name in enumerate(plan["class_names"])
            if profile["label_tiers"].get(name) == "tail"
        ]
        weak_mask = compute_weak_component_mask(
            labels,
            component_power,
            args.weak_threshold_db,
        )[plan["early_stop_indices"]]
        metrics_by_mode = {}
        for threshold_mode, prediction in predictions_by_mode.items():
            mode_metrics = compute_multilabel_metrics(
                early_output.targets,
                prediction,
                probabilities=early_probability,
                class_names=plan["class_names"],
                tail_indices=tail_indices,
                weak_component_mask=weak_mask,
                train_combo_counts=train_combo_counts,
                combo_tiers=combo_tiers,
            )
            mode_metrics["evaluation_role"] = "early_stop_development_diagnostic"
            mode_metrics["threshold_mode"] = threshold_mode
            mode_metrics["locked_test_used"] = False
            metrics_by_mode[threshold_mode] = mode_metrics
        metrics = metrics_by_mode["shrinkage_per_label_f1"]
        write_json(output_dir / "metrics.json", metrics)
        fixed_metrics = metrics_by_mode["fixed_0p5"]
        write_json(output_dir / "metrics_fixed_0p5.json", fixed_metrics)
        write_json(output_dir / "metrics_threshold_modes.json", metrics_by_mode)
        write_json(
            output_dir / "calibration.json",
            {
                "temperature": temperature.temperature,
                "loss_before": temperature.loss_before,
                "loss_after": temperature.loss_after,
                "source_role": "calibration",
            },
        )
        write_json(
            output_dir / "thresholds.json",
            {
                "per_label": thresholds.thresholds,
                "subset_acc_on_selection_role": thresholds.subset_acc,
                "source_role": "threshold_selection",
                "primary_mode": "shrinkage_per_label_f1",
                "modes": {
                    "fixed_0p5": threshold_modes["fixed_0p5"],
                    "global": global_thresholds.thresholds,
                    "per_label_f1": per_label_f1_thresholds.thresholds,
                    "per_label_f2": per_label_f2_thresholds.thresholds,
                    "shrinkage_per_label_f1": thresholds.thresholds,
                    "per_label_youden": per_label_youden_thresholds.thresholds,
                },
                "selection_subset_acc": {
                    "global": global_thresholds.subset_acc,
                    "per_label_f1": per_label_f1_thresholds.subset_acc,
                    "per_label_f2": per_label_f2_thresholds.subset_acc,
                    "shrinkage_per_label_f1": thresholds.subset_acc,
                    "per_label_youden": per_label_youden_thresholds.subset_acc,
                },
            },
        )
        np.savez_compressed(
            str(output_dir / "predictions.npz"),
            logits=early_output.logits,
            probabilities=early_probability,
            predictions=early_prediction,
            targets=early_output.targets,
            sample_ids=early_output.sample_ids,
        )
        parameter_metrics = None
        if variant["parameter"]:
            parameter_metrics = _parameter_metrics(
                model,
                early_loader,
                device,
                parameter_normalizer,
                plan["class_names"],
                args.use_amp,
            )
            parameter_metrics["evaluation_role"] = "early_stop_development_diagnostic"
            write_json(output_dir / "parameter_metrics.json", parameter_metrics)
        receipt = record.complete(
            {
                "variant": args.variant,
                "pair_views": variant["pair_views"],
                "power_distillation": variant["distillation"],
                "directional_ordering": variant["ordering"],
                "protocol": args.protocol,
                "best_epoch": result.best_epoch,
                "selection_metric": result.selection_metric,
                "best_selection_value": result.best_selection_value,
                "epochs_ran": result.epochs_ran,
                "best_subset_acc": result.best_subset_acc,
                "best_hamming_loss": result.best_hamming_loss,
                "macro_f1": metrics["macro_f1"],
                "macro_f1_fixed_0p5": fixed_metrics["macro_f1"],
                "macro_aupr": metrics["macro_aupr"],
                "weak_component_recall": metrics["weak_component_recall"],
                "weak_component_count": metrics["weak_component_count"],
                "directional_margin_mean": best_epoch_diagnostics[
                    "directional_margin_mean"
                ],
                "directional_satisfaction_rate": best_epoch_diagnostics[
                    "directional_satisfaction_rate"
                ],
                "unchanged_drift_p90": best_epoch_diagnostics[
                    "unchanged_drift_p90"
                ],
                "train_seconds": result.train_seconds,
                "training_parameters": count_trainable_parameters(model),
                "deployment_parameters": count_deployment_parameters(model),
                "device": str(device),
                "split_sha256": plan["split_sha256"],
                "manifest_sha256": plan["manifest_sha256"],
                "trainable_scope": args.trainable_scope,
                "initial_checkpoint": (
                    None
                    if args.initial_checkpoint is None
                    else str(Path(args.initial_checkpoint))
                ),
                "initial_checkpoint_sha256": initial_checkpoint_sha256,
                "frozen_representation": args.trainable_scope == "classifier",
                "partially_adapted_representation": (
                    args.trainable_scope == "last_block_classifier"
                ),
                "representation_learning_rate_scale": (
                    args.representation_learning_rate_scale
                ),
                "penultimate_representation_learning_rate_scale": (
                    args.penultimate_representation_learning_rate_scale
                ),
                "starting_point_weight": args.starting_point_weight,
                "tail_label_index": args.tail_label_index,
                "tail_loss_weight": args.tail_loss_weight,
                "selection_tail_index": args.selection_tail_index,
                "pair_sha256": None if pair_index is None else pair_index.sha256,
                "locked_test_used": False,
            }
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return receipt
    except Exception as error:
        record.fail(error)
        raise
    finally:
        for dataset in datasets:
            if hasattr(dataset, "close"):
                dataset.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "tapc_four_class.yaml",
    )
    parser.add_argument("--variant", choices=list(tapc_variant_matrix()), default="T4")
    parser.add_argument(
        "--protocol",
        choices=["balanced", "label_lt", "combo_lt", "power_lt", "mixed_lt"],
        default="mixed_lt",
    )
    parser.add_argument(
        "--loss",
        choices=[
            "bce", "focal", "asl", "cb_bce", "cb_asl", "db",
            "asl_db_mix", "tail_asl_db",
        ],
        default="cb_asl",
    )
    parser.add_argument("--backbone", default="complex_1d_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--relation-dropout", type=float, default=0.1)
    parser.add_argument(
        "--conditioning-mode",
        choices=["independent", "gated"],
        default="independent",
    )
    parser.add_argument("--prototype-weight", type=float, default=0.20)
    parser.add_argument("--power-weight", type=float, default=0.20)
    parser.add_argument("--stability-weight", type=float, default=0.05)
    parser.add_argument("--parameter-weight", type=float, default=0.10)
    parser.add_argument("--ordering-margin", type=float, default=0.20)
    parser.add_argument(
        "--ordering-mode",
        choices=["bidirectional", "strong_only"],
        default="bidirectional",
    )
    parser.add_argument(
        "--ordering-gradient-mode",
        choices=["shared", "backbone_stop_gradient"],
        default="shared",
    )
    parser.add_argument("--ordering-start-fraction", type=float, default=0.0)
    parser.add_argument("--stability-tolerance", type=float, default=0.25)
    parser.add_argument("--prototype-temperature", type=float, default=0.10)
    parser.add_argument("--prototype-momentum", type=float, default=0.99)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--ramp-end-fraction", type=float, default=0.30)
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=3.0)
    parser.add_argument("--negative-clip", type=float, default=0.05)
    parser.add_argument("--effective-beta", type=float, default=0.99)
    parser.add_argument(
        "--db-mix-weight",
        type=float,
        default=0.5,
        help="DB coefficient for asl_db_mix; ASL coefficient is one minus this value.",
    )
    parser.add_argument(
        "--tail-label-index",
        type=int,
        default=None,
        help="Registered tail output for tail_asl_db.",
    )
    parser.add_argument(
        "--tail-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier on the DB loss column used by tail_asl_db.",
    )
    parser.add_argument("--label-imbalance-factor", type=float, default=10.0)
    parser.add_argument("--combo-imbalance-factor", type=float, default=10.0)
    parser.add_argument("--label-count-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--label-tail-count-anchor",
        type=int,
        default=None,
        help="Optional fixed positive count for the tail label in controlled label_lt.",
    )
    parser.add_argument(
        "--label-target-samples",
        type=int,
        default=0,
        help="Optional fixed total training sample count for controlled label_lt.",
    )
    parser.add_argument(
        "--label-sample-count-tolerance",
        type=float,
        default=0.0,
        help="Relative tolerance for --label-target-samples; zero enforces an exact total.",
    )
    parser.add_argument(
        "--label-rank-order",
        type=int,
        nargs="+",
        default=None,
        help="Fixed head-to-tail label index order for label_lt.",
    )
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--power-levels", type=int, nargs="+", default=[10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--reference-level", type=int, default=30)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--trainable-scope",
        choices=[
            "all", "classifier", "last_block_classifier",
            "last_two_blocks_classifier",
        ],
        default="all",
        help="Train all parameters, only the head, or the last temporal block plus head.",
    )
    parser.add_argument(
        "--representation-learning-rate-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the final temporal block.",
    )
    parser.add_argument(
        "--penultimate-representation-learning-rate-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the penultimate temporal block.",
    )
    parser.add_argument(
        "--starting-point-weight",
        type=float,
        default=0.0,
        help="L2-SP weight on the adapted final temporal block.",
    )
    parser.add_argument("--parameter-stop-gradient", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["subset_acc", "macro_f1", "macro_aupr", "tail_aupr"],
        default="subset_acc",
    )
    parser.add_argument(
        "--selection-tail-index",
        type=int,
        default=None,
        help="Tail output used for per-epoch logging and tail_aupr selection.",
    )
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
