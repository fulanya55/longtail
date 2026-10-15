"""Trainer for tail prototypes and identity-matched power consistency."""

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from cjlt.engine.evaluator import compute_multilabel_metrics
from cjlt.losses.paired_power import (
    directional_power_ordering_loss,
    label_paired_power_consistency_loss,
    paired_power_diagnostic_values,
    power_curriculum_max_gap,
    unchanged_component_stability_loss,
)
from cjlt.losses.parameter import (
    masked_parameter_smooth_l1,
    select_parameter_targets,
)
from cjlt.losses.prototype import label_prototype_contrastive_loss
from cjlt.models.prototype_memory import TailPrototypeMemory, prototype_diagnostics
from cjlt.models.student_a import (
    keep_frozen_representation_in_eval_mode,
    keep_last_block_adaptation_modes,
    keep_last_two_block_adaptation_modes,
)


@dataclass
class StudentATrainingResult:
    best_epoch: int
    epochs_ran: int
    best_subset_acc: float
    best_hamming_loss: float
    train_seconds: float
    history: List[Dict[str, float]]
    prototype_diagnostics: Dict[str, object]
    selection_metric: str
    best_selection_value: float


def _auxiliary_scale(progress, warmup_fraction, ramp_end_fraction):
    warmup = float(warmup_fraction)
    ramp_end = float(ramp_end_fraction)
    if warmup < 0.0 or ramp_end < warmup or ramp_end > 1.0:
        raise ValueError("expected 0 <= warmup_fraction <= ramp_end_fraction <= 1")
    if progress < warmup:
        return 0.0
    if ramp_end <= warmup:
        return 1.0
    return min(1.0, max(0.0, (float(progress) - warmup) / (ramp_end - warmup)))


def _predict(model, loader, device, use_amp=False):
    model.eval()
    logits_parts = []
    target_parts = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            signal = batch.get("signal", batch.get("reference_signal"))
            if signal is None:
                raise KeyError("validation batch needs signal or reference_signal")
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(signal.to(device, non_blocking=True))
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32))
            target_parts.append(batch["label"].numpy().astype(np.uint8))
    logits = np.concatenate(logits_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(logits.astype(np.float64), -40.0, 40.0))
    )
    prediction = (probability >= 0.5).astype(np.uint8)
    return compute_multilabel_metrics(
        targets, prediction, probabilities=probability
    )


def _is_better(metrics, best_primary, best_hamming, selection_metric):
    metric = str(selection_metric)
    if metric not in ("subset_acc", "macro_f1", "macro_aupr", "tail_aupr"):
        raise ValueError(
            "selection_metric must be subset_acc, macro_f1, macro_aupr, or tail_aupr"
        )
    primary = float(metrics[metric])
    hamming = float(metrics["hamming_loss"])
    return bool(
        primary > best_primary
        or (np.isclose(primary, best_primary) and hamming < best_hamming)
    )


def _parameter_batch(batch, prefix, class_names, normalizer, device):
    values_key = "{}parameter_values".format(prefix)
    mask_key = "{}parameter_mask".format(prefix)
    if values_key not in batch or mask_key not in batch:
        raise KeyError("parameter-enabled training requires {} and {}".format(values_key, mask_key))
    target, mask = select_parameter_targets(
        batch[values_key], batch[mask_key], class_names
    )
    target = target.to(device)
    mask = mask.to(device)
    return normalizer.transform(target), mask


def train_student_a(
    model,
    train_loader,
    validation_loader,
    classification_loss,
    device,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    checkpoint_path: Optional[Path] = None,
    use_prototype=True,
    use_power_consistency=True,
    use_pair_views=None,
    use_power_distillation=None,
    use_directional_ordering=None,
    use_stability=False,
    use_parameter_heads=False,
    prototype_weight=0.2,
    power_weight=0.2,
    stability_weight=0.05,
    parameter_weight=0.10,
    ordering_margin=0.20,
    ordering_mode="bidirectional",
    ordering_gradient_mode="shared",
    ordering_start_fraction=0.0,
    stability_tolerance=0.25,
    prototype_temperature=0.1,
    prototype_momentum=0.99,
    label_weights=None,
    warmup_fraction=0.1,
    ramp_end_fraction=0.3,
    use_amp=False,
    gradient_clip=5.0,
    class_names=None,
    parameter_normalizer=None,
    parameter_stop_gradient=False,
    selection_metric="subset_acc",
    freeze_representation=False,
    trainable_scope="all",
    representation_learning_rate_scale=1.0,
    starting_point_weight=0.0,
    selection_tail_index=None,
    penultimate_representation_learning_rate_scale=1.0,
):
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1")
    selection_metric_value = str(selection_metric)
    if selection_metric_value not in (
        "subset_acc", "macro_f1", "macro_aupr", "tail_aupr"
    ):
        raise ValueError(
            "selection_metric must be subset_acc, macro_f1, macro_aupr, or tail_aupr"
        )
    tail_index = None if selection_tail_index is None else int(selection_tail_index)
    if tail_index is not None and (tail_index < 0 or tail_index >= model.num_classes):
        raise ValueError("selection_tail_index must identify one model output")
    if selection_metric_value == "tail_aupr" and tail_index is None:
        raise ValueError("tail_aupr selection requires selection_tail_index")
    pair_views_enabled = (
        bool(use_power_consistency)
        if use_pair_views is None
        else bool(use_pair_views)
    )
    distillation_enabled = (
        bool(use_power_consistency)
        if use_power_distillation is None
        else bool(use_power_distillation)
    )
    ordering_enabled = (
        bool(use_power_consistency)
        if use_directional_ordering is None
        else bool(use_directional_ordering)
    )
    if (distillation_enabled or ordering_enabled or bool(use_stability)) and not pair_views_enabled:
        raise ValueError("power auxiliary losses require use_pair_views=True")
    if str(ordering_gradient_mode) not in ("shared", "backbone_stop_gradient"):
        raise ValueError(
            "ordering_gradient_mode must be shared or backbone_stop_gradient"
        )
    if not 0.0 <= float(ordering_start_fraction) < 1.0:
        raise ValueError("ordering_start_fraction must be in [0, 1)")
    if bool(use_parameter_heads) and (class_names is None or parameter_normalizer is None):
        raise ValueError("parameter heads require class_names and a fitted parameter_normalizer")
    scope = str(trainable_scope)
    if bool(freeze_representation):
        scope = "classifier"
    if scope not in (
        "all", "classifier", "last_block_classifier", "last_two_blocks_classifier"
    ):
        raise ValueError(
            "trainable_scope must be all, classifier, last_block_classifier, "
            "or last_two_blocks_classifier"
        )
    representation_scale = float(representation_learning_rate_scale)
    if representation_scale <= 0.0 or representation_scale > 1.0:
        raise ValueError("representation_learning_rate_scale must be in (0, 1]")
    penultimate_scale = float(penultimate_representation_learning_rate_scale)
    if penultimate_scale <= 0.0 or penultimate_scale > 1.0:
        raise ValueError(
            "penultimate representation scale must be in (0, 1]"
        )
    if (
        scope == "last_two_blocks_classifier"
        and penultimate_scale > representation_scale
    ):
        raise ValueError(
            "penultimate representation scale must not exceed final block scale"
        )
    anchor_weight = float(starting_point_weight)
    if anchor_weight < 0.0:
        raise ValueError("starting_point_weight must be nonnegative")
    if anchor_weight > 0.0 and scope != "last_block_classifier":
        raise ValueError(
            "starting-point regularization requires last_block_classifier scope"
        )
    model.to(device)
    classification_loss.to(device)
    memory = TailPrototypeMemory(
        model.num_classes,
        model.relation_dim,
        momentum=float(prototype_momentum),
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model must expose at least one trainable parameter")
    if scope in ("last_block_classifier", "last_two_blocks_classifier"):
        temporal_blocks = [
            module
            for module in model.backbone.temporal_backbone
            if any(True for _ in module.parameters())
        ]
        final_block_parameters = [
            parameter
            for parameter in temporal_blocks[-1].parameters()
            if parameter.requires_grad
        ]
        final_block_ids = {id(parameter) for parameter in final_block_parameters}
        penultimate_block_parameters = []
        penultimate_block_ids = set()
        if scope == "last_two_blocks_classifier":
            penultimate_block_parameters = [
                parameter
                for parameter in temporal_blocks[-2].parameters()
                if parameter.requires_grad
            ]
            penultimate_block_ids = {
                id(parameter) for parameter in penultimate_block_parameters
            }
        classifier_parameters = [
            parameter
            for parameter in trainable_parameters
            if id(parameter) not in final_block_ids
            and id(parameter) not in penultimate_block_ids
        ]
        if not final_block_parameters or not classifier_parameters:
            raise ValueError(
                "last_block_classifier scope requires both representation and head parameters"
            )
        optimizer_parameters = [
            {
                "params": final_block_parameters,
                "lr": float(learning_rate) * representation_scale,
            },
            {"params": classifier_parameters, "lr": float(learning_rate)},
        ]
        if penultimate_block_parameters:
            optimizer_parameters.insert(
                0,
                {
                    "params": penultimate_block_parameters,
                    "lr": float(learning_rate) * penultimate_scale,
                },
            )
    else:
        final_block_ids = set()
        optimizer_parameters = trainable_parameters
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    named_parameters = dict(model.named_parameters())
    starting_point_anchors = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters.items()
        if id(parameter) in final_block_ids
    }
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state = None
    best_memory_state = None
    best_epoch = 0
    best_primary = -math.inf
    best_subset = -math.inf
    best_hamming = math.inf
    bad_epochs = 0
    history = []
    total_steps = max(1, int(epochs) * len(train_loader))
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        if scope == "classifier":
            keep_frozen_representation_in_eval_mode(model)
        elif scope == "last_block_classifier":
            keep_last_block_adaptation_modes(model)
        elif scope == "last_two_blocks_classifier":
            keep_last_two_block_adaptation_modes(model)
        sums = {
            "total": 0.0,
            "classification": 0.0,
            "prototype": 0.0,
            "power_distillation": 0.0,
            "power_ordering": 0.0,
            "stability": 0.0,
            "parameter": 0.0,
            "starting_point": 0.0,
        }
        pair_view_samples = 0
        power_eligible_samples = 0
        directional_margins = []
        unchanged_drifts = []
        gradient_cosines = []
        classification_gradient_norms = []
        ordering_gradient_norms = []
        gradient_diagnostic_recorded = False
        sample_count = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            progress = float(global_step) / float(max(total_steps - 1, 1))
            auxiliary_scale = _auxiliary_scale(
                progress, warmup_fraction, ramp_end_fraction
            )
            targets = batch["label"].to(device, non_blocking=True).float()
            is_pair = "reference_signal" in batch
            reference_signal = batch.get("reference_signal", batch.get("signal"))
            reference_signal = reference_signal.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                reference = model(
                    reference_signal,
                    return_aux=True,
                    parameter_stop_gradient=parameter_stop_gradient,
                )
                classification_value = classification_loss(
                    reference["logits"], targets
                )
                prototype_value = reference["logits"].sum() * 0.0
                power_distillation = reference["logits"].sum() * 0.0
                power_ordering = reference["logits"].sum() * 0.0
                stability_value = reference["logits"].sum() * 0.0
                parameter_value = reference["logits"].sum() * 0.0
                starting_point_value = reference["logits"].sum() * 0.0
                update_embeddings = reference["label_embeddings"]
                update_labels = targets

                eligible = None
                paired = None
                if is_pair and pair_views_enabled:
                    paired_signal = batch["paired_signal"].to(
                        device, non_blocking=True
                    )
                    paired = model(
                        paired_signal,
                        return_aux=True,
                        parameter_stop_gradient=parameter_stop_gradient,
                    )
                    classification_value = 0.5 * (
                        classification_value
                        + classification_loss(paired["logits"], targets)
                    )
                    pair_view_samples += len(targets)
                    update_embeddings = torch.cat(
                        [update_embeddings, paired["label_embeddings"]], dim=0
                    )
                    update_labels = torch.cat([update_labels, targets], dim=0)

                    gap = torch.as_tensor(
                        batch["relative_power_gap"],
                        dtype=torch.float32,
                        device=device,
                    )
                    maximum_gap = power_curriculum_max_gap(progress)
                    eligible = gap <= float(maximum_gap)
                    if bool(eligible.any()):
                        paired_targets = targets[eligible]
                        paired_eligible = {
                            key: (
                                value[eligible]
                                if torch.is_tensor(value) and value.shape[0] == len(targets)
                                else value
                            )
                            for key, value in paired.items()
                        }
                        power_eligible_samples += int(eligible.sum().item())
                        if distillation_enabled:
                            power_distillation = label_paired_power_consistency_loss(
                                reference["label_embeddings"][eligible],
                                paired_eligible["label_embeddings"],
                                paired_targets,
                                batch["stronger_level_by_label"].to(device)[eligible],
                                torch.as_tensor(batch["reference_level"], device=device)[eligible],
                                torch.as_tensor(batch["paired_level"], device=device)[eligible],
                                label_weights=label_weights,
                            )
                        if ordering_enabled:
                            ordering_reference_logits = reference["logits"]
                            ordering_paired_logits = paired_eligible["logits"]
                            if str(ordering_gradient_mode) == "backbone_stop_gradient":
                                ordering_reference_logits = model.backbone.forward_classifier(
                                    reference["features"][eligible].detach()
                                )
                                ordering_paired_logits = model.backbone.forward_classifier(
                                    paired_eligible["features"].detach()
                                )
                            power_ordering = directional_power_ordering_loss(
                                ordering_reference_logits,
                                ordering_paired_logits,
                                paired_targets,
                                batch["stronger_level_by_label"].to(device)[eligible],
                                torch.as_tensor(batch["reference_level"], device=device)[eligible],
                                torch.as_tensor(batch["paired_level"], device=device)[eligible],
                                margin=ordering_margin,
                                label_weights=label_weights,
                                mode=ordering_mode,
                            )
                            if not gradient_diagnostic_recorded:
                                shared_parameters = [
                                    parameter
                                    for parameter in model.backbone.parameters()
                                    if parameter.requires_grad
                                ]
                                classification_gradients = torch.autograd.grad(
                                    classification_value,
                                    shared_parameters,
                                    retain_graph=True,
                                    allow_unused=True,
                                )
                                ordering_gradients = torch.autograd.grad(
                                    power_ordering,
                                    shared_parameters,
                                    retain_graph=True,
                                    allow_unused=True,
                                )
                                classification_flat = torch.cat(
                                    [
                                        (
                                            gradient.reshape(-1)
                                            if gradient is not None
                                            else torch.zeros_like(parameter).reshape(-1)
                                        )
                                        for parameter, gradient in zip(
                                            shared_parameters,
                                            classification_gradients,
                                        )
                                    ]
                                )
                                ordering_flat = torch.cat(
                                    [
                                        (
                                            gradient.reshape(-1)
                                            if gradient is not None
                                            else torch.zeros_like(parameter).reshape(-1)
                                        )
                                        for parameter, gradient in zip(
                                            shared_parameters,
                                            ordering_gradients,
                                        )
                                    ]
                                )
                                classification_norm = classification_flat.norm()
                                ordering_norm = ordering_flat.norm()
                                classification_gradient_norms.append(
                                    float(classification_norm.detach().cpu().item())
                                )
                                ordering_gradient_norms.append(
                                    float(ordering_norm.detach().cpu().item())
                                )
                                if float(ordering_norm.detach().cpu().item()) > 0.0:
                                    cosine = torch.dot(
                                        classification_flat, ordering_flat
                                    ) / (
                                        classification_norm.clamp_min(1e-12)
                                        * ordering_norm.clamp_min(1e-12)
                                    )
                                    gradient_cosines.append(
                                        float(cosine.detach().cpu().item())
                                    )
                                gradient_diagnostic_recorded = True
                        if bool(use_stability):
                            stability_value = unchanged_component_stability_loss(
                                reference["logits"][eligible],
                                paired_eligible["logits"],
                                paired_targets,
                                batch["stronger_level_by_label"].to(device)[eligible],
                                torch.as_tensor(batch["reference_level"], device=device)[eligible],
                                torch.as_tensor(batch["paired_level"], device=device)[eligible],
                                tolerance=stability_tolerance,
                                label_weights=label_weights,
                            )
                        margin_values, drift_values = paired_power_diagnostic_values(
                            reference["logits"][eligible],
                            paired_eligible["logits"],
                            paired_targets,
                            batch["stronger_level_by_label"].to(device)[eligible],
                            torch.as_tensor(batch["reference_level"], device=device)[eligible],
                            torch.as_tensor(batch["paired_level"], device=device)[eligible],
                        )
                        directional_margins.extend(
                            margin_values.detach().cpu().float().tolist()
                        )
                        unchanged_drifts.extend(
                            drift_values.detach().cpu().float().tolist()
                        )

                if bool(use_prototype):
                    prototype_value = label_prototype_contrastive_loss(
                        update_embeddings,
                        update_labels,
                        memory,
                        label_weights=label_weights,
                        temperature=prototype_temperature,
                    )
                if bool(use_parameter_heads):
                    parameter_target, parameter_mask = _parameter_batch(
                        batch,
                        "reference_" if is_pair else "",
                        class_names,
                        parameter_normalizer,
                        device,
                    )
                    parameter_value = masked_parameter_smooth_l1(
                        reference["parameter_predictions"],
                        parameter_target,
                        targets,
                        parameter_mask,
                    )
                ordering_active = float(
                    progress >= float(ordering_start_fraction)
                )
                if starting_point_anchors:
                    starting_point_value = sum(
                        torch.sum(
                            torch.square(named_parameters[name] - anchor)
                        )
                        for name, anchor in starting_point_anchors.items()
                    )
                total = classification_value + auxiliary_scale * (
                    float(prototype_weight) * prototype_value
                    + float(power_weight)
                    * (power_distillation + ordering_active * power_ordering)
                    + float(stability_weight) * stability_value
                    + float(parameter_weight) * parameter_value
                ) + anchor_weight * starting_point_value
            if not bool(torch.isfinite(total).all()):
                sample_ids = batch.get("identity", batch.get("sample_index", "unknown"))
                raise FloatingPointError("NaN/Inf loss for samples {}".format(sample_ids))
            scaler.scale(total).backward()
            if float(gradient_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip)
                )
            scaler.step(optimizer)
            scaler.update()
            if bool(use_prototype):
                memory.update(update_embeddings, update_labels)
            batch_size = len(targets)
            sums["total"] += float(total.detach().cpu().item()) * batch_size
            sums["classification"] += float(
                classification_value.detach().cpu().item()
            ) * batch_size
            sums["prototype"] += float(
                prototype_value.detach().cpu().item()
            ) * batch_size
            sums["power_distillation"] += float(
                power_distillation.detach().cpu().item()
            ) * batch_size
            sums["power_ordering"] += float(
                ordering_active * power_ordering.detach().cpu().item()
            ) * batch_size
            sums["stability"] += float(
                stability_value.detach().cpu().item()
            ) * batch_size
            sums["parameter"] += float(
                parameter_value.detach().cpu().item()
            ) * batch_size
            sums["starting_point"] += float(
                starting_point_value.detach().cpu().item()
            ) * batch_size
            sample_count += batch_size
            global_step += 1

        validation = _predict(
            model, validation_loader, device, use_amp=use_amp
        )
        validation_tail_aupr = float("nan")
        if tail_index is not None:
            validation_tail_aupr = float(
                validation["per_label"][tail_index]["aupr"]
            )
            validation["tail_aupr"] = validation_tail_aupr
        margin_array = np.asarray(directional_margins, dtype=np.float64)
        drift_array = np.asarray(unchanged_drifts, dtype=np.float64)
        prototype_state = prototype_diagnostics(memory)
        row = {
            "epoch": float(epoch),
            "train_loss": sums["total"] / max(sample_count, 1),
            "classification_loss": sums["classification"] / max(sample_count, 1),
            "prototype_loss": sums["prototype"] / max(sample_count, 1),
            "power_distillation_loss": sums["power_distillation"] / max(sample_count, 1),
            "power_ordering_loss": sums["power_ordering"] / max(sample_count, 1),
            "ordering_active": float(
                float(epoch) / float(max(int(epochs), 1))
                >= float(ordering_start_fraction)
            ),
            "ordering_gradient_cosine": (
                float(np.mean(gradient_cosines))
                if gradient_cosines else float("nan")
            ),
            "classification_gradient_norm": (
                float(np.mean(classification_gradient_norms))
                if classification_gradient_norms else float("nan")
            ),
            "ordering_gradient_norm": (
                float(np.mean(ordering_gradient_norms))
                if ordering_gradient_norms else float("nan")
            ),
            "power_loss": (
                sums["power_distillation"] + sums["power_ordering"]
            ) / max(sample_count, 1),
            "stability_loss": sums["stability"] / max(sample_count, 1),
            "parameter_loss": sums["parameter"] / max(sample_count, 1),
            "starting_point_loss": sums["starting_point"] / max(sample_count, 1),
            "weighted_starting_point_loss": (
                anchor_weight * sums["starting_point"] / max(sample_count, 1)
            ),
            "paired_view_samples": int(pair_view_samples),
            "power_eligible_samples": int(power_eligible_samples),
            "directional_positions": int(len(margin_array)),
            "directional_margin_mean": (
                float(margin_array.mean()) if len(margin_array) else float("nan")
            ),
            "directional_margin_p10": (
                float(np.quantile(margin_array, 0.10)) if len(margin_array) else float("nan")
            ),
            "directional_satisfaction_rate": (
                float((margin_array >= float(ordering_margin)).mean())
                if len(margin_array)
                else float("nan")
            ),
            "unchanged_positions": int(len(drift_array)),
            "unchanged_drift_mean": (
                float(drift_array.mean()) if len(drift_array) else float("nan")
            ),
            "unchanged_drift_p50": (
                float(np.quantile(drift_array, 0.50)) if len(drift_array) else float("nan")
            ),
            "unchanged_drift_p90": (
                float(np.quantile(drift_array, 0.90)) if len(drift_array) else float("nan")
            ),
            "unchanged_drift_p95": (
                float(np.quantile(drift_array, 0.95)) if len(drift_array) else float("nan")
            ),
            "prototype_positive_cosine_max_offdiag": float(
                prototype_state["positive_cosine_max_offdiag"]
            ),
            "prototype_negative_cosine_max_offdiag": float(
                prototype_state["negative_cosine_max_offdiag"]
            ),
            "auxiliary_scale": float(_auxiliary_scale(
                float(epoch) / float(max(int(epochs), 1)),
                warmup_fraction,
                ramp_end_fraction,
            )),
            "subset_acc": float(validation["subset_acc"]),
            "macro_f1": float(validation["macro_f1"]),
            "hamming_loss": float(validation["hamming_loss"]),
            "tail_aupr": validation_tail_aupr,
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(row)
        if _is_better(
            validation, best_primary, best_hamming, selection_metric_value
        ):
            best_primary = float(validation[selection_metric_value])
            best_subset = float(validation["subset_acc"])
            best_hamming = float(validation["hamming_loss"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_memory_state = {
                key: value.detach().cpu().clone()
                for key, value in memory.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break

    if best_state is None or best_memory_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    memory.load_state_dict(best_memory_state)
    diagnostics = prototype_diagnostics(memory)
    result = StudentATrainingResult(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_subset_acc=best_subset,
        best_hamming_loss=best_hamming,
        train_seconds=float(time.perf_counter() - started),
        history=history,
        prototype_diagnostics=diagnostics,
        selection_metric=selection_metric_value,
        best_selection_value=best_primary,
    )
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "prototype_memory_state_dict": best_memory_state,
                "best_epoch": result.best_epoch,
                "history": result.history,
                "prototype_diagnostics": diagnostics,
                "selection_metric": result.selection_metric,
                "best_selection_value": result.best_selection_value,
            },
            str(path),
        )
    return result
