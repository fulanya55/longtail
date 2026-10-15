"""Trainer and diagnostics for supervised joint parameter awareness."""

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cjlt.engine.evaluator import compute_multilabel_metrics
from cjlt.losses.parameter import (
    masked_parameter_smooth_l1,
    masked_parameter_smooth_l1_by_target,
    select_parameter_targets,
)


@dataclass
class JointPredictionOutput:
    logits: np.ndarray
    targets: np.ndarray
    sample_ids: np.ndarray
    parameter_predictions: Optional[np.ndarray]
    parameter_targets: Optional[np.ndarray]
    parameter_masks: Optional[np.ndarray]


@dataclass
class JointTrainingResult:
    best_epoch: int
    epochs_ran: int
    best_macro_aupr: float
    best_hamming_loss: float
    train_seconds: float
    history: List[Dict[str, float]]


def parameter_weight_at_progress(progress, maximum, warmup_fraction, ramp_end_fraction):
    value = float(progress)
    maximum_value = float(maximum)
    warmup = float(warmup_fraction)
    ramp_end = float(ramp_end_fraction)
    if maximum_value < 0.0:
        raise ValueError("maximum parameter weight must be nonnegative")
    if not 0.0 <= warmup <= ramp_end <= 1.0:
        raise ValueError("require 0 <= warmup_fraction <= ramp_end_fraction <= 1")
    if value <= warmup:
        return 0.0
    if ramp_end <= warmup or value >= ramp_end:
        return maximum_value
    return maximum_value * (value - warmup) / (ramp_end - warmup)


def gradient_dot_and_norms(first, second):
    """Return dot product and squared norms for two aligned gradient lists."""

    dot = None
    first_norm = None
    second_norm = None
    for first_value, second_value in zip(first, second):
        if first_value is None or second_value is None:
            continue
        current_dot = (first_value * second_value).sum()
        current_first = first_value.square().sum()
        current_second = second_value.square().sum()
        dot = current_dot if dot is None else dot + current_dot
        first_norm = current_first if first_norm is None else first_norm + current_first
        second_norm = (
            current_second if second_norm is None else second_norm + current_second
        )
    if dot is None:
        return None, None, None
    return dot, first_norm, second_norm


def project_conflicting_gradients(task_gradients, reference_gradients, epsilon=1.0e-12):
    """Remove only the component opposing the classification gradient."""

    dot, _, reference_norm = gradient_dot_and_norms(
        task_gradients, reference_gradients
    )
    if dot is None or float(dot.detach().cpu()) >= 0.0:
        return list(task_gradients), False
    coefficient = dot / reference_norm.clamp_min(float(epsilon))
    projected = []
    for task_value, reference_value in zip(task_gradients, reference_gradients):
        if task_value is None:
            projected.append(None)
        elif reference_value is None:
            projected.append(task_value)
        else:
            projected.append(task_value - coefficient * reference_value)
    return projected, True


def _gradient_diagnostics(task_gradients, classification_gradients):
    dot, task_norm, classification_norm = gradient_dot_and_norms(
        task_gradients, classification_gradients
    )
    if dot is None:
        return {"cosine": float("nan"), "task_norm": 0.0}
    denominator = (task_norm * classification_norm).clamp_min(1.0e-24).sqrt()
    return {
        "cosine": float((dot / denominator).detach().cpu()),
        "task_norm": float(task_norm.clamp_min(0.0).sqrt().detach().cpu()),
    }


def _assign_gradient(parameter, gradient):
    parameter.grad = None if gradient is None else gradient.detach().clone()


def _selective_parameter_backward(
    model,
    classification_value,
    parameter_value,
    target_losses,
    parameter_weight,
    transfer_mask,
    strategy,
    collect_diagnostics,
):
    """Backpropagate private parameter heads and selectively route shared gradients."""

    shared_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    shared_ids = {id(parameter) for parameter in shared_parameters}
    private_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in shared_ids
    ]
    classification_gradients = torch.autograd.grad(
        classification_value,
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    combined_gradients = [
        None if value is None else value.clone() for value in classification_gradients
    ]
    diagnostics = {}
    projected_count = 0
    selected_losses = []
    selected_indices = []
    for class_index in range(target_losses.shape[0]):
        for parameter_index in range(target_losses.shape[1]):
            if bool(transfer_mask[class_index, parameter_index]):
                selected_losses.append(target_losses[class_index, parameter_index])
                selected_indices.append((class_index, parameter_index))

    if strategy == "selective":
        if selected_losses:
            transferable_value = torch.stack(selected_losses).sum()
            transferable_gradients = torch.autograd.grad(
                float(parameter_weight) * transferable_value,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            for index, value in enumerate(transferable_gradients):
                if value is None:
                    continue
                combined_gradients[index] = (
                    value
                    if combined_gradients[index] is None
                    else combined_gradients[index] + value
                )
        diagnostic_indices = selected_indices if collect_diagnostics else []
        for class_index, parameter_index in diagnostic_indices:
            gradients = torch.autograd.grad(
                target_losses[class_index, parameter_index],
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            diagnostics[(class_index, parameter_index)] = _gradient_diagnostics(
                gradients, classification_gradients
            )
    elif strategy == "target_pcgrad":
        for class_index, parameter_index in selected_indices:
            raw_gradients = torch.autograd.grad(
                target_losses[class_index, parameter_index],
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if collect_diagnostics:
                diagnostics[(class_index, parameter_index)] = _gradient_diagnostics(
                    raw_gradients, classification_gradients
                )
            weighted_gradients = [
                None if value is None else float(parameter_weight) * value
                for value in raw_gradients
            ]
            routed_gradients, projected = project_conflicting_gradients(
                weighted_gradients, classification_gradients
            )
            projected_count += int(projected)
            for index, value in enumerate(routed_gradients):
                if value is None:
                    continue
                combined_gradients[index] = (
                    value
                    if combined_gradients[index] is None
                    else combined_gradients[index] + value
                )
    else:
        raise ValueError("unknown selective gradient strategy: {!r}".format(strategy))

    private_gradients = torch.autograd.grad(
        float(parameter_weight) * parameter_value,
        private_parameters,
        retain_graph=False,
        allow_unused=True,
    )
    for parameter, gradient in zip(shared_parameters, combined_gradients):
        _assign_gradient(parameter, gradient)
    for parameter, gradient in zip(private_parameters, private_gradients):
        _assign_gradient(parameter, gradient)
    diagnostics["projected_target_count"] = int(projected_count)
    return diagnostics


def _sigmoid(logits):
    values = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(
        np.float32
    )


def predict_joint(
    model,
    loader,
    device,
    class_names,
    parameter_normalizer=None,
    use_amp=False,
):
    model.eval()
    logits_parts = []
    target_parts = []
    id_parts = []
    prediction_parts = []
    parameter_target_parts = []
    parameter_mask_parts = []
    generated_offset = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device, non_blocking=True)
            label = batch["label"]
            sample_ids = batch.get("sample_index")
            if sample_ids is None:
                sample_ids = torch.arange(
                    generated_offset, generated_offset + len(signal), dtype=torch.int64
                )
            elif not torch.is_tensor(sample_ids):
                sample_ids = torch.as_tensor(sample_ids, dtype=torch.int64)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(signal, return_aux=True)
            logits_parts.append(output["logits"].detach().cpu().numpy().astype(np.float32))
            target_parts.append(label.detach().cpu().numpy().astype(np.uint8))
            id_parts.append(sample_ids.detach().cpu().numpy().astype(np.int64))
            if output["parameter_predictions"] is not None:
                if parameter_normalizer is None:
                    raise ValueError("parameter_normalizer is required for parameter outputs")
                prediction = parameter_normalizer.inverse_transform(
                    output["parameter_predictions"]
                )
                if output.get("mft_count_logits") is not None:
                    mft_index = int(model.mft_class_index)
                    values = torch.arange(
                        3,
                        9,
                        device=output["mft_count_logits"].device,
                        dtype=output["mft_count_logits"].dtype,
                    )
                    expected_count = (
                        output["mft_count_logits"].softmax(dim=-1) * values[None, :]
                    ).sum(dim=-1)
                    prediction = prediction.clone()
                    prediction[:, mft_index, 1] = expected_count
                target, mask = select_parameter_targets(
                    batch["parameter_values"], batch["parameter_mask"], class_names
                )
                prediction_parts.append(
                    prediction.detach().cpu().numpy().astype(np.float32)
                )
                parameter_target_parts.append(target.numpy().astype(np.float32))
                parameter_mask_parts.append(mask.numpy().astype(bool))
            generated_offset += len(signal)
    if not logits_parts:
        raise ValueError("prediction loader must not be empty")
    return JointPredictionOutput(
        logits=np.concatenate(logits_parts, axis=0),
        targets=np.concatenate(target_parts, axis=0),
        sample_ids=np.concatenate(id_parts, axis=0),
        parameter_predictions=(
            np.concatenate(prediction_parts, axis=0) if prediction_parts else None
        ),
        parameter_targets=(
            np.concatenate(parameter_target_parts, axis=0)
            if parameter_target_parts
            else None
        ),
        parameter_masks=(
            np.concatenate(parameter_mask_parts, axis=0)
            if parameter_mask_parts
            else None
        ),
    )


def compute_parameter_metrics(
    output,
    class_names,
    parameter_specs,
    parameter_normalizer,
    class_predictions=None,
):
    if output.parameter_predictions is None:
        return None
    prediction = np.asarray(output.parameter_predictions, dtype=np.float64)
    target = np.asarray(output.parameter_targets, dtype=np.float64)
    raw_mask = np.asarray(output.parameter_masks, dtype=bool)
    labels = np.asarray(output.targets, dtype=np.uint8)
    valid = np.logical_and(raw_mask, labels[:, :, None] == 1)
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("parameter prediction, target, and mask shapes must match")
    mean = parameter_normalizer.mean.detach().cpu().numpy().astype(np.float64)
    scale = parameter_normalizer.scale.detach().cpu().numpy().astype(np.float64)
    normalized_error = np.abs(prediction - target) / scale[None, :, :]
    rows = []
    oracle_values = []
    for class_index, class_name in enumerate(class_names):
        for parameter_index, spec in enumerate(parameter_specs[class_name]):
            selected = valid[:, class_index, parameter_index]
            error = np.abs(
                prediction[selected, class_index, parameter_index]
                - target[selected, class_index, parameter_index]
            )
            squared = np.square(error)
            nerror = normalized_error[selected, class_index, parameter_index]
            oracle_values.extend(nerror.tolist())
            rows.append(
                {
                    "class": str(class_name),
                    "parameter": str(spec.name),
                    "unit": str(spec.unit),
                    "count": int(selected.sum()),
                    "mae": float(error.mean()) if len(error) else float("nan"),
                    "rmse": float(np.sqrt(squared.mean())) if len(error) else float("nan"),
                    "normalized_mae": (
                        float(nerror.mean()) if len(nerror) else float("nan")
                    ),
                    "training_mean": float(mean[class_index, parameter_index]),
                    "training_scale": float(scale[class_index, parameter_index]),
                    **(
                        {
                            "rounded_mae": float(
                                np.abs(np.rint(prediction[selected, class_index, parameter_index]) - target[selected, class_index, parameter_index]).mean()
                            )
                            if len(error)
                            else float("nan"),
                            "rounded_accuracy": float(
                                (
                                    np.rint(prediction[selected, class_index, parameter_index])
                                    == target[selected, class_index, parameter_index]
                                ).mean()
                            )
                            if len(error)
                            else float("nan"),
                        }
                        if str(spec.unit) == "count"
                        else {}
                    ),
                }
            )
    result = {
        "oracle_valid_values": int(valid.sum()),
        "oracle_macro_target_nmae": float(
            np.mean([row["normalized_mae"] for row in rows])
        ),
        "oracle_micro_nmae": float(np.mean(oracle_values)),
        "per_target": rows,
    }
    if class_predictions is not None:
        predicted = np.asarray(class_predictions, dtype=np.uint8)
        if predicted.shape != labels.shape:
            raise ValueError("class_predictions must match label shape")
        detected = np.logical_and(valid, predicted[:, :, None] == 1)
        detected_values = normalized_error[detected]
        result.update(
            {
                "system_detected_valid_values": int(detected.sum()),
                "system_parameter_coverage": float(detected.sum() / max(valid.sum(), 1)),
                "system_detected_micro_nmae": (
                    float(detected_values.mean())
                    if len(detected_values)
                    else float("nan")
                ),
            }
        )
    return result


def train_supervised_joint(
    model,
    train_loader,
    validation_loader,
    classification_loss,
    parameter_normalizer,
    class_names,
    device,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    checkpoint_path: Optional[Path] = None,
    use_parameter_heads=True,
    detach_parameter_backbone=False,
    maximum_parameter_weight=0.1,
    warmup_fraction=0.0,
    ramp_end_fraction=0.0,
    use_amp=False,
    gradient_clip=5.0,
    gradient_strategy="standard",
    parameter_transfer_mask=None,
    parameter_target_names=None,
    collect_gradient_diagnostics=False,
):
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1")
    if bool(use_parameter_heads) and parameter_normalizer is None:
        raise ValueError("parameter_normalizer is required when parameter heads are enabled")
    model.to(device)
    classification_loss.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    strategy = str(gradient_strategy).strip().lower()
    if strategy not in ("standard", "selective", "target_pcgrad"):
        raise ValueError("gradient_strategy must be standard, selective, or target_pcgrad")
    # Manual multi-loss gradient routing is kept in float32 so every target uses
    # one common, auditable scale.  Standard C2 retains the original AMP path.
    amp_enabled = bool(use_amp and device.type == "cuda" and strategy == "standard")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state = None
    best_epoch = 0
    best_macro_aupr = -math.inf
    best_hamming = math.inf
    bad_epochs = 0
    history = []
    total_steps = max(int(epochs) * len(train_loader), 1)
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        sums = {
            "total": 0.0,
            "classification": 0.0,
            "parameter": 0.0,
            "mft_count": 0.0,
        }
        epoch_gradient_diagnostics = {}
        weight_sum = 0.0
        sample_count = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            signal = batch["signal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()
            optimizer.zero_grad()
            progress = float(global_step) / float(max(total_steps - 1, 1))
            parameter_weight = (
                parameter_weight_at_progress(
                    progress,
                    maximum_parameter_weight,
                    warmup_fraction,
                    ramp_end_fraction,
                )
                if use_parameter_heads
                else 0.0
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(
                    signal,
                    return_aux=True,
                    detach_parameter_backbone=detach_parameter_backbone,
                )
                classification_value = classification_loss(output["logits"], labels)
                parameter_value = classification_value * 0.0
                mft_count_value = classification_value * 0.0
                target_losses = None
                if use_parameter_heads:
                    raw_target, mask = select_parameter_targets(
                        batch["parameter_values"], batch["parameter_mask"], class_names
                    )
                    raw_target = raw_target.to(device)
                    target = parameter_normalizer.transform(raw_target)
                    mask = mask.to(device)
                    regression_mask = mask
                    if output.get("mft_count_logits") is not None:
                        mft_index = int(model.mft_class_index)
                        regression_mask = mask.clone()
                        regression_mask[:, mft_index, 1] = False
                        count_valid = torch.logical_and(
                            labels[:, mft_index] > 0.5,
                            mask[:, mft_index, 1],
                        )
                        if bool(count_valid.any()):
                            count_target = (
                                raw_target[count_valid, mft_index, 1].round().long() - 3
                            ).clamp(0, 5)
                            mft_count_value = F.cross_entropy(
                                output["mft_count_logits"][count_valid], count_target
                            )
                    target_losses = masked_parameter_smooth_l1_by_target(
                        output["parameter_predictions"],
                        target,
                        labels,
                        regression_mask,
                    )
                    if output.get("mft_count_logits") is not None:
                        target_losses = target_losses.clone()
                        target_losses[mft_index, 1] = mft_count_value
                    parameter_value = target_losses.sum()
                total = classification_value + parameter_weight * parameter_value
            collect_batch_diagnostics = bool(
                collect_gradient_diagnostics
                and use_parameter_heads
                and parameter_weight > 0.0
                and not epoch_gradient_diagnostics
            )
            if strategy == "standard":
                if collect_batch_diagnostics:
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
                    for class_index in range(target_losses.shape[0]):
                        for parameter_index in range(target_losses.shape[1]):
                            gradients = torch.autograd.grad(
                                target_losses[class_index, parameter_index],
                                shared_parameters,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            epoch_gradient_diagnostics[(class_index, parameter_index)] = (
                                _gradient_diagnostics(
                                    gradients, classification_gradients
                                )
                            )
                scaler.scale(total).backward()
            else:
                if parameter_transfer_mask is None:
                    transfer_mask = torch.ones_like(target_losses, dtype=torch.bool)
                else:
                    transfer_mask = torch.as_tensor(
                        parameter_transfer_mask,
                        dtype=torch.bool,
                        device=target_losses.device,
                    )
                if transfer_mask.shape != target_losses.shape:
                    raise ValueError("parameter_transfer_mask must match [C, K]")
                batch_diagnostics = _selective_parameter_backward(
                    model=model,
                    classification_value=classification_value,
                    parameter_value=parameter_value,
                    target_losses=target_losses,
                    parameter_weight=parameter_weight,
                    transfer_mask=transfer_mask,
                    strategy=strategy,
                    collect_diagnostics=collect_batch_diagnostics,
                )
                if collect_batch_diagnostics:
                    epoch_gradient_diagnostics = batch_diagnostics
            if float(gradient_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(signal)
            sums["total"] += float(total.detach().cpu().item()) * batch_size
            sums["classification"] += (
                float(classification_value.detach().cpu().item()) * batch_size
            )
            sums["parameter"] += float(parameter_value.detach().cpu().item()) * batch_size
            sums["mft_count"] += float(mft_count_value.detach().cpu().item()) * batch_size
            weight_sum += float(parameter_weight) * batch_size
            sample_count += batch_size
            global_step += 1

        validation_output = predict_joint(
            model,
            validation_loader,
            device,
            class_names=class_names,
            parameter_normalizer=parameter_normalizer if use_parameter_heads else None,
            use_amp=use_amp,
        )
        probabilities = _sigmoid(validation_output.logits)
        predictions = (probabilities >= 0.5).astype(np.uint8)
        metrics = compute_multilabel_metrics(
            validation_output.targets, predictions, probabilities=probabilities
        )
        row = {
            "epoch": float(epoch),
            "train_loss": sums["total"] / max(sample_count, 1),
            "classification_loss": sums["classification"] / max(sample_count, 1),
            "parameter_loss": sums["parameter"] / max(sample_count, 1),
            "mft_count_loss": sums["mft_count"] / max(sample_count, 1),
            "mean_parameter_weight": weight_sum / max(sample_count, 1),
            "macro_aupr": float(metrics["macro_aupr"]),
            "macro_f1": float(metrics["macro_f1"]),
            "subset_acc": float(metrics["subset_acc"]),
            "hamming_loss": float(metrics["hamming_loss"]),
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        target_names = parameter_target_names or []
        negative_count = 0
        finite_count = 0
        for key, values in epoch_gradient_diagnostics.items():
            if key == "projected_target_count":
                row["projected_target_count"] = float(values)
                continue
            class_index, parameter_index = key
            flat_index = class_index * 2 + parameter_index
            name = (
                target_names[flat_index]
                if flat_index < len(target_names)
                else "c{}_p{}".format(class_index, parameter_index)
            )
            safe_name = "".join(
                character if character.isalnum() else "_" for character in name
            )
            cosine = float(values["cosine"])
            row["grad_cos_{}".format(safe_name)] = cosine
            row["grad_norm_{}".format(safe_name)] = float(values["task_norm"])
            if math.isfinite(cosine):
                finite_count += 1
                negative_count += int(cosine < 0.0)
        if finite_count:
            row["negative_gradient_fraction"] = float(negative_count / finite_count)
        history.append(row)
        macro_aupr = float(metrics["macro_aupr"])
        hamming = float(metrics["hamming_loss"])
        better = macro_aupr > best_macro_aupr or (
            np.isclose(macro_aupr, best_macro_aupr) and hamming < best_hamming
        )
        if better:
            best_macro_aupr = macro_aupr
            best_hamming = hamming
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    result = JointTrainingResult(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_macro_aupr=best_macro_aupr,
        best_hamming_loss=best_hamming,
        train_seconds=float(time.perf_counter() - started),
        history=history,
    )
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": result.best_epoch,
                "epochs_ran": result.epochs_ran,
                "best_macro_aupr": result.best_macro_aupr,
                "best_hamming_loss": result.best_hamming_loss,
                "history": result.history,
            },
            str(path),
        )
    return result
