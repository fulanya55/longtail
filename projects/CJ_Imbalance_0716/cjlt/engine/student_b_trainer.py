"""DAPS-CJNet-v2 trainer for low-label multi-label recognition."""

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn import functional as F

from cjlt.data.augmentations import strong_complex_augmentation, weak_complex_augmentation
from cjlt.engine.evaluator import compute_multilabel_metrics
from cjlt.losses.parameter import (
    masked_parameter_consistency,
    masked_parameter_smooth_l1,
    select_parameter_targets,
)
from cjlt.losses.pseudo_label import (
    DynamicPseudoLabelThresholds,
    build_pseudo_labels,
    masked_pseudo_label_bce,
    prototype_reliability,
)
from cjlt.models.ema_teacher import EMATeacher
from cjlt.models.prototype_memory import TailPrototypeMemory, prototype_diagnostics


@dataclass
class StudentBTrainingResult:
    best_epoch: int
    epochs_ran: int
    best_subset_acc: float
    best_hamming_loss: float
    selection_metric: str
    best_selection_value: float
    train_seconds: float
    history: List[Dict[str, float]]
    pseudo_label_statistics: List[Dict[str, object]]
    prototype_diagnostics: Dict[str, object]
    final_thresholds: Dict[str, List[float]]


def _ramp(progress, warmup_fraction):
    warmup = float(warmup_fraction)
    if warmup <= 0.0:
        return 1.0
    return min(1.0, max(0.0, float(progress) / warmup))


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _sigmoid(logits):
    return torch.sigmoid(logits)


def _feature_consistency(student, teacher, mask):
    loss = 1.0 - (student * teacher.detach()).sum(dim=-1)
    weights = mask.float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _validation_metrics(model, loader, device, use_amp):
    model.eval()
    logits_parts = []
    target_parts = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch["signal"].to(device, non_blocking=True))
            logits_parts.append(logits.detach().cpu().numpy())
            target_parts.append(batch["label"].numpy())
    logits = np.concatenate(logits_parts, axis=0)
    target = np.concatenate(target_parts, axis=0).astype(np.uint8)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    prediction = (probability >= 0.5).astype(np.uint8)
    return compute_multilabel_metrics(target, prediction, probabilities=probability)


def _is_better(metrics, best_primary, best_hamming, selection_metric):
    metric = str(selection_metric)
    if metric not in ("subset_acc", "macro_f1", "macro_aupr"):
        raise ValueError("selection_metric must be subset_acc, macro_f1, or macro_aupr")
    primary = float(metrics[metric])
    hamming = float(metrics["hamming_loss"])
    return bool(
        primary > best_primary
        or (np.isclose(primary, best_primary) and hamming < best_hamming)
    )


def _labeled_parameter_batch(batch, class_names, normalizer, device):
    if "parameter_target" in batch:
        target = batch["parameter_target"].to(device).float()
        mask = batch["parameter_target_mask"].to(device).bool()
    else:
        if class_names is None:
            raise ValueError("class_names are required for raw 79-D parameters")
        target, mask = select_parameter_targets(
            batch["parameter_values"], batch["parameter_mask"], class_names
        )
        target = target.to(device)
        mask = mask.to(device)
    return normalizer.transform(target), mask


@torch.no_grad()
def _calibration_precision_coverage(
    teacher,
    loader,
    memory,
    thresholds,
    device,
    use_prototype_gate,
    reliability_threshold,
):
    positive_count = np.zeros(memory.num_labels, dtype=np.int64)
    true_positive = np.zeros(memory.num_labels, dtype=np.int64)
    sample_count = 0
    teacher.model.eval()
    for batch in loader:
        output = teacher(batch["signal"].to(device), return_aux=True)
        probability = _sigmoid(output["logits"])
        if use_prototype_gate and bool(
            torch.logical_and(
                memory.positive_initialized, memory.negative_initialized
            ).any()
        ):
            reliability = prototype_reliability(
                output["label_embeddings"],
                memory.positive_prototype,
                memory.negative_prototype,
            )
        else:
            reliability = torch.ones_like(probability)
        pseudo = build_pseudo_labels(
            probability,
            thresholds.tau_pos,
            thresholds.tau_neg,
            reliability=reliability,
            reliability_threshold=reliability_threshold if use_prototype_gate else 0.0,
        )
        truth = batch["label"].to(device).bool()
        positive_count += pseudo.positive_mask.sum(dim=0).cpu().numpy()
        true_positive += torch.logical_and(
            pseudo.positive_mask, truth
        ).sum(dim=0).cpu().numpy()
        sample_count += len(truth)
    precision = np.divide(
        true_positive,
        np.maximum(positive_count, 1),
        dtype=np.float64,
    )
    coverage = positive_count.astype(np.float64) / float(max(sample_count, 1))
    return precision, coverage


def train_student_b(
    model,
    labeled_loader,
    unlabeled_loader,
    validation_loader,
    calibration_loader,
    classification_loss,
    parameter_normalizer,
    parameter_names,
    device,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    checkpoint_path: Optional[Path] = None,
    class_names=None,
    labeled_prior=None,
    use_ema_teacher=True,
    use_dynamic_threshold=True,
    use_prototype_gate=True,
    use_feature_consistency=True,
    use_distribution_alignment=False,
    use_parameter_heads=False,
    parameter_stop_gradient=False,
    tau_pos=0.95,
    tau_neg=0.05,
    tau_pos_floor=0.70,
    tau_neg_ceiling=0.20,
    reliability_threshold=0.60,
    stability_ceiling=0.10,
    lambda_u=1.0,
    lambda_feature=0.10,
    lambda_supervised_parameter=0.10,
    lambda_unsupervised_parameter=0.20,
    lambda_marginal=0.02,
    unsupervised_warmup_fraction=0.20,
    teacher_warmup_epochs=0,
    threshold_update_interval=5,
    target_pseudo_precision=0.95,
    minimum_pseudo_coverage=0.10,
    balance_pseudo_sides=True,
    max_pseudo_side_fraction=1.0,
    prototype_momentum=0.99,
    gradient_clip=5.0,
    use_amp=False,
    augmentation_seed=2026,
    selection_metric="subset_acc",
):
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1")
    if use_parameter_heads and parameter_normalizer is None:
        raise ValueError("parameter_normalizer is required for parameter heads")
    model.to(device)
    classification_loss.to(device)
    teacher = EMATeacher(model).to(device)
    memory = TailPrototypeMemory(
        model.num_classes, model.relation_dim, momentum=prototype_momentum
    ).to(device)
    thresholds = DynamicPseudoLabelThresholds(
        model.num_classes,
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        tau_pos_floor=tau_pos_floor,
        tau_neg_ceiling=tau_neg_ceiling,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    generator = torch.Generator().manual_seed(int(augmentation_seed))
    best_state = None
    best_teacher_state = None
    best_epoch = 0
    selection_metric_value = str(selection_metric)
    if selection_metric_value not in ("subset_acc", "macro_f1", "macro_aupr"):
        raise ValueError("selection_metric must be subset_acc, macro_f1, or macro_aupr")
    best_primary = -math.inf
    best_subset = -math.inf
    best_hamming = math.inf
    bad_epochs = 0
    history = []
    pseudo_history = []
    marginal_ema = torch.full((model.num_classes,), 0.5, device=device)
    if labeled_prior is None:
        prior = torch.full((model.num_classes,), 0.5, device=device)
    else:
        prior = torch.as_tensor(labeled_prior, dtype=torch.float32, device=device).clamp(0.01, 0.99)
    semi_supervised = unlabeled_loader is not None and bool(use_ema_teacher)
    # Keep supervised exposure identical across B0 and semi-supervised variants.
    # The unlabeled iterator is cycled to match one full labeled pass per epoch.
    steps_per_epoch = len(labeled_loader)
    total_steps = max(1, int(epochs) * steps_per_epoch)
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        teacher.model.eval()
        labeled_iterator = iter(labeled_loader)
        unlabeled_iterator = iter(unlabeled_loader) if semi_supervised else None
        sums = {
            "total": 0.0,
            "sup_cls": 0.0,
            "unsup_cls": 0.0,
            "feature": 0.0,
            "sup_param": 0.0,
            "unsup_param": 0.0,
            "marginal": 0.0,
        }
        accepted_count = torch.zeros(model.num_classes, dtype=torch.long)
        positive_count = torch.zeros(model.num_classes, dtype=torch.long)
        confidence_candidate_count = torch.zeros(model.num_classes, dtype=torch.long)
        stability_pass_count = torch.zeros(model.num_classes, dtype=torch.long)
        stability_rejected_count = torch.zeros(model.num_classes, dtype=torch.long)
        confidence_sum = torch.zeros(model.num_classes)
        reliability_sum = torch.zeros(model.num_classes)
        probability_parts = []
        unlabeled_examples = 0
        labeled_examples = 0
        epoch_started = time.perf_counter()

        for _ in range(steps_per_epoch):
            progress = float(global_step) / float(max(total_steps - 1, 1))
            unsupervised_scale = _ramp(progress, unsupervised_warmup_fraction)
            labeled_batch, labeled_iterator = _next_or_restart(
                labeled_iterator, labeled_loader
            )
            label = labeled_batch["label"].to(device).float()
            labeled_examples += len(label)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                supervised = model(
                    labeled_batch["signal"].to(device),
                    return_aux=True,
                    parameter_stop_gradient=parameter_stop_gradient,
                )
                supervised_classification = classification_loss(
                    supervised["logits"], label
                )
                supervised_parameter = supervised_classification * 0.0
                if use_parameter_heads:
                    parameter_target, parameter_mask = _labeled_parameter_batch(
                        labeled_batch, class_names, parameter_normalizer, device
                    )
                    supervised_parameter = masked_parameter_smooth_l1(
                        supervised["parameter_predictions"],
                        parameter_target,
                        label,
                        parameter_mask,
                    )

                memory.update(supervised["label_embeddings"], label)
                unsupervised_classification = supervised_classification * 0.0
                feature_value = supervised_classification * 0.0
                unsupervised_parameter = supervised_classification * 0.0
                marginal_value = supervised_classification * 0.0

                if semi_supervised and epoch > int(teacher_warmup_epochs):
                    unlabeled_batch, unlabeled_iterator = _next_or_restart(
                        unlabeled_iterator, unlabeled_loader
                    )
                    raw_unlabeled = unlabeled_batch["signal"]
                    weak_one = weak_complex_augmentation(
                        raw_unlabeled, parameter_names, generator=generator
                    )
                    weak_two = weak_complex_augmentation(
                        raw_unlabeled, parameter_names, generator=generator
                    )
                    strong = strong_complex_augmentation(
                        raw_unlabeled, parameter_names, generator=generator
                    )
                    with torch.no_grad():
                        teacher_one = teacher(
                            weak_one.signal.to(device), return_aux=True
                        )
                        teacher_two = teacher(
                            weak_two.signal.to(device), return_aux=True
                        )
                        teacher_logits = teacher_one["logits"]
                        teacher_probability = _sigmoid(teacher_logits)
                        second_probability = _sigmoid(teacher_two["logits"])
                        stability = torch.abs(
                            teacher_probability - second_probability
                        )
                        if use_distribution_alignment:
                            marginal_ema.mul_(0.99).add_(
                                teacher_probability.mean(dim=0), alpha=0.01
                            )
                            correction = (
                                torch.logit(prior) - torch.logit(marginal_ema.clamp(0.01, 0.99))
                            ).clamp(-1.0, 1.0)
                            teacher_probability = _sigmoid(
                                teacher_logits + 0.5 * correction[None, :]
                            )
                        if use_prototype_gate:
                            reliability = prototype_reliability(
                                teacher_one["label_embeddings"],
                                memory.positive_prototype,
                                memory.negative_prototype,
                            )
                        else:
                            reliability = torch.ones_like(teacher_probability)
                        pseudo = build_pseudo_labels(
                            teacher_probability,
                            thresholds.tau_pos,
                            thresholds.tau_neg,
                            reliability=reliability,
                            reliability_threshold=(
                                reliability_threshold if use_prototype_gate else 0.0
                            ),
                            stability=stability,
                            stability_ceiling=stability_ceiling,
                            max_positive_fraction=max_pseudo_side_fraction,
                            max_negative_fraction=max_pseudo_side_fraction,
                        )
                    tau_pos_tensor = torch.as_tensor(
                        thresholds.tau_pos,
                        dtype=teacher_probability.dtype,
                        device=device,
                    )[None, :]
                    tau_neg_tensor = torch.as_tensor(
                        thresholds.tau_neg,
                        dtype=teacher_probability.dtype,
                        device=device,
                    )[None, :]
                    confidence_candidate = torch.logical_or(
                        teacher_probability >= tau_pos_tensor,
                        teacher_probability <= tau_neg_tensor,
                    )
                    stable_mask = stability <= float(stability_ceiling)
                    probability_parts.append(teacher_probability.detach().cpu())
                    confidence_candidate_count += confidence_candidate.sum(dim=0).cpu()
                    stability_pass_count += stable_mask.sum(dim=0).cpu()
                    stability_rejected_count += torch.logical_and(
                        confidence_candidate, torch.logical_not(stable_mask)
                    ).sum(dim=0).cpu()
                    if bool(pseudo.accepted_mask.any()):
                        student_unlabeled = model(
                            strong.signal.to(device),
                            return_aux=True,
                            parameter_stop_gradient=parameter_stop_gradient,
                        )
                        unsupervised_classification = masked_pseudo_label_bce(
                            student_unlabeled["logits"],
                            pseudo.targets,
                            pseudo.accepted_mask,
                            balance_sides=balance_pseudo_sides,
                        )
                        if use_feature_consistency:
                            feature_value = _feature_consistency(
                                student_unlabeled["label_embeddings"],
                                teacher_one["label_embeddings"],
                                pseudo.accepted_mask,
                            )
                        if use_parameter_heads:
                            parameter_count = model.parameters_per_label
                            weak_mask = weak_one.preserves_parameter_mask.to(device).reshape(
                                -1, model.num_classes, parameter_count
                            )
                            strong_mask = strong.preserves_parameter_mask.to(device).reshape(
                                -1, model.num_classes, parameter_count
                            )
                            parameter_valid = torch.logical_and(weak_mask, strong_mask)
                            parameter_valid = torch.logical_and(
                                parameter_valid,
                                pseudo.positive_mask[:, :, None],
                            )
                            unsupervised_parameter = masked_parameter_consistency(
                                student_unlabeled["parameter_predictions"],
                                teacher_one["parameter_predictions"],
                                parameter_valid,
                            )
                        if use_distribution_alignment:
                            marginal_value = F.mse_loss(
                                _sigmoid(student_unlabeled["logits"]).mean(dim=0), prior
                            )
                    accepted_cpu = pseudo.accepted_mask.sum(dim=0).cpu()
                    accepted_count += accepted_cpu
                    positive_count += pseudo.positive_mask.sum(dim=0).cpu()
                    confidence_sum += (
                        teacher_probability * pseudo.accepted_mask.float()
                    ).sum(dim=0).cpu()
                    reliability_sum += (
                        pseudo.reliability * pseudo.accepted_mask.float()
                    ).sum(dim=0).cpu()
                    unlabeled_examples += len(raw_unlabeled)

                total = (
                    supervised_classification
                    + float(lambda_supervised_parameter) * supervised_parameter
                    + unsupervised_scale
                    * (
                        float(lambda_u) * unsupervised_classification
                        + float(lambda_feature) * feature_value
                        + float(lambda_unsupervised_parameter) * unsupervised_parameter
                        + float(lambda_marginal) * marginal_value
                    )
                )
            scaler.scale(total).backward()
            if float(gradient_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            momentum = teacher.update(model, progress) if use_ema_teacher else 0.0
            for key, value in (
                ("total", total),
                ("sup_cls", supervised_classification),
                ("unsup_cls", unsupervised_classification),
                ("feature", feature_value),
                ("sup_param", supervised_parameter),
                ("unsup_param", unsupervised_parameter),
                ("marginal", marginal_value),
            ):
                sums[key] += float(value.detach().cpu().item())
            global_step += 1

        if (
            use_dynamic_threshold
            and calibration_loader is not None
            and int(threshold_update_interval) > 0
            and epoch > int(teacher_warmup_epochs)
            and epoch % int(threshold_update_interval) == 0
        ):
            precision, coverage = _calibration_precision_coverage(
                teacher,
                calibration_loader,
                memory,
                thresholds,
                device,
                use_prototype_gate,
                reliability_threshold,
            )
            thresholds.update(
                precision,
                coverage,
                target_precision=target_pseudo_precision,
                minimum_coverage=minimum_pseudo_coverage,
            )

        validation = _validation_metrics(model, validation_loader, device, use_amp)
        denominator = float(max(steps_per_epoch, 1))
        total_positions = float(max(unlabeled_examples * model.num_classes, 1))
        row = {
            "epoch": float(epoch),
            "train_loss": sums["total"] / denominator,
            "supervised_classification_loss": sums["sup_cls"] / denominator,
            "unsupervised_classification_loss": sums["unsup_cls"] / denominator,
            "feature_consistency_loss": sums["feature"] / denominator,
            "supervised_parameter_loss": sums["sup_param"] / denominator,
            "unsupervised_parameter_loss": sums["unsup_param"] / denominator,
            "marginal_loss": sums["marginal"] / denominator,
            "pseudo_coverage": float(accepted_count.sum().item()) / total_positions,
            "subset_acc": float(validation["subset_acc"]),
            "macro_f1": float(validation["macro_f1"]),
            "macro_aupr": float(validation["macro_aupr"]),
            "hamming_loss": float(validation["hamming_loss"]),
            "teacher_momentum": float(momentum),
            "steps_per_epoch": int(steps_per_epoch),
            "labeled_examples": int(labeled_examples),
            "unlabeled_examples": int(unlabeled_examples),
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(row)
        if probability_parts:
            probability_values = torch.cat(probability_parts, dim=0).numpy()
            probability_quantiles = {
                "p10": np.quantile(probability_values, 0.10, axis=0).tolist(),
                "p50": np.quantile(probability_values, 0.50, axis=0).tolist(),
                "p90": np.quantile(probability_values, 0.90, axis=0).tolist(),
                "p95": np.quantile(probability_values, 0.95, axis=0).tolist(),
            }
        else:
            probability_quantiles = {
                key: [float("nan")] * model.num_classes
                for key in ("p10", "p50", "p90", "p95")
            }
        pseudo_history.append(
            {
                "epoch": epoch,
                "accepted_count": accepted_count.tolist(),
                "positive_count": positive_count.tolist(),
                "negative_count": (accepted_count - positive_count).tolist(),
                "confidence_candidate_count": confidence_candidate_count.tolist(),
                "stability_pass_count": stability_pass_count.tolist(),
                "stability_rejected_count": stability_rejected_count.tolist(),
                "probability_quantiles": probability_quantiles,
                "coverage": (
                    accepted_count.float() / float(max(unlabeled_examples, 1))
                ).tolist(),
                "mean_confidence": (
                    confidence_sum / accepted_count.float().clamp_min(1.0)
                ).tolist(),
                "mean_reliability": (
                    reliability_sum / accepted_count.float().clamp_min(1.0)
                ).tolist(),
            }
        )
        if _is_better(validation, best_primary, best_hamming, selection_metric_value):
            best_primary = float(validation[selection_metric_value])
            best_subset = float(validation["subset_acc"])
            best_hamming = float(validation["hamming_loss"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_teacher_state = {
                key: value.detach().cpu().clone()
                for key, value in teacher.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break

    if best_state is None or best_teacher_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    teacher.load_state_dict(best_teacher_state)
    final_thresholds = {
        "tau_pos": [float(value) for value in thresholds.tau_pos],
        "tau_neg": [float(value) for value in thresholds.tau_neg],
    }
    result = StudentBTrainingResult(
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_subset_acc=best_subset,
        best_hamming_loss=best_hamming,
        selection_metric=selection_metric_value,
        best_selection_value=best_primary,
        train_seconds=float(time.perf_counter() - started),
        history=history,
        pseudo_label_statistics=pseudo_history,
        prototype_diagnostics=prototype_diagnostics(memory),
        final_thresholds=final_thresholds,
    )
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "teacher_state_dict": best_teacher_state,
                "prototype_memory_state_dict": memory.state_dict(),
                "parameter_normalizer_mean": None if parameter_normalizer is None else parameter_normalizer.mean,
                "parameter_normalizer_scale": None if parameter_normalizer is None else parameter_normalizer.scale,
                "thresholds": final_thresholds,
                "history": history,
            },
            str(path),
        )
    return result
