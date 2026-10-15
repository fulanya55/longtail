"""Minimal-schema parameter selection, normalization, and masked regression."""

import torch
from torch.nn import functional as F

from cjlt.data.parameter_schema import get_parameter_specs


def select_parameter_targets(values79, mask79, class_names):
    values = torch.as_tensor(values79).float()
    mask = torch.as_tensor(mask79).bool()
    if values.shape != mask.shape or values.ndim != 2 or values.shape[1] != 79:
        raise ValueError("parameter values/mask must have shape [B, 79]")
    selected_values = []
    selected_masks = []
    for class_name in class_names:
        specs = get_parameter_specs(class_name)
        columns = [spec.column_index for spec in specs]
        selected_values.append(values[:, columns])
        selected_masks.append(mask[:, columns])
    return torch.stack(selected_values, dim=1), torch.stack(selected_masks, dim=1)


class ParameterNormalizer:
    def __init__(self, epsilon=1e-6):
        self.epsilon = float(epsilon)
        self.mean = None
        self.scale = None

    def fit(self, targets, valid_mask):
        values = torch.as_tensor(targets).float()
        mask = torch.as_tensor(valid_mask).bool()
        if values.shape != mask.shape or values.ndim != 3:
            raise ValueError("targets and valid_mask must match [B, C, K]")
        mean = torch.zeros(values.shape[1:], dtype=values.dtype)
        scale = torch.ones(values.shape[1:], dtype=values.dtype)
        for class_index in range(values.shape[1]):
            for parameter_index in range(values.shape[2]):
                valid = mask[:, class_index, parameter_index]
                if not bool(valid.any()):
                    raise ValueError("every parameter requires at least one valid training value")
                selected = values[valid, class_index, parameter_index]
                mean[class_index, parameter_index] = selected.mean()
                std = selected.std(unbiased=False)
                scale[class_index, parameter_index] = std.clamp_min(self.epsilon)
        self.mean = mean
        self.scale = scale
        return self

    def transform(self, targets):
        if self.mean is None or self.scale is None:
            raise RuntimeError("ParameterNormalizer must be fitted first")
        values = torch.as_tensor(targets).float()
        return (values - self.mean.to(values.device)) / self.scale.to(values.device)

    def inverse_transform(self, normalized):
        if self.mean is None or self.scale is None:
            raise RuntimeError("ParameterNormalizer must be fitted first")
        values = torch.as_tensor(normalized).float()
        return values * self.scale.to(values.device) + self.mean.to(values.device)


def masked_parameter_smooth_l1(prediction, target, labels, parameter_mask):
    predicted = prediction.float()
    expected = target.float()
    label = labels.float()
    mask = parameter_mask.bool()
    if predicted.shape != expected.shape or predicted.shape != mask.shape:
        raise ValueError("prediction, target and parameter_mask must match")
    if label.shape != predicted.shape[:2]:
        raise ValueError("labels must match prediction batch and class dimensions")
    valid = torch.logical_and(mask, label[:, :, None] > 0.5)
    if bool(torch.logical_and(valid, ~torch.isfinite(expected)).any()):
        raise ValueError("valid parameter targets must be finite")
    if not bool(valid.any()):
        return predicted.sum() * 0.0
    return F.smooth_l1_loss(predicted[valid], expected[valid], reduction="mean")


def masked_parameter_smooth_l1_by_target(
    prediction, target, labels, parameter_mask
):
    """Decompose the masked loss without changing its global normalization.

    The returned tensor has shape ``[C, K]`` and sums exactly to
    :func:`masked_parameter_smooth_l1`.  Keeping the original denominator is
    important: target-selective gradient routing must not silently reweight
    rare parameter targets.
    """

    predicted = prediction.float()
    expected = target.float()
    label = labels.float()
    mask = parameter_mask.bool()
    if predicted.shape != expected.shape or predicted.shape != mask.shape:
        raise ValueError("prediction, target and parameter_mask must match")
    if label.shape != predicted.shape[:2]:
        raise ValueError("labels must match prediction batch and class dimensions")
    valid = torch.logical_and(mask, label[:, :, None] > 0.5)
    if bool(torch.logical_and(valid, ~torch.isfinite(expected)).any()):
        raise ValueError("valid parameter targets must be finite")
    finite_expected = torch.where(valid, expected, torch.zeros_like(expected))
    elementwise = F.smooth_l1_loss(
        predicted, finite_expected, reduction="none"
    )
    denominator = valid.sum().clamp_min(1).to(elementwise.dtype)
    return (elementwise * valid.to(elementwise.dtype)).sum(dim=0) / denominator


def masked_parameter_consistency(student_prediction, teacher_prediction, valid_mask):
    student = student_prediction.float()
    teacher = teacher_prediction.detach().float()
    mask = valid_mask.float()
    if student.shape != teacher.shape or student.shape != mask.shape:
        raise ValueError("student, teacher and valid_mask must match")
    loss = F.smooth_l1_loss(student, teacher, reduction="none")
    denominator = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denominator
