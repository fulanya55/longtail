import pytest
import torch

from cjlt.losses.parameter import (
    ParameterNormalizer,
    masked_parameter_smooth_l1,
    select_parameter_targets,
)
from cjlt.losses.pseudo_label import (
    DynamicPseudoLabelThresholds,
    build_pseudo_labels,
    masked_pseudo_label_bce,
    prototype_reliability,
)
from cjlt.models.ema_teacher import EMATeacher, ema_momentum
from cjlt.models.parameter_heads import LabelGatedParameterHeads


def test_ema_teacher_is_frozen_and_tracks_student_parameters():
    student = torch.nn.Linear(4, 2)
    teacher = EMATeacher(student)
    before = teacher.model.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(1.0)

    momentum = teacher.update(student, progress=0.5)

    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())
    assert 0.99 < momentum < 0.999
    assert not torch.allclose(before, teacher.model.weight)
    assert ema_momentum(0.0) == 0.99
    assert ema_momentum(1.0) == 0.999


def test_per_label_pseudo_labels_keep_uncertain_positions_masked():
    probability = torch.tensor([[0.98, 0.02, 0.55], [0.80, 0.10, 0.99]])
    reliability = torch.tensor([[0.9, 0.9, 0.9], [0.2, 0.9, 0.8]])
    stability = torch.tensor([[0.01, 0.01, 0.01], [0.01, 0.01, 0.20]])

    pseudo = build_pseudo_labels(
        probability,
        tau_pos=[0.95, 0.95, 0.95],
        tau_neg=[0.05, 0.05, 0.05],
        reliability=reliability,
        reliability_threshold=0.5,
        stability=stability,
        stability_ceiling=0.10,
    )

    assert pseudo.positive_mask.tolist() == [
        [True, False, False],
        [False, False, False],
    ]
    assert pseudo.negative_mask.tolist() == [
        [False, True, False],
        [False, False, False],
    ]
    assert pseudo.accepted_mask.tolist() == [
        [True, True, False],
        [False, False, False],
    ]
    logits = torch.zeros_like(probability, requires_grad=True)
    loss = masked_pseudo_label_bce(logits, pseudo.targets, pseudo.accepted_mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_prototype_reliability_uses_positive_negative_distance_margin():
    embedding = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0]]]), dim=-1)
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[-1.0, 0.0]])

    score = prototype_reliability(embedding, positive, negative)

    assert score.shape == (1, 1)
    assert score.item() > 0.8


def test_dynamic_thresholds_follow_precision_coverage_rules_without_crossing_bounds():
    controller = DynamicPseudoLabelThresholds(
        num_labels=2,
        tau_pos=0.95,
        tau_neg=0.05,
        tau_pos_floor=0.70,
        tau_neg_ceiling=0.20,
    )

    controller.update(
        positive_precision=[0.99, 0.70],
        positive_coverage=[0.01, 0.50],
        target_precision=0.95,
        minimum_coverage=0.10,
    )

    assert controller.tau_pos.tolist() == pytest.approx([0.94, 0.97])
    assert all(controller.tau_pos >= 0.70)
    assert all(controller.tau_neg <= 0.20)


def test_dynamic_thresholds_accept_per_label_initial_values():
    controller = DynamicPseudoLabelThresholds(
        num_labels=2,
        tau_pos=[0.8, 0.9],
        tau_neg=[0.2, 0.1],
    )
    assert controller.tau_pos.tolist() == pytest.approx([0.8, 0.9])
    assert controller.tau_neg.tolist() == pytest.approx([0.2, 0.1])


def test_label_gated_parameter_heads_and_masked_loss_use_only_valid_positive_targets():
    torch.manual_seed(2026)
    heads = LabelGatedParameterHeads(
        num_labels=2,
        embedding_dim=8,
        parameters_per_label=2,
    )
    embeddings = torch.randn(3, 2, 8)
    prediction = heads(embeddings)
    values79 = torch.zeros(3, 79)
    mask79 = torch.zeros(3, 79, dtype=torch.bool)
    values79[:, [0, 2, 16, 18]] = torch.tensor(
        [[10.0, 100.0, 20.0, 200.0], [30.0, 300.0, 40.0, 400.0], [50.0, 500.0, 60.0, 600.0]]
    )
    mask79[:, [0, 2, 16, 18]] = True
    target, target_mask = select_parameter_targets(
        values79, mask79, class_names=["FMB", "SinFM"]
    )
    labels = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    normalizer = ParameterNormalizer().fit(target, target_mask & (labels[:, :, None] > 0.5))
    normalized = normalizer.transform(target)

    loss = masked_parameter_smooth_l1(
        prediction,
        normalized,
        labels,
        target_mask,
    )

    assert prediction.shape == (3, 2, 2)
    assert target.shape == (3, 2, 2)
    assert torch.isfinite(loss)


def test_masked_parameter_loss_ignores_nan_values_outside_valid_mask():
    prediction = torch.zeros(2, 1, 2, requires_grad=True)
    target = torch.tensor([[[1.0, float("nan")]], [[float("nan"), 2.0]]])
    labels = torch.tensor([[1.0], [1.0]])
    mask = torch.tensor([[[True, False]], [[False, True]]])

    loss = masked_parameter_smooth_l1(prediction, target, labels, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_balanced_pseudo_bce_gives_positive_and_negative_sides_equal_weight():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    loss = masked_pseudo_label_bce(
        logits, targets, mask, balance_sides=True
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert abs(float(logits.grad[0, 0])) == pytest.approx(
        abs(float(logits.grad[0, 1:].sum()))
    )


def test_pseudo_label_side_caps_are_applied_independently_per_label():
    probability = torch.tensor(
        [
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.95, 0.05],
            [0.94, 0.06],
            [0.93, 0.07],
            [0.92, 0.08],
            [0.91, 0.09],
            [0.90, 0.10],
        ]
    )
    pseudo = build_pseudo_labels(
        probability,
        tau_pos=[0.5, 0.5],
        tau_neg=[0.5, 0.5],
        max_positive_fraction=0.30,
        max_negative_fraction=0.30,
    )
    assert pseudo.positive_mask.sum(dim=0).tolist() == [3, 0]
    assert pseudo.negative_mask.sum(dim=0).tolist() == [0, 3]
    assert pseudo.positive_mask[:3, 0].all()
    assert pseudo.negative_mask[:3, 1].all()
