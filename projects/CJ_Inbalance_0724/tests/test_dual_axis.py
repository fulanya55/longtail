import numpy as np
import torch

from cj0724.dual_axis import (
    build_balanced_manifest,
    build_identity_roles,
    build_target_label_lt_manifest,
)
from cj0724.losses import (
    AsymmetricLoss,
    DistributionBalancedLoss,
    PositiveReferenceDistillationLoss,
    SpecializedVFTResidualLoss,
    TargetHardNegativeLoss,
)


def synthetic_full_labels():
    combinations = np.asarray(
        [[int(bit) for bit in format(value, "06b")] for value in range(1, 64)],
        dtype=np.uint8,
    )
    return np.repeat(combinations, 400, axis=0)


def test_identity_roles_and_balanced_manifest_are_exact():
    labels = synthetic_full_labels()
    roles = build_identity_roles(labels, seed=20260724)
    assert {name: len(values) for name, values in roles.items()} == {
        "optimization": 17640,
        "early_stop": 2520,
        "calibration": 2520,
        "development": 2520,
    }
    assert len(np.unique(np.concatenate(list(roles.values())))) == 25200
    manifest = build_balanced_manifest(labels, roles["optimization"], 20260724)
    selected = np.asarray(manifest["selected_indices"])
    assert len(selected) == 2400
    assert manifest["label_counts"] == [1217] * 6
    assert len(manifest["combo_counts"]) == 63


def test_target_label_lt_manifest_keeps_all_combinations_and_exact_if3():
    labels = synthetic_full_labels()
    roles = build_identity_roles(labels, seed=20260724)
    manifest = build_target_label_lt_manifest(
        labels, roles["optimization"], target_label_index=1, seed=20260724
    )
    assert manifest["sample_count"] == 2400
    assert manifest["achieved_label_counts"] == [1200, 400, 1200, 1200, 1200, 1200]
    assert len(manifest["combo_counts"]) == 63
    assert min(manifest["combo_counts"].values()) >= 1


def test_distribution_balanced_loss_is_finite():
    criterion = DistributionBalancedLoss([1200, 400, 1200, 1200, 1200, 1200], 2400)
    logits = torch.randn(8, 6, requires_grad=True)
    targets = torch.randint(0, 2, (8, 6)).float()
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_asymmetric_loss_is_finite():
    criterion = AsymmetricLoss()
    logits = torch.randn(8, 6, requires_grad=True)
    targets = torch.randint(0, 2, (8, 6)).float()
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_target_hard_negative_loss_is_finite():
    criterion = TargetHardNegativeLoss(
        target_index=4,
        ohem_weight=0.5,
        negative_to_positive_ratio=2.0,
        ranking_weight=0.2,
        ranking_margin=0.5,
    )
    logits = torch.randn(16, 6, requires_grad=True)
    targets = torch.randint(0, 2, (16, 6)).float()
    targets[:4, 4] = 1.0
    targets[4:, 4] = 0.0
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_target_hard_negative_loss_penalizes_highest_negative_more():
    criterion = TargetHardNegativeLoss(
        target_index=4,
        ohem_weight=1.0,
        negative_to_positive_ratio=1.0,
    )
    targets = torch.zeros(4, 6)
    targets[0, 4] = 1.0
    low = torch.zeros(4, 6, requires_grad=True)
    high = low.detach().clone().requires_grad_(True)
    high.data[1, 4] = 4.0
    assert criterion(high, targets) > criterion(low, targets)


def test_specialized_vft_residual_loss_is_finite():
    criterion = SpecializedVFTResidualLoss()
    targets = torch.randint(0, 2, (16, 6)).float()
    targets[:4, 4] = 1.0
    targets[4:, 4] = 0.0
    components = {
        "logits": torch.randn(16, 6, requires_grad=True),
        "general_logits": torch.randn(16, 6, requires_grad=True),
        "weak_positive_score": torch.randn(16, requires_grad=True),
        "hard_negative_score": torch.randn(16, requires_grad=True),
    }
    loss = criterion(components, targets)
    loss.backward()
    assert torch.isfinite(loss)


def test_positive_reference_distillation_only_updates_weak_logits():
    criterion = PositiveReferenceDistillationLoss(
        target_index=4, distillation_weight=0.2, max_logit_delta=2.0
    )
    weak = torch.zeros(4, 6, requires_grad=True)
    reference = torch.zeros(4, 6, requires_grad=True)
    targets = torch.zeros(4, 6)
    targets[:2, 4] = 1.0
    reference.data[:2, 4] = 4.0
    loss = criterion(weak, reference, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert weak.grad is not None and torch.isfinite(weak.grad).all()
    assert reference.grad is None


def test_positive_reference_distillation_is_bounded():
    targets = torch.zeros(2, 6)
    targets[:, 4] = 1.0
    weak = torch.zeros(2, 6)
    moderate = torch.zeros(2, 6)
    extreme = torch.zeros(2, 6)
    moderate[:, 4] = 2.0
    extreme[:, 4] = 200.0
    criterion = PositiveReferenceDistillationLoss(max_logit_delta=2.0)
    assert torch.allclose(
        criterion(weak, moderate, targets), criterion(weak, extreme, targets)
    )
