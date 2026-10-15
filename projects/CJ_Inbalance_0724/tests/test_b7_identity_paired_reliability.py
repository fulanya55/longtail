import numpy as np
import torch

from cj0724.b7_identity_paired_reliability import (
    BoundedReliabilityResidual,
    label_preserving_unpaired_teacher,
    reliability_loss,
)


def test_b7_residual_is_zero_initialized_bounded_and_teacher_free_at_inference():
    model = BoundedReliabilityResidual(feature_dim=4, hidden_dim=8, residual_bound=1.25)
    features = torch.randn(5, 4)
    reference = torch.linspace(0.1, 0.9, 5)
    output, residual = model(features, reference)
    assert torch.allclose(torch.sigmoid(output), reference, atol=1e-6)
    assert float(residual.detach().abs().max()) <= 1.25


def test_b7_unpaired_teacher_breaks_identity_but_preserves_label_distributions():
    teacher = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    shuffled = label_preserving_unpaired_teacher(teacher, labels, seed=42)
    assert not np.array_equal(shuffled, teacher)
    assert np.array_equal(np.sort(shuffled[labels == 0]), np.sort(teacher[labels == 0]))
    assert np.array_equal(np.sort(shuffled[labels == 1]), np.sort(teacher[labels == 1]))


def test_b7_pair_type_loss_uses_only_reliable_positive_teacher_pairs():
    output = torch.zeros(4, requires_grad=True)
    residual = torch.zeros(4, requires_grad=True)
    reference = torch.full((4,), 0.2)
    teacher = torch.tensor([0.9, 0.1, 0.9, 0.9])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    hard_negative = torch.tensor([0.0, 0.0, 1.0, 0.0])
    loss, parts = reliability_loss(
        "pair_type_aware_reliability", output, residual, reference, teacher,
        labels, hard_negative, teacher_threshold=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert int(parts["eligible_count"]) == 1
    assert output.grad is not None


def test_b7_all_pair_invariance_distills_negatives_too():
    output = torch.zeros(4, requires_grad=True)
    residual = torch.zeros(4, requires_grad=True)
    reference = torch.full((4,), 0.2)
    teacher = torch.tensor([0.9, 0.1, 0.9, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    hard_negative = torch.zeros(4)
    _, parts = reliability_loss(
        "all_pair_invariance", output, residual, reference, teacher,
        labels, hard_negative, teacher_threshold=0.5,
    )
    assert int(parts["eligible_count"]) == 4
