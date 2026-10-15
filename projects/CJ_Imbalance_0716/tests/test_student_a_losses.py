import numpy as np
import pytest
import torch

from cjlt.losses.classification import build_classification_loss

from cjlt.losses.prototype import label_prototype_contrastive_loss
from cjlt.losses.paired_power import (
    label_paired_power_consistency_loss,
    power_curriculum_max_gap,
)
from cjlt.models.label_conditioned import LabelConditionedFeatureBank
from cjlt.models.prototype_memory import (
    TailPrototypeMemory,
    prototype_diagnostics,
)


def test_label_conditioned_feature_bank_outputs_one_embedding_per_label():
    torch.manual_seed(2026)
    bank = LabelConditionedFeatureBank(
        feature_dim=32,
        num_labels=4,
        relation_dim=16,
        dropout=0.0,
    )
    features = torch.randn(5, 32)

    embeddings = bank(features)

    assert embeddings.shape == (5, 4, 16)
    assert torch.allclose(
        embeddings.norm(dim=-1),
        torch.ones(5, 4),
        atol=1e-5,
        rtol=1e-5,
    )
    assert bank.gates().shape == (4, 16)


def test_independent_label_conditioning_does_not_start_nearly_collinear():
    torch.manual_seed(2026)
    bank = LabelConditionedFeatureBank(
        feature_dim=32,
        num_labels=4,
        relation_dim=16,
        dropout=0.0,
        conditioning_mode="independent",
    )
    embeddings = bank(torch.randn(64, 32)).mean(dim=0)
    normalized = torch.nn.functional.normalize(embeddings, dim=-1)
    cosine = normalized @ normalized.t()
    off_diagonal = cosine[~torch.eye(4, dtype=torch.bool)]
    assert float(off_diagonal.max().detach()) < 0.90


def test_prototype_memory_updates_positive_and_negative_prototypes_without_gradients():
    torch.manual_seed(2026)
    memory = TailPrototypeMemory(num_labels=3, embedding_dim=8, momentum=0.9)
    embeddings = torch.nn.functional.normalize(torch.randn(6, 3, 8), dim=-1)
    labels = torch.tensor(
        [[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
        dtype=torch.float32,
    )

    memory.update(embeddings, labels)

    assert memory.positive_initialized.tolist() == [True, True, True]
    assert memory.negative_initialized.tolist() == [True, True, True]
    assert memory.positive_count.tolist() == [3, 3, 3]
    assert memory.negative_count.tolist() == [3, 3, 3]
    assert memory.positive_prototype.requires_grad is False
    assert memory.negative_prototype.requires_grad is False
    assert torch.allclose(
        memory.positive_prototype.norm(dim=-1),
        torch.ones(3),
        atol=1e-5,
    )


def test_label_prototype_contrastive_loss_is_finite_and_backpropagates():
    torch.manual_seed(2026)
    memory = TailPrototypeMemory(num_labels=3, embedding_dim=8, momentum=0.9)
    seed_embeddings = torch.nn.functional.normalize(torch.randn(8, 3, 8), dim=-1)
    seed_labels = torch.tensor(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )
    memory.update(seed_embeddings, seed_labels)
    embeddings = torch.nn.functional.normalize(
        torch.randn(4, 3, 8, requires_grad=True),
        dim=-1,
    )
    embeddings.retain_grad()
    labels = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
        dtype=torch.float32,
    )

    loss = label_prototype_contrastive_loss(
        embeddings,
        labels,
        memory,
        label_weights=[1.0, 2.0, 3.0],
        temperature=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_prototype_diagnostics_exposes_collapse_indicators():
    memory = TailPrototypeMemory(num_labels=2, embedding_dim=4, momentum=0.9)
    memory.positive_prototype[:] = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    memory.negative_prototype[:] = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    memory.positive_initialized[:] = True
    memory.negative_initialized[:] = True

    diagnostics = prototype_diagnostics(memory)

    assert diagnostics["positive_cosine_max_offdiag"] == pytest.approx(1.0)
    assert diagnostics["negative_cosine_max_offdiag"] == pytest.approx(1.0)
    assert diagnostics["positive_initialized"] == 2
    assert diagnostics["negative_initialized"] == 2


def test_paired_power_consistency_stops_gradient_on_stronger_view():
    reference = torch.nn.functional.normalize(
        torch.randn(2, 3, 8, requires_grad=True), dim=-1
    )
    paired = torch.nn.functional.normalize(
        torch.randn(2, 3, 8, requires_grad=True), dim=-1
    )
    reference.retain_grad()
    paired.retain_grad()
    labels = torch.ones(2, 3)
    stronger = torch.full((2, 3), 30, dtype=torch.int16)

    loss = label_paired_power_consistency_loss(
        reference,
        paired,
        labels,
        stronger_level_by_label=stronger,
        reference_level=30,
        paired_level=10,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert reference.grad is None or torch.allclose(
        reference.grad, torch.zeros_like(reference)
    )
    assert paired.grad is not None
    assert torch.isfinite(paired.grad).all()


def test_paired_power_consistency_is_directionally_symmetric_when_inputs_swap():
    reference = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    paired = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    labels = torch.ones(2, 3)
    stronger_reference = torch.full((2, 3), 30, dtype=torch.int16)
    stronger_paired = torch.full((2, 3), 30, dtype=torch.int16)

    forward = label_paired_power_consistency_loss(
        reference,
        paired,
        labels,
        stronger_reference,
        reference_level=30,
        paired_level=10,
    )
    reverse = label_paired_power_consistency_loss(
        paired,
        reference,
        labels,
        stronger_paired,
        reference_level=10,
        paired_level=30,
    )

    assert forward.item() == pytest.approx(reverse.item(), rel=1e-6)


def test_paired_power_consistency_returns_zero_without_directional_labels():
    reference = torch.randn(2, 3, 8, requires_grad=True)
    paired = torch.randn(2, 3, 8, requires_grad=True)
    labels = torch.ones(2, 3)
    stronger = torch.zeros(2, 3, dtype=torch.int16)

    loss = label_paired_power_consistency_loss(
        reference,
        paired,
        labels,
        stronger,
        reference_level=30,
        paired_level=30,
    )

    assert loss.item() == pytest.approx(0.0)


def test_paired_power_consistency_supports_mixed_levels_inside_one_batch():
    reference = torch.nn.functional.normalize(
        torch.randn(2, 2, 8, requires_grad=True), dim=-1
    )
    paired = torch.nn.functional.normalize(
        torch.randn(2, 2, 8, requires_grad=True), dim=-1
    )
    labels = torch.ones(2, 2)
    stronger = torch.tensor([[30, 10], [35, 30]], dtype=torch.int16)

    loss = label_paired_power_consistency_loss(
        reference,
        paired,
        labels,
        stronger,
        reference_level=torch.tensor([30, 30]),
        paired_level=torch.tensor([10, 35]),
    )
    loss.backward()

    assert torch.isfinite(loss)


def test_power_curriculum_expands_gap_by_training_progress():
    assert power_curriculum_max_gap(0.0) == 5.0
    assert power_curriculum_max_gap(0.2) == 10.0
    assert power_curriculum_max_gap(0.5) == 15.0
    assert power_curriculum_max_gap(0.8) == float("inf")


def test_distribution_balanced_loss_handles_cooccurring_tail_labels():
    loss_fn = build_classification_loss(
        "db",
        positive_counts=[100, 60, 25, 10],
        sample_count=200,
    )
    logits = torch.randn(5, 4, requires_grad=True)
    targets = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    loss = loss_fn(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
