import numpy as np
import torch

from cj0724.p3n_cross_interference_backbone import (
    conditional_margin_loss,
    make_identity_roles,
    variant_classification_loss,
)


def test_conditional_margin_ignores_coexistence_and_absence():
    logits = torch.zeros(4, 6, requires_grad=True)
    labels = torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    loss = conditional_margin_loss(logits, labels, margin=1.0)
    expected = torch.nn.functional.softplus(torch.tensor(1.0))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[2].abs().sum() == 0
    assert logits.grad[3].abs().sum() == 0


def test_hard_negative_loss_upweights_only_mft_without_isrj():
    logits = torch.zeros(2, 6)
    labels = torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1],
    ], dtype=torch.float32)
    baseline, _ = variant_classification_loss(logits, labels, "B1_original_bce")
    weighted, _ = variant_classification_loss(
        logits, labels, "B2_hard_negative", hard_negative_weight=3.0
    )
    # At zero logits every BCE element is equal, so normalized reweighting preserves the mean.
    assert torch.allclose(weighted, baseline)
    shifted = logits.clone(); shifted[0, 5] = 2.0
    baseline_shifted, _ = variant_classification_loss(shifted, labels, "B1_original_bce")
    weighted_shifted, _ = variant_classification_loss(
        shifted, labels, "B2_hard_negative", hard_negative_weight=3.0
    )
    assert weighted_shifted > baseline_shifted


def test_identity_roles_are_7_1_1_1_per_combination():
    combinations = []
    for value in range(1, 64):
        bits = [(value >> index) & 1 for index in range(6)]
        combinations.extend([bits] * 10)
    labels = np.asarray(combinations, dtype=np.uint8)
    roles = make_identity_roles(labels, 17)
    assert {name: len(indices) for name, indices in roles.items()} == {
        "fit": 441, "early": 63, "calibration": 63, "selection": 63
    }
    combined = np.concatenate(list(roles.values()))
    assert len(np.unique(combined)) == 630
    assert set(combined.tolist()) == set(range(630))
