import numpy as np
import torch

from cj0724.four_class_composition import (
    LossWeights,
    FourClassMLCNN,
    UNSEEN_COMBINATION_FOLDS,
    combination_ids,
    make_composition_disjoint_roles,
    metric_summary,
    train_loss,
)


def synthetic_four_class_data():
    combinations = np.asarray(
        [[(combo >> bit) & 1 for bit in range(4)] for combo in range(1, 16)],
        dtype=np.uint8,
    )
    labels = np.repeat(combinations, 400, axis=0)
    indices = np.arange(len(labels), dtype=np.int64)
    return indices, labels


def test_composition_roles_are_identity_and_combination_disjoint():
    indices, labels = synthetic_four_class_data()
    roles = make_composition_disjoint_roles(
        indices, labels, fold=0, seed=7,
        train_per_combo=8, early_per_combo=4,
        calibration_per_combo=4, seen_confirm_per_combo=4,
        unseen_per_combo=10,
    )
    for left_index, left in enumerate(roles):
        for right in list(roles)[left_index + 1:]:
            assert np.intersect1d(roles[left], roles[right]).size == 0
    train_combos = set(combination_ids(labels[roles["train"]]))
    unseen_combos = set(combination_ids(labels[roles["unseen"]]))
    assert unseen_combos == set(UNSEEN_COMBINATION_FOLDS[0])
    assert train_combos.isdisjoint(unseen_combos)


def test_metric_summary_perfect_predictions():
    labels = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    probabilities = labels * 0.9 + (1 - labels) * 0.1
    metrics = metric_summary(probabilities, labels, np.full(4, 0.5))
    assert metrics["subset_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["hamming_loss"] == 0.0


def test_train_only_auxiliary_loss_is_finite():
    model = FourClassMLCNN("prototype_relation", head_dim=32)
    logits = torch.randn(6, 4, requires_grad=True)
    embeddings = torch.randn(6, 32, requires_grad=True)
    labels = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.float32)
    loss, parts = train_loss(model, logits, embeddings, labels, 2, LossWeights())
    assert torch.isfinite(loss)
    assert {"bce", "prototype", "relation", "hard_negative", "separation", "power_consistency"} <= set(parts)
    loss.backward()
