import numpy as np
import torch

from cjlt.models.frequency_reconstruction import (
    FrozenLinearHead,
    build_pair_map,
    label_balanced_bce,
    normalized_anchor_loss,
    pair_ranking_loss,
)


LABELS = np.asarray(
    [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
    ],
    dtype=np.uint8,
)


def test_pair_maps_are_reproducible_and_legal():
    for mode in ("hard", "random"):
        first = build_pair_map(LABELS, 17, mode)
        second = build_pair_map(LABELS, 17, mode)
        assert np.array_equal(first, second)
        rows, columns = np.nonzero(LABELS == 1)
        assert np.all(LABELS[first[rows, columns], columns] == 0)


def test_label_balanced_bce_is_finite_and_differentiable():
    logits = torch.zeros((6, 3), requires_grad=True)
    targets = torch.as_tensor(LABELS, dtype=torch.float32)
    loss = label_balanced_bce(logits, targets, targets.sum(0), 6)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_rank_and_anchor_losses_are_finite():
    features = torch.randn(6, 4)
    head = FrozenLinearHead(torch.randn(3, 4), torch.randn(3))
    pair_map = torch.as_tensor(build_pair_map(LABELS, 23, "hard"))
    positions = torch.arange(6)
    targets = torch.as_tensor(LABELS, dtype=torch.float32)
    rank = pair_ranking_loss(head, features, positions, targets, pair_map)
    anchor = normalized_anchor_loss(
        head, head.linear.weight.detach().clone(), head.linear.bias.detach().clone()
    )
    assert torch.isfinite(rank)
    assert anchor.item() == 0.0
