import torch

from experiments.run_route_a_bounded_local_contrast_v2 import (
    weak_hard_negative_ranking_loss,
)


def test_weak_ranking_prefers_weak_positive_above_hard_negative():
    targets = torch.tensor([[1.0], [0.0], [0.0]])
    weak = torch.tensor([[1], [0], [0]], dtype=torch.uint8)
    bad = torch.tensor([[-1.0], [1.0], [0.5]], requires_grad=True)
    good = torch.tensor([[2.0], [0.0], [-0.5]])
    bad_loss = weak_hard_negative_ranking_loss(bad, targets, weak)
    good_loss = weak_hard_negative_ranking_loss(good, targets, weak)
    assert good_loss < bad_loss
    bad_loss.backward()
    assert bad.grad[0, 0] < 0

