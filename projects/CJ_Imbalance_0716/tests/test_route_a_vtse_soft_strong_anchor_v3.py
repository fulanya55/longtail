import torch

from experiments.run_route_a_vtse_soft_strong_anchor_v3 import _soft_anchor_loss


def test_soft_anchor_loss_is_finite_and_uses_all_three_strata():
    logits = torch.zeros(6, 3, requires_grad=True)
    targets = torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ], dtype=torch.float32)
    weak = torch.zeros_like(targets, dtype=torch.uint8)
    weak[0, 3] = 1; weak[2, 4] = 1; weak[4, 5] = 1

    loss = _soft_anchor_loss(logits, targets, weak, strong_weight=0.2)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
