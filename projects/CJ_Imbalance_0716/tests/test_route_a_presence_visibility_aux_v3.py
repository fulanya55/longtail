import torch

from experiments.run_route_a_presence_visibility_aux_v3 import PresenceVisibilityHead, _dual_loss


def test_dual_head_shapes_and_same_loss_path():
    model = PresenceVisibilityHead(8, 4)
    presence, visibility = model(torch.zeros(5, 8))
    assert presence.shape == (5, 3)
    assert visibility.shape == (5, 3)
    target = torch.tensor([[1,0,1]] * 5, dtype=torch.float32)
    gap = torch.zeros_like(target); mask = target.bool()
    loss, auxiliary = _dual_loss(presence, visibility, target, gap, mask, torch.ones(3), .25)
    loss.backward()
    assert torch.isfinite(loss)
    assert auxiliary >= 0

