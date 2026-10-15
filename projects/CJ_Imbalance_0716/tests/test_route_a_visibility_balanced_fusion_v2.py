import torch

from experiments.run_route_a_visibility_balanced_fusion_v2 import (
    VisibilityBalancedFusion,
    _visibility_balanced_loss,
)


def test_visibility_fusion_is_bounded_and_loss_is_finite():
    model = VisibilityBalancedFusion(residual_cap=0.5, uncertainty_power=0.5)
    base = torch.randn(12, 6)
    expert = torch.randn(12, 3)
    targets = torch.randint(0, 2, (12, 6)).float()
    weak = torch.zeros(12, 6, dtype=torch.uint8)
    weak[:3, 3:] = targets[:3, 3:].to(torch.uint8)
    logits, residual = model(base, expert, return_residual=True)
    loss = _visibility_balanced_loss(logits, targets, weak, residual,
                                     model.bias, 0.02, 0.05)
    assert torch.isfinite(loss)
    assert float(residual.detach().abs().max()) <= 0.500001
    assert torch.allclose(residual[:, :3], torch.zeros_like(residual[:, :3]))

