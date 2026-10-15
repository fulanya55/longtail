import torch

from cj0724.losses import PairedSideRoutedResidualLoss, PairedWeakVisibilityResidualLoss
from cj0724.models import SideRoutedPhysicsResidualHead, WeakVisibilityResidualHead


def test_weak_visibility_residual_is_zero_initialized_and_bounded():
    model = WeakVisibilityResidualHead(width=4, hidden_dim=8, residual_bound=1.25)
    iq = torch.randn(3, 2, 128)
    base = torch.randn(3, 6)
    components = model.forward_components(iq, base)
    assert torch.allclose(components["logits"], base, atol=1e-6)
    assert float(components["residual"].detach().abs().max()) <= 1.25


def test_paired_visibility_loss_backpropagates_only_into_residual_head():
    model = WeakVisibilityResidualHead(width=4, hidden_dim=8)
    iq = torch.randn(4, 2, 128)
    base = torch.randn(4, 6, requires_grad=True)
    target = torch.tensor(
        [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0]],
        dtype=torch.float32,
    )
    weak = target.bool()
    criterion = PairedWeakVisibilityResidualLoss()
    loss, parts = criterion(model.forward_components(iq, base), target, weak, base + 1.0)
    loss.backward()
    assert base.grad is None
    assert model.head[-1].weight.grad is not None
    assert set(parts) == {"base", "weak", "ranking", "hard_negative", "residual"}


def test_side_routed_physics_head_is_zero_initialized_and_routes_three_sides():
    model = SideRoutedPhysicsResidualHead(width=2, hidden_dim=8, maximum_period_lag=64, maximum_delay_lag=96)
    iq = torch.randn(3, 2, 256)
    base = torch.randn(3, 6, requires_grad=True)
    components = model.forward_components(iq, base)
    assert torch.allclose(components["logits"], base, atol=1e-6)
    assert torch.allclose(components["route_probability"].sum(1), torch.ones(3), atol=1e-6)
    target = torch.randint(0, 2, (3, 6)).float()
    weak = target.bool()
    criterion = PairedSideRoutedResidualLoss(route_weight=0.3)
    loss, parts = criterion(components, target, weak, base + 0.5, torch.tensor([10, 30, 50]))
    loss.backward()
    assert base.grad is None
    assert model.router[-1].weight.grad is not None
    assert "route" in parts
