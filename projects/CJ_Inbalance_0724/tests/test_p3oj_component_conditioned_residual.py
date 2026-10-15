import torch
from torch import nn
from cj0724.p3oj_component_conditioned_residual import ComponentConditionedResidual
from cj0724.p3oj_component_conditioned_residual_seed import finite_valid_per_sample_parameter_loss

class _Base(nn.Module):
    def __init__(self): super().__init__(); self.w=nn.Parameter(torch.ones(1))
    def forward_with_embedding(self, x):
        b=x.shape[0]; return torch.zeros(b,6), torch.zeros(b,6,2), torch.zeros(b,64)

def test_p3oj_detached_posterior_preserves_frozen_base_state_and_logits():
    model=ComponentConditionedResidual(_Base()).train(); x=torch.randn(2,2,3000); before={k:v.clone() for k,v in model.base.state_dict().items()}
    logits, _, _=model.forward_with_embedding(x)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert all(torch.equal(v,model.base.state_dict()[k]) for k,v in before.items())
    assert not model.base.training


def test_component_toggle_only_changes_component_gradient_contract():
    model = ComponentConditionedResidual(_Base())
    model.set_component_enabled(False); assert not model.component.requires_grad
    model.set_component_enabled(True); assert model.component.requires_grad


def test_p3oj_component_is_disabled_for_j0_and_receives_finite_gradient_for_j1():
    model = ComponentConditionedResidual(_Base()).train()
    model.set_component_enabled(False)
    _, j0_parameters, _ = model.forward_with_embedding(torch.randn(2, 2, 3000))
    assert not model.component.requires_grad and torch.equal(model.component, torch.zeros_like(model.component))
    model.set_component_enabled(True)
    _, j1_parameters, _ = model.forward_with_embedding(torch.randn(2, 2, 3000))
    j1_parameters.square().mean().backward()
    assert model.component.requires_grad and model.component.grad is not None
    assert torch.isfinite(model.component.grad).all()
    assert torch.isfinite(j0_parameters).all()


def test_p3oj_finite_loss_excludes_inactive_nan_targets():
    prediction = torch.randn(2, 6, 2, requires_grad=True)
    target = torch.full((2, 6, 2), float("nan")); target[:, 0] = 0.25
    labels = torch.zeros(2, 6); labels[:, 0] = 1
    mask = torch.zeros(2, 6, 2); mask[:, 0] = 1
    loss = finite_valid_per_sample_parameter_loss(prediction, target, labels, mask).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
