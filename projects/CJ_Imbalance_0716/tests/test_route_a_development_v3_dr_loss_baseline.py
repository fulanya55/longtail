import torch
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _dr_loss

def test_dr_loss_is_finite_per_label():
    logits=torch.tensor([[1.0,-1.0],[0.5,0.2]],requires_grad=True); labels=torch.tensor([[1.,0.],[0.,1.]])
    loss=_dr_loss(logits,labels)
    assert loss.shape == (2,) and torch.isfinite(loss).all()
    loss.mean().backward(); assert torch.isfinite(logits.grad).all()
