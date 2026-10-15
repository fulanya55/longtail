import torch

from experiments.run_route_a_development_v16_isrj_width_probe import WidthProbe


def test_width_probe_preserves_iq_to_scalar_gradient_contract():
    model = WidthProbe(width=8)
    signal = torch.randn(3, 2, 3000, requires_grad=True)
    prediction = model(signal)

    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert signal.grad is not None
    assert torch.isfinite(signal.grad).all()
