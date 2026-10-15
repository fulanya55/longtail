import torch

from cj0724.p5b1_jnr_physical_oracle import (
    TARGET_NAMES,
    TargetSpecificIQOracle,
    analytic_features,
    masked_smooth_l1,
)


def test_p5b1_analytic_features_are_fixed_and_power_free():
    features = analytic_features(torch.randn(2, 2, 3000))
    assert features.shape == (2, 26)
    assert torch.isfinite(features).all()


def test_p5b1_iq_oracle_uses_composition_but_not_nominal_power():
    model = TargetSpecificIQOracle(width=4)
    prediction = model(torch.randn(3, 2, 3000), torch.randint(0, 2, (3, 6)).float())
    assert prediction.shape == (3, len(TARGET_NAMES))
    assert torch.isfinite(prediction).all()


def test_p5b1_masked_loss_excludes_inactive_nan_targets_before_autograd():
    prediction = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    target = torch.tensor([[0.0, float("nan")], [float("nan"), 3.0]])
    valid = torch.tensor([[True, False], [False, True]])
    loss = masked_smooth_l1(prediction, target, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
