import torch

from cj0724.p3oh_finite_masked_combo_cvar import finite_valid_per_sample_parameter_loss


def test_inactive_nan_targets_do_not_enter_loss_gradient_or_adamw_step():
    parameter = torch.nn.Parameter(torch.randn(2, 6, 2))
    target = torch.full((2, 6, 2), float("nan"))
    labels = torch.zeros(2, 6)
    mask = torch.zeros(2, 6, 2, dtype=torch.bool)
    labels[0, 0] = 1; labels[1, 3] = 1
    mask[0, 0] = True; mask[1, 3] = True
    target[0, 0] = torch.tensor([0.5, -0.5]); target[1, 3] = torch.tensor([1.0, 2.0])
    loss = finite_valid_per_sample_parameter_loss(parameter, target, labels, mask).mean()
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert torch.isfinite(loss)
    assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    assert bool(torch.isfinite(parameter).all())


def test_nonfinite_target_in_a_valid_position_is_rejected():
    prediction = torch.zeros(1, 6, 2)
    target = torch.full((1, 6, 2), float("nan"))
    labels = torch.zeros(1, 6); labels[0, 1] = 1
    mask = torch.zeros(1, 6, 2, dtype=torch.bool); mask[0, 1, 0] = True
    try:
        finite_valid_per_sample_parameter_loss(prediction, target, labels, mask)
    except ValueError as error:
        assert "valid parameter targets" in str(error)
    else:
        raise AssertionError("a valid nonfinite target must be rejected")
