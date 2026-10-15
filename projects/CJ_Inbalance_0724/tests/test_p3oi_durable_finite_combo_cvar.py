import torch

from cj0724.p3oi_durable_finite_combo_cvar_seed import finite_valid_per_sample_parameter_loss


def test_p3oi_valid_indexing_keeps_inactive_nan_out_of_loss():
    prediction = torch.nn.Parameter(torch.ones(1, 6, 2)); target = torch.full((1, 6, 2), float("nan"))
    label = torch.zeros(1, 6); mask = torch.zeros(1, 6, 2, dtype=torch.bool)
    label[0, 4] = 1; mask[0, 4] = True; target[0, 4] = torch.tensor([2.0, 3.0])
    loss = finite_valid_per_sample_parameter_loss(prediction, target, label, mask).mean(); loss.backward()
    assert torch.isfinite(loss) and bool(torch.isfinite(prediction.grad).all())
