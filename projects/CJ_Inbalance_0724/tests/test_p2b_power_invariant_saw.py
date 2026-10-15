import torch

from cj0724.p2b_power_invariant_saw import paired_saw_loss


def test_paired_saw_loss_has_zero_consistency_for_power_invariant_prediction():
    prediction = torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    target = torch.tensor([0.5, -0.5])
    total, supervised, consistency = paired_saw_loss(prediction, target, 0.0, 1.0, 0.05)
    assert torch.isclose(supervised, torch.tensor(0.0))
    assert torch.isclose(consistency, torch.tensor(0.0))
    assert torch.isclose(total, torch.tensor(0.0))


def test_paired_saw_loss_penalizes_power_dependent_prediction():
    prediction = torch.tensor([[0.0, 1.0, 2.0]])
    target = torch.tensor([1.0])
    total, supervised, consistency = paired_saw_loss(prediction, target, 0.0, 1.0, 0.5)
    assert consistency > 0
    assert total > supervised
