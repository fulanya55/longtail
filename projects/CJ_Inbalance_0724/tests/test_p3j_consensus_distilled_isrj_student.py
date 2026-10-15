import torch

from cj0724.p3j_consensus_distilled_isrj_student import consensus_loss


def test_consensus_loss_is_finite():
    loss = consensus_loss(torch.randn(5), torch.randn(5), torch.tensor([0., 1., 0., 1., 1.]), torch.rand(5), torch.tensor([0., 1., 0., 1., 0.]), torch.ones(5))
    assert torch.isfinite(loss)
