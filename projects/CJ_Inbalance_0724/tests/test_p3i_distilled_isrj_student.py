import torch

from cj0724.p3i_distilled_isrj_student import DistilledISRJStudent, distillation_loss


def test_distilled_student_contract():
    model = DistilledISRJStudent(32)
    final, changed = model(torch.randn(4, 32), torch.randn(4))
    loss = distillation_loss(final, changed, torch.tensor([0., 1., 0., 1.]), torch.tensor([0.1, 0.9, 0.2, 0.8]), torch.tensor([0., 1., 0., 1.]))
    assert final.shape == changed.shape == (4,)
    assert torch.isfinite(loss)
