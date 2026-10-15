import torch

from cj0724.p3k_raw_iq_advantage_student import RawIQAdvantageStudent, advantage_loss


def test_raw_iq_advantage_student_shapes_and_finite_loss():
    model = RawIQAdvantageStudent(14)
    iq = torch.randn(3, 2, 3000)
    context = torch.randn(3, 14)
    main_logit = torch.randn(3)
    final, helpful = model(iq, context, main_logit)
    loss = advantage_loss(final, helpful, torch.tensor([0.0, 1.0, 1.0]), torch.tensor([0.1, 0.9, 0.7]), torch.sigmoid(main_logit))
    assert final.shape == (3,)
    assert helpful.shape == (3,)
    assert torch.isfinite(loss)
