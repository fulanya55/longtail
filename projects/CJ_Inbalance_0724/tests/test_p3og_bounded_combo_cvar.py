import torch

from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.p3og_bounded_combo_cvar import BoundedStatePreservingResidual, compound_cvar_loss


def test_train_mode_preserves_all_base_state_and_logits():
    torch.manual_seed(17)
    base = ParameterMatchedMultiView("F_time_frequency_matched")
    side = BoundedStatePreservingResidual(base, residual_bound=0.25)
    iq = torch.randn(3, 2, 2048)
    before = {key: value.detach().clone() for key, value in base.state_dict().items()}
    logits_before = base.eval()(iq)[0]
    side.train()
    logits_after = side(iq)[0]
    assert base.training is False
    assert torch.equal(logits_before, logits_after)
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in before.items())


def test_compound_cvar_uses_upper_tail_of_equal_group_means():
    per_sample = torch.tensor([1.0, 3.0, 9.0, 5.0])
    labels = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 0.]])
    # Group means are 2, 9, 5. ceil(25% * 3)=1, so the upper tail is 9.
    assert torch.isclose(compound_cvar_loss(per_sample, labels, 0.25), torch.tensor(9.0))
