import torch

from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.p3oe_invariant_frequency_delay_parameter_residual import InvariantFrozenFrequencyDelayResidual


def test_train_mode_keeps_base_state_and_logits_bitwise_invariant():
    torch.manual_seed(13)
    base = ParameterMatchedMultiView("F_time_frequency_matched")
    side = InvariantFrozenFrequencyDelayResidual(base)
    iq = torch.randn(3, 2, 2048)
    before = {key: value.detach().clone() for key, value in side.base.state_dict().items()}
    logits_before = side.base.eval()(iq)[0]
    side.train()
    logits_after = side(iq)[0]
    assert side.base.training is False
    assert torch.equal(logits_before, logits_after)
    assert all(torch.equal(value, side.base.state_dict()[key]) for key, value in before.items())
