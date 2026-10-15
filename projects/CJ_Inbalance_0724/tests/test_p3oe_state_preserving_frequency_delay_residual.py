import torch

from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.p3oe_state_preserving_frequency_delay_residual import StatePreservingFrequencyDelayResidual, _f_buffers, _immutable_f_audit


def test_p3oe_train_mode_keeps_all_f_buffers_and_logits_bitwise_fixed_after_optimizer_step():
    torch.manual_seed(2051)
    base = ParameterMatchedMultiView("F_time_frequency_matched").eval()
    side = StatePreservingFrequencyDelayResidual(base)
    iq = torch.randn(3, 2, 2048)
    assert _immutable_f_audit(base, side, iq)
    before = _f_buffers(base); logits = base(iq)[0].detach().clone()
    optimizer = torch.optim.AdamW((parameter for parameter in side.parameters() if parameter.requires_grad), lr=1e-4)
    side.train(); _, prediction = side(iq); prediction.square().mean().backward(); optimizer.step()
    assert not base.training and not any(module.training for module in base.modules())
    assert all(torch.equal(before[name], value) for name, value in base.named_buffers())
    assert torch.equal(logits, base(iq)[0]) and torch.equal(logits, side(iq)[0])
