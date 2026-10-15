import torch
import pytest

from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.p3od_frozen_frequency_delay_parameter_residual import FrozenFrequencyDelayResidual
from cj0724.p3od_frozen_frequency_delay_parameter_residual import _class_nmae


def test_residual_side_branch_keeps_frozen_frequency_logits_bitwise_equal():
    torch.manual_seed(7)
    base = ParameterMatchedMultiView("F_time_frequency_matched").eval()
    side = FrozenFrequencyDelayResidual(base).eval()
    iq = torch.randn(3, 2, 2048)
    assert torch.equal(base(iq)[0], side(iq)[0])
    assert all(not parameter.requires_grad for parameter in side.base.parameters())
    assert any(parameter.requires_grad for parameter in side.delay.parameters())


def test_residual_side_branch_changes_only_parameter_path():
    torch.manual_seed(8)
    base = ParameterMatchedMultiView("F_time_frequency_matched").eval()
    side = FrozenFrequencyDelayResidual(base).eval()
    iq = torch.randn(2, 2, 2048)
    base_logits, base_parameters = base(iq)
    side_logits, side_parameters = side(iq)
    assert torch.equal(base_logits, side_logits)
    assert side_parameters.shape == base_parameters.shape == (2, 6, 2)


def test_class_nmae_uses_both_parameter_columns_of_one_class():
    record = {"parameter": {"per_target": [
        {"class_name": "MFT", "normalized_mae": 0.2},
        {"class_name": "MFT", "normalized_mae": 0.4},
        {"class_name": "ISRJ", "normalized_mae": 0.8},
    ]}}
    assert _class_nmae(record, "MFT") == pytest.approx(0.3)
