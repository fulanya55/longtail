import numpy as np
import torch

from cj0724.p8b_axis_stratified_longtail import ReplacementBatchSampler
from cj0724.p8c_vft_hard_negative_logit_cap import VFT, hard_negative_logit_cap_loss


def _labels():
    return torch.tensor([[0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]], dtype=torch.float32)


def test_cap_applies_only_to_pre_registered_vft_hard_negative_rows():
    labels = _labels(); logits = torch.zeros((3, 6), dtype=torch.float32, requires_grad=True)
    logits.data[0, VFT] = 2.0; logits.data[1, VFT] = 9.0; logits.data[2, VFT] = 9.0
    loss, cap, count = hard_negative_logit_cap_loss(logits, labels, 0.05, 0.0)
    assert count == 1
    assert torch.isclose(cap, torch.tensor(4.0))
    loss.backward()
    assert logits.grad[0, VFT] > 0
    expected_bce_only = (torch.sigmoid(logits.detach()[1, VFT]) - 1.0) / logits.numel()
    assert torch.isclose(logits.grad[1, VFT], expected_bce_only, atol=1e-6)


def test_cap_is_zero_below_limit_or_without_matching_rows():
    labels = _labels(); logits = torch.full((3, 6), -2.0)
    _, cap, count = hard_negative_logit_cap_loss(logits, labels, 0.05, 0.0)
    assert count == 1 and cap.item() == 0.0
    no_hard = labels.clone(); no_hard[0, VFT] = 1.0
    _, cap_none, count_none = hard_negative_logit_cap_loss(logits, no_hard, 0.05, 0.0)
    assert count_none == 0 and cap_none.item() == 0.0


def test_c0_and_c1_can_share_identical_random_batch_sequence():
    c0 = ReplacementBatchSampler(2400, 64, 38, 18201); c1 = ReplacementBatchSampler(2400, 64, 38, 18201)
    assert list(c0) == list(c1)
