import numpy as np
import torch

from cj0724.p3om_sparse_peak_ordinal import (
    OrdinalTokenNet,
    decode_ordinal,
    fixed_bin_tokens,
    ordinal_loss,
    sparse_peak_tokens,
)


def _channels():
    result = np.zeros((1, 4, 32), dtype=np.float32)
    result[:, 0] = np.linspace(0.0, 1.0, 32)
    result[:, 1, [3, 9, 21]] = [0.5, 1.0, 0.75]
    result[:, 2] = 0.25
    result[:, 3] = np.linspace(0.0, 1.0, 32)
    return result


def test_p3om_equal_token_contract_and_sparse_time_order():
    channels = _channels()
    control = fixed_bin_tokens(channels)
    candidate = sparse_peak_tokens(channels)
    assert control.shape == candidate.shape == (1, 16, 4)
    active = candidate[0, candidate[0, :, 1] > 0]
    assert active.shape[0] == 3
    assert np.all(np.diff(active[:, 3]) > 0)
    assert np.all(candidate[0, 3:] == 0.0)


def test_p3om_ordinal_head_has_same_contract_for_both_tokenizations():
    model = OrdinalTokenNet()
    control = torch.from_numpy(fixed_bin_tokens(_channels()))
    candidate = torch.from_numpy(sparse_peak_tokens(_channels()))
    logits0, logits1 = model(control), model(candidate)
    assert logits0.shape == logits1.shape == (1, 7)
    loss = ordinal_loss(logits1, torch.tensor([4.0]))
    assert torch.isfinite(loss)
    prediction = decode_ordinal(logits1)
    assert bool(torch.all((prediction >= 1.0) & (prediction <= 8.0)))


def test_p3om_candidate_and_control_use_identical_parameter_count():
    assert sum(value.numel() for value in OrdinalTokenNet().parameters()) == sum(
        value.numel() for value in OrdinalTokenNet().parameters()
    )
