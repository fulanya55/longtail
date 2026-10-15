import numpy as np
import torch

from cj0724.p2s_acf_consistent_separator import normalized_complex_autocorrelation


def test_normalized_autocorrelation_shape_and_origin() -> None:
    iq = torch.zeros(2, 2, 4096)
    iq[:, 0, ::100] = 1.0
    result = normalized_complex_autocorrelation(iq)
    assert result.shape == (2, 391)
    assert torch.isfinite(result).all()


def test_normalized_autocorrelation_rejects_bad_shape() -> None:
    try:
        normalized_complex_autocorrelation(torch.zeros(2, 4096))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
