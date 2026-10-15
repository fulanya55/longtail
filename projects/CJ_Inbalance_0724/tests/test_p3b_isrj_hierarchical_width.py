import numpy as np
import torch

from cj0724.p3b_isrj_hierarchical_width import decode_hierarchical, hierarchical_loss, width_band_numpy


def test_width_band_boundaries() -> None:
    samples = np.asarray([10, 19, 20, 49, 50, 199, 200, 299, 300, 500])
    assert width_band_numpy(samples).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_hierarchical_decode_and_loss() -> None:
    band_logits = torch.full((2, 5), -10.0)
    band_logits[0, 1] = 10.0
    band_logits[1, 4] = 10.0
    local_logits = torch.zeros((2, 5))
    prediction, band = decode_hierarchical(band_logits, local_logits)
    assert band.tolist() == [1, 4]
    assert torch.allclose(prediction, torch.tensor([34.5, 400.0]))
    assert float(hierarchical_loss(band_logits, local_logits, prediction)) < 1.0e-3
