import numpy as np
import torch

from cj0724.p3c_mft_sparse_deconvolution import reconstruct_mft_component
from cj0724.p3d_mft_inverse_ordinal import (
    InverseOrdinalNet, decode_ordinal, extract_inverse_delay_sequences, ordinal_loss,
)


def test_inverse_delay_feature_shape_and_finiteness():
    delays = np.full((2, 8), np.nan)
    amplitudes = np.full((2, 8), np.nan)
    delays[0, :2], amplitudes[0, :2] = [40e-6, 50e-6], [1.0, 0.8]
    delays[1, :4], amplitudes[1, :4] = [40e-6, 40.1e-6, 45e-6, 50e-6], 1.0
    features = extract_inverse_delay_sequences(reconstruct_mft_component(delays, amplitudes))
    assert features.shape == (2, 4, 751)
    assert np.isfinite(features).all()


def test_ordinal_network_contract():
    model = InverseOrdinalNet()
    logits = model(torch.randn(3, 4, 751))
    target = torch.tensor([1.0, 4.0, 8.0])
    assert logits.shape == (3, 7)
    assert torch.isfinite(ordinal_loss(logits, target))
    prediction = decode_ordinal(logits)
    assert torch.all((prediction >= 1.0) & (prediction <= 8.0))
