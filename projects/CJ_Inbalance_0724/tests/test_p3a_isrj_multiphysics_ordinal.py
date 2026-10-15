import numpy as np
import torch

from cj0724.p2w_isrj_width_oracle_ood import reconstruct_isrj_component
from cj0724.p3a_isrj_multiphysics_ordinal import decode_ordinal, extract_multiphysics_sequences, ordinal_loss


def test_multiphysics_sequence_shape_and_finiteness() -> None:
    iq = reconstruct_isrj_component(np.asarray([1.0e-6, 12.0e-6, 50.0e-6]), np.asarray([2, 3, 4]), np.asarray([0.0, 7.0e-6, 13.0e-6]))
    features = extract_multiphysics_sequences(iq)
    assert features.shape == (3, 5, 491)
    assert np.isfinite(features).all()


def test_ordinal_decode_and_loss_are_consistent() -> None:
    logits = torch.full((2, 490), -20.0)
    logits[0, :90] = 20.0
    logits[1, :390] = 20.0
    decoded = decode_ordinal(logits)
    assert torch.allclose(decoded, torch.tensor([100.0, 400.0]), atol=1.0e-3)
    targets = torch.tensor([100.0, 400.0])
    assert float(ordinal_loss(logits, targets)) < 1.0e-3
