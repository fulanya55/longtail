import numpy as np
import torch

from cj0724.shared_physics_frontend import shared_feature_batch


def test_shared_frontend_shapes_and_finiteness():
    iq = torch.randn(3, 2, 3000)
    output = shared_feature_batch(iq)
    assert output["mft_delay"].shape == (3, 190)
    assert output["isrj_periodicity"].shape == (3, 310)
    assert output["combined"].shape == (3, 320)
    assert all(np.isfinite(value.numpy()).all() for value in output.values())
