import numpy as np
import torch

from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import (
    _block_permutation, _block_shuffle, _joint_periodicity_features,
)


def test_joint_periodicity_feature_contract():
    signal = torch.randn(2, 2, 3000)
    features = _joint_periodicity_features(signal)
    assert features.shape == (2, 310)
    assert torch.isfinite(features).all()


def test_block_shuffle_preserves_local_blocks_and_iq_pairing():
    signal = np.arange(2 * 2 * 3000, dtype=np.float32).reshape(2, 2, 3000)
    permutation, digest = _block_permutation(120, 123)
    shuffled = _block_shuffle(signal, permutation, block_size=25)
    original_blocks = signal.reshape(2, 2, 120, 25)
    shuffled_blocks = shuffled.reshape(2, 2, 120, 25)
    np.testing.assert_array_equal(shuffled_blocks[:, :, 0], original_blocks[:, :, permutation[0]])
    assert len(digest) == 64

