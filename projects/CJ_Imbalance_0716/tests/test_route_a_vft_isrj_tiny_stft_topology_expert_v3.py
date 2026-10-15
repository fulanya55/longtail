import numpy as np
import torch

from experiments.run_route_a_vft_isrj_tiny_stft_topology_expert_v3 import (
    TinySTFTTopologyExpert, _apply_time_permutation, _permutation, _stft_images,
)


def test_stft_images_and_expert_shapes_are_finite():
    signal = torch.randn(3, 2, 512)
    image = _stft_images(signal, grid_size=32)
    assert image.shape == (3, 1, 32, 32)
    model = TinySTFTTopologyExpert()
    score = model(image)
    assert score.shape == (3,)
    assert torch.isfinite(score).all()


def test_time_permutation_preserves_values_and_changes_order():
    image = np.arange(2 * 1 * 4 * 4).reshape(2, 1, 4, 4)
    permutation, digest = _permutation(4, 123)
    shuffled = _apply_time_permutation(image, permutation)
    np.testing.assert_array_equal(np.sort(shuffled, axis=3), np.sort(image, axis=3))
    assert len(digest) == 64

