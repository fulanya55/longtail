import numpy as np

from cj0724.b1_physics_sufficiency import _pool, best_threshold


def test_pool_and_threshold():
    sequence = np.arange(24, dtype=np.float32).reshape(2, 12)
    pooled = _pool(sequence, 3)
    assert pooled.shape == (2, 3)
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < best_threshold(labels, probabilities) <= 0.8
