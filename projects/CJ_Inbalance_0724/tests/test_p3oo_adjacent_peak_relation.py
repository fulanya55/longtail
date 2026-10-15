import numpy as np
import torch

from cj0724.p3oo_adjacent_peak_relation import PairTokenOrdinalNet, adjacent_peak_relation_tokens


def _channels():
    channels = np.zeros((1, 4, 32), dtype=np.float32)
    channels[:, 0] = np.linspace(0.0, 1.0, 32); channels[:, 2] = 0.25; channels[:, 3] = np.linspace(0.0, 1.0, 32)
    channels[:, 1, [3, 9, 21]] = [0.5, 1.0, 0.75]
    return channels


def test_p3oo_controls_capacity_and_relation_visibility():
    zero = adjacent_peak_relation_tokens(_channels(), False)
    relation = adjacent_peak_relation_tokens(_channels(), True)
    assert zero.shape == relation.shape == (1, 16, 8)
    assert np.array_equal(zero[:, :, :4], relation[:, :, :4])
    assert not np.any(zero[:, :, 4:])
    assert relation[0, 0, 7] > 0.0
    assert relation[0, 1, 7] > 0.0
    assert not np.any(relation[0, 2:, 4:])
    assert sum(item.numel() for item in PairTokenOrdinalNet().parameters()) == sum(item.numel() for item in PairTokenOrdinalNet().parameters())
    assert PairTokenOrdinalNet()(torch.from_numpy(relation)).shape == (1, 7)
