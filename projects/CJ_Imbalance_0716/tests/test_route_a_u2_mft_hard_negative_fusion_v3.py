import numpy as np

from experiments.run_route_a_u2_mft_hard_negative_fusion_v3 import _fusion_features, _weights


def test_physical_fusion_features_and_hard_negative_weighting():
    logits = np.zeros((4, 6), dtype=np.float32)
    z = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    assert _fusion_features(logits, z).shape == (4, 4)
    assert _fusion_features(logits, None).shape == (4, 1)
    labels = np.array([[0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    weak = np.zeros_like(labels, dtype=bool)
    weak[0, 3] = True
    weights = _weights(labels, weak, 4.0, 4.0)
    assert weights[0] == 4.0
    assert weights[1] == 4.0
    assert weights[3] == 1.0
