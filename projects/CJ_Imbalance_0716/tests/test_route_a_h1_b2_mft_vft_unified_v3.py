import numpy as np

from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _fusion_features, _replace_vft


def test_vft_fusion_feature_shape_and_finiteness():
    logits = np.zeros((7, 6), dtype=np.float32)
    z = np.linspace(-2, 2, 7, dtype=np.float32)
    features = _fusion_features(logits, z)
    assert features.shape == (7, 4)
    assert np.isfinite(features).all()


def test_replace_vft_preserves_all_other_logits():
    logits = np.arange(24, dtype=np.float32).reshape(4, 6)
    output = _replace_vft(logits, np.asarray([-1, 0, 1, 2], dtype=np.float32))
    assert np.array_equal(output[:, [0, 1, 2, 3, 5]], logits[:, [0, 1, 2, 3, 5]])
    assert np.array_equal(output[:, 4], np.asarray([-1, 0, 1, 2], dtype=np.float32))
