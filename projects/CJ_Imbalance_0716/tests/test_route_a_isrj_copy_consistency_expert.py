import torch

from experiments.run_route_a_isrj_copy_consistency_expert import _copy_consistency_features


def test_copy_consistency_features_are_finite_and_fixed_width():
    torch.manual_seed(3)
    signal = torch.randn(4, 2, 3000)
    features = _copy_consistency_features(signal)
    assert features.shape == (4, 20)
    assert torch.isfinite(features).all()

