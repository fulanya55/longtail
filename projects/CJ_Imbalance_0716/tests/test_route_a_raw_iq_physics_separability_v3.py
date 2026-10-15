import torch

from experiments.run_route_a_raw_iq_physics_separability_v3 import _isrj_features,_mft_features,_vft_features


def test_fixed_raw_physics_features_are_finite_and_batched():
    signal=torch.randn(3,2,512)
    for function in (_mft_features,_vft_features,_isrj_features):
        values=function(signal)
        assert values.shape[0]==3
        assert values.ndim==2
        assert torch.isfinite(values).all()


def test_vft_grid_controls_only_image_dimension():
    signal=torch.randn(2,2,512)
    fine=_vft_features(signal,grid_size=16)
    coarse=_vft_features(signal,grid_size=8)
    assert fine.shape[1]-coarse.shape[1]==16*16-8*8
