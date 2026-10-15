import torch

from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import FrozenFeatureHead


def test_linear_and_mlp_heads_emit_six_logits():
    x=torch.randn(7,128)
    assert FrozenFeatureHead(128,"linear")(x).shape==(7,6)
    assert FrozenFeatureHead(128,"mlp",hidden_dim=4)(x).shape==(7,6)


def test_mlp_is_labelwise_and_low_capacity():
    model=FrozenFeatureHead(128,"mlp",hidden_dim=4)
    assert len(model.head)==6
    assert sum(p.numel() for p in model.parameters())==3126
