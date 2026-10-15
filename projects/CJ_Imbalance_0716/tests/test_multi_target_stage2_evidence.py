import torch

from cjlt.models.evidence_readout import MultiTargetStage2LowRankEvidence


def test_multi_target_stage2_shapes_bounds_and_gradients():
    model = MultiTargetStage2LowRankEvidence(channels=64, targets=3, rank=8,
                                             hidden_channels=32, topk=8)
    values = torch.randn(4, 64, 40, requires_grad=True)
    logits, maps, diagnostics = model(values, return_map=True)
    assert logits.shape == (4, 3)
    assert maps.shape == (4, 3, 40)
    assert diagnostics["adapter_scale"].abs() <= 0.20
    assert torch.allclose(maps.mean(dim=-1), torch.zeros(4, 3), atol=1e-5)
    logits.mean().backward()
    assert torch.isfinite(values.grad).all()

