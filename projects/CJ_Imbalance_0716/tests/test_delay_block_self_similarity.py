from __future__ import annotations

import torch

from cjlt.models.delay_block_self_similarity import DelayBlockReplicaProbe, DelayBlockSelfSimilarityMap, ScaleHoughDelayBlockMap, ScaleHoughReplicaProbe, WidthNormalizedDelayBlockMap, WidthNormalizedDelayBlockReplicaProbe


def test_delay_block_map_has_fixed_finite_shape():
    feature = DelayBlockSelfSimilarityMap()
    value = feature(torch.randn(3, 2, 3000))
    assert value.shape == (3, 2, 26, 12)
    assert torch.isfinite(value).all()


def test_delay_block_map_retains_complex_phase_relation():
    feature = DelayBlockSelfSimilarityMap()
    t = torch.arange(3000, dtype=torch.float32)[None]
    phase = 0.03 * t
    original = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)
    changed = torch.stack([torch.cos(-phase), torch.sin(-phase)], dim=1)
    original_map = feature(original)
    changed_map = feature(changed)
    assert not torch.equal(original_map, changed_map)
    assert torch.allclose(original_map[:, 0], changed_map[:, 0], atol=1e-5)
    assert torch.allclose(original_map[:, 1], -changed_map[:, 1], atol=1e-5)


def test_delay_block_replica_probe_has_scalar_output():
    model = DelayBlockReplicaProbe()
    output = model(torch.randn(4, 2, 3000))
    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_width_normalized_map_and_probe_have_fixed_contract():
    signal = torch.randn(3, 2, 3000)
    widths = torch.tensor([100, 200, 300])
    mapped = WidthNormalizedDelayBlockMap()(signal, widths)
    assert mapped.shape == (3, 2, 8, 12)
    assert torch.isfinite(mapped).all()
    assert WidthNormalizedDelayBlockReplicaProbe()(signal, widths).shape == (3,)


def test_scale_hough_map_and_probe_have_fixed_inference_ready_contract():
    signal = torch.randn(3, 2, 3000)
    mapped = ScaleHoughDelayBlockMap()(signal)
    assert mapped.shape == (3, 2, 88, 12)
    assert torch.isfinite(mapped).all()
    assert ScaleHoughReplicaProbe().forward_map(mapped).shape == (3,)
