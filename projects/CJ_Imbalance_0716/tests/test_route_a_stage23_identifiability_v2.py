import torch

from experiments.run_route_a_stage23_identifiability_v2 import (
    complex_physics_statistics,
    temporal_statistics,
)


def test_stage_statistics_dimensions_and_finiteness():
    values = torch.randn(4, 12, 31)
    assert temporal_statistics(values, topk=8).shape == (4, 48)
    real, imag = torch.randn(4, 12, 31), torch.randn(4, 12, 31)
    result = complex_physics_statistics(real, imag, topk=8)
    assert result.shape == (4, 12 * 4 * 6)
    assert torch.isfinite(result).all()

