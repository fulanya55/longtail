"""Contract tests for the H4 P0 fixed beamspace-map observable."""
from __future__ import annotations

import torch

from cjlt.models.vft_staggered_pri import (
    DELAY_START,
    DELAY_STOP,
    FREQUENCY_HZ,
    record_samples,
    staggered_vft_beamspace_maps,
)


def test_h4_beamspace_maps_are_finite_and_coordinate_preserving() -> None:
    torch.manual_seed(20260803)
    iq = torch.randn(2, 2, record_samples())
    true_map, permuted_map = staggered_vft_beamspace_maps(iq)
    assert true_map.shape == (2, 3, len(FREQUENCY_HZ), DELAY_STOP - DELAY_START + 1)
    assert permuted_map.shape == true_map.shape
    assert torch.isfinite(true_map).all()
    assert torch.isfinite(permuted_map).all()


def test_h4_timestamp_control_changes_map_without_changing_capacity() -> None:
    torch.manual_seed(20260804)
    iq = torch.randn(1, 2, record_samples())
    true_map, permuted_map = staggered_vft_beamspace_maps(iq)
    assert true_map.numel() == permuted_map.numel()
    assert not torch.allclose(true_map, permuted_map)
