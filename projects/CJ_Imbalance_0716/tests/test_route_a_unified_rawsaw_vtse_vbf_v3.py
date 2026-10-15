import numpy as np
import torch

from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (
    GenericBoundedFusion,
    _variant_logits,
)


def test_generic_fusion_has_same_six_trainable_scalars_as_vbf_contract():
    model = GenericBoundedFusion(residual_cap=0.5)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6


def test_generic_fusion_residual_is_bounded_and_target_only():
    model = GenericBoundedFusion(residual_cap=0.5)
    base = torch.zeros(4, 6)
    expert = torch.full((4, 3), 100.0)
    _, residual = model(base, expert, return_residual=True)
    assert torch.max(torch.abs(residual)).item() <= 0.500001
    assert torch.count_nonzero(residual[:, :3]).item() == 0


def test_unified_routing_changes_only_preregistered_labels():
    base = {"early": np.zeros((2, 6), dtype=np.float32)}
    raw = {"early": np.asarray([3.0, 4.0], dtype=np.float32)}
    visibility = {"early": np.ones((2, 6), dtype=np.float32)}
    generic = {"early": np.full((2, 6), 2.0, dtype=np.float32)}

    routed = _variant_logits(base, raw, visibility, generic)
    unified = routed["C3_Unified_RawSaw_VTSE_VBF"]["early"]

    assert np.allclose(unified[:, 0], 0.0)
    assert np.allclose(unified[:, 1], raw["early"])
    assert np.allclose(unified[:, 2], 0.0)
    assert np.allclose(unified[:, 3:6], 1.0)
