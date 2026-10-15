from __future__ import annotations

import torch

from cjlt.models.raat_local_map import RAATLocalMapNet, extract_raat_map, parameter_count


def _iq() -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.randn(2, 2, 3000, generator=generator)


def test_maps_are_finite_deterministic_and_reference_sensitive() -> None:
    iq = _iq()
    expected = {"MFT": (2, 2, 24, 17), "VFT": (2, 2, 24, 17), "ISRJ": (2, 2, 12, 24)}
    for target, shape in expected.items():
        first = extract_raat_map(iq, target)
        second = extract_raat_map(iq, target)
        wrong = extract_raat_map(iq, target, reverse_chirp=True)
        assert first.shape == shape
        assert torch.isfinite(first).all()
        assert torch.equal(first, second)
        assert torch.linalg.vector_norm(first - wrong).item() > 0.0


def test_fixed_network_contract() -> None:
    assert parameter_count() == 7425
    output = RAATLocalMapNet()(extract_raat_map(_iq(), "MFT"))
    assert output.shape == (2,)
