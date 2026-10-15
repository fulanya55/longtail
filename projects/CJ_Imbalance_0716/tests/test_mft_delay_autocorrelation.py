from __future__ import annotations

import torch

from cjlt.models.mft_delay_autocorrelation import (
    delay_autocorrelation_effective_support,
    fixed_phase_permutation,
    permuted_delay_autocorrelation_effective_support,
)


def _copies(count: int) -> torch.Tensor:
    length = 3000
    t = torch.arange(1000, dtype=torch.float64) / 10e6
    pulse = torch.polar(torch.ones_like(t), torch.pi * (5e6 / 100e-6) * t.square()).to(torch.complex64)
    delays = [410, 435, 470, 520, 580, 655, 745, 850]
    x = torch.zeros((4, length), dtype=torch.complex64)
    for delay in delays[:count]:
        x[:, delay : delay + 1000] += pulse
    return torch.stack((x.real, x.imag), dim=1)


def test_effective_autocorrelation_support_increases_with_copy_count() -> None:
    low = delay_autocorrelation_effective_support(_copies(3))
    high = delay_autocorrelation_effective_support(_copies(8))
    assert torch.all(high > low)


def test_phase_permutation_is_deterministic_and_nonidentity() -> None:
    index = fixed_phase_permutation(3000)
    assert torch.equal(index, fixed_phase_permutation(3000))
    assert not torch.equal(index, torch.arange(3000))
    result = permuted_delay_autocorrelation_effective_support(_copies(3))
    assert torch.isfinite(result).all()


def test_autocorrelation_contract_rejects_invalid_input() -> None:
    try:
        delay_autocorrelation_effective_support(torch.zeros((2, 3000)))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("invalid I/Q shape must be rejected")
