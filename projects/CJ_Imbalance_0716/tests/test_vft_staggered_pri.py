from __future__ import annotations

import torch

from cjlt.models.vft_staggered_pri import FS_HZ, PULSE_SAMPLES, lfm_pulse, pulse_start_samples, record_samples, staggered_vft_contrasts, staggered_vft_evidence, staggered_vft_features


def _signal(doppler_hz: float, arrival: int = 280) -> torch.Tensor:
    starts = pulse_start_samples()
    signal = torch.zeros(record_samples(), dtype=torch.complex64)
    pulse = lfm_pulse(dtype=torch.complex64)
    for start in starts.tolist():
        index = torch.arange(PULSE_SAMPLES, dtype=torch.float64) + start + arrival
        signal[start + arrival : start + arrival + PULSE_SAMPLES] = pulse * torch.exp(2j * torch.pi * doppler_hz * index / FS_HZ).to(torch.complex64)
    return torch.stack((signal.real, signal.imag), dim=0)


def test_fixed_staggered_contract() -> None:
    assert pulse_start_samples().tolist() == [0, 3000, 6310, 9310, 12620, 15620, 18930, 21930]
    assert record_samples() == 24930


def test_true_timestamp_contrast_orders_clean_low_and_high() -> None:
    low = _signal(1500.0)
    high = _signal(6250.0)
    candidate, control = staggered_vft_contrasts(torch.stack((low, high)))
    assert candidate.shape == control.shape == (2,)
    assert torch.isfinite(candidate).all() and torch.isfinite(control).all()
    assert candidate[1] > candidate[0]


def test_rejects_wrong_input_shape() -> None:
    try:
        staggered_vft_contrasts(torch.zeros(1, 2, record_samples() - 1))
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("wrong input length must fail")


def test_evidence_exposes_matched_timestamp_and_control_scores() -> None:
    candidate, control = staggered_vft_evidence(_signal(6250.0).unsqueeze(0))
    assert candidate.shape == control.shape == (1,)
    assert torch.isfinite(candidate).all() and torch.isfinite(control).all()


def test_feature_grids_are_finite_and_matched_shape() -> None:
    candidate, control = staggered_vft_features(_signal(6250.0).unsqueeze(0))
    assert candidate.shape == control.shape == (1, 25)
    assert torch.isfinite(candidate).all() and torch.isfinite(control).all()
