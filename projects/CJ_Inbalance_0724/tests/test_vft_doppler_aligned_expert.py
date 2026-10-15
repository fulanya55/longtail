import numpy as np
import torch

from cj0724.vft_doppler_aligned_expert import (
    CHIRP_RATE_HZ_PER_S,
    FS_HZ,
    SIGNAL_LENGTH,
    estimate_aligned_doppler,
    lfm_reference,
    regression_metrics,
)


def test_aligned_estimator_recovers_clean_shifted_vft():
    device = torch.device("cpu")
    reference = lfm_reference(device)
    time_s = torch.arange(SIGNAL_LENGTH, dtype=torch.float32) / FS_HZ
    doppler_hz = 4300.0
    signal = torch.roll(
        reference * torch.exp(2j * torch.pi * doppler_hz * time_s), 317
    )
    iq = torch.stack((signal.real, signal.imag), dim=0)[None]
    result = estimate_aligned_doppler(iq)
    assert abs(float(result["prediction_hz"][0]) - doppler_hz) <= 100.0
    assert abs(float(result["delay_samples"][0]) - 317.0) <= 2.0


def test_wrong_reference_does_not_match_clean_vft_as_well():
    device = torch.device("cpu")
    reference = lfm_reference(device)
    time_s = torch.arange(SIGNAL_LENGTH, dtype=torch.float32) / FS_HZ
    signal = torch.roll(
        reference * torch.exp(2j * torch.pi * 5100.0 * time_s), 240
    )
    iq = torch.stack((signal.real, signal.imag), dim=0)[None]
    correct = estimate_aligned_doppler(iq)
    wrong = estimate_aligned_doppler(iq, chirp_rate_hz_per_s=0.8 * CHIRP_RATE_HZ_PER_S)
    assert float(correct["peak_score"][0]) > float(wrong["peak_score"][0])


def test_regression_metrics_use_training_constant_baseline():
    target = np.array([1000.0, 2000.0, 3000.0])
    prediction = np.array([1100.0, 1900.0, 2900.0])
    metrics = regression_metrics(prediction, target, 2000.0)
    assert metrics["mae_hz"] < metrics["constant_median_mae_hz"]
    assert metrics["relative_mae_reduction_vs_constant"] > 0.0
