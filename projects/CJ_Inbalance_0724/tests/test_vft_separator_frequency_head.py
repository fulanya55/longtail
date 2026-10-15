import torch

from cj0724.vft_separator_frequency_head import instantaneous_frequency_features
from cj0724.vft_frequency_mechanism_controls import global_summary


def test_instantaneous_frequency_features_have_fixed_finite_dimensions():
    iq = torch.randn(3, 2, 3000)
    output = instantaneous_frequency_features(iq, bins=20)
    assert output["if_trajectory"].shape == (3, 120)
    assert output["if_dynamics"].shape == (3, 180)
    assert output["if_combined"].shape == (3, 300)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_constant_tone_keeps_phase_increment_trajectory_stable():
    phase = torch.arange(3000, dtype=torch.float32) * 0.2
    iq = torch.stack((phase.cos(), phase.sin()))[None]
    output = instantaneous_frequency_features(iq, bins=30)["if_trajectory"]
    raw_means = output[:, :30]
    assert float(raw_means.std()) < 1.0e-4


def test_global_summary_collapses_time_bins_only():
    trajectory = torch.randn(4, 6 * 20).numpy()
    assert global_summary(trajectory, bins=20).shape == (4, 24)
