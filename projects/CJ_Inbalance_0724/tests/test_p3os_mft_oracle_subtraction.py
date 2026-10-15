import h5py
import numpy as np

from cj0724.data import complex_rms_normalize
from cj0724.p3c_mft_sparse_deconvolution import estimate_sparse_count
from cj0724.p3os_mft_oracle_subtraction import (
    paired_bootstrap_source_fpr_reduction,
    phase_scrambled_energy_control,
    reconstruct_power_scaled_mft,
    _read_noise_power,
    rms_normalize_batch,
    transformed_iq,
)


def _mft_parameters():
    delays = np.full((2, 8), np.nan)
    amplitudes = np.full((2, 8), np.nan)
    delays[0, :2] = [300, 420]
    delays[1, :3] = [320, 400, 480]
    amplitudes[0, :2] = [1.0, 0.8]
    amplitudes[1, :3] = [1.0, 0.7, 0.6]
    return delays / 10.0e6, amplitudes


def test_p3os_power_scaled_mft_preserves_shape_count_and_target_rms() -> None:
    delays, amplitudes = _mft_parameters()
    component = reconstruct_power_scaled_mft(delays, amplitudes, np.asarray([20.0, 30.0]), noise_power=1.0)
    rms = np.sqrt(np.mean(component[:, 0] ** 2 + component[:, 1] ** 2, axis=1))
    assert np.allclose(rms, [10.0, 10.0 ** 1.5], rtol=1.0e-5)
    normalized = rms_normalize_batch(component)
    assert estimate_sparse_count(normalized)["prediction"].tolist() == [2.0, 3.0]


def test_p3os_controls_use_the_same_normalization_and_preserve_energy() -> None:
    delays, amplitudes = _mft_parameters()
    component = reconstruct_power_scaled_mft(delays, amplitudes, np.asarray([10.0, 10.0]), noise_power=1.0)
    raw = component + 0.01
    identities = np.asarray([101, 102], dtype=np.int64)
    c0 = transformed_iq(raw, component, identities, 25, "C0_raw")
    assert np.allclose(c0[0], complex_rms_normalize(raw[0]), atol=1.0e-6)
    control_a = phase_scrambled_energy_control(component, identities, 25)
    control_b = phase_scrambled_energy_control(component, identities, 25)
    assert np.array_equal(control_a, control_b)
    energy = lambda value: np.mean(value[:, 0] ** 2 + value[:, 1] ** 2, axis=1)
    assert np.allclose(energy(component), energy(control_a), rtol=1.0e-5, atol=1.0e-6)
    assert not np.allclose(component, control_a)
    assert transformed_iq(raw, component, identities, 25, "C1_exact_mft_subtraction").shape == raw.shape
    assert transformed_iq(raw, component, identities, 25, "C2_energy_matched_non_mft_control").shape == raw.shape


def test_p3os_identity_clustered_bootstrap_recovers_source_fpr_reduction() -> None:
    identities = np.repeat(np.arange(10, dtype=np.int64), 4)
    labels = np.zeros((len(identities), 6), dtype=np.uint8)
    labels[:, 3] = 1
    # C0 crosses the frozen ISRJ threshold on every source row; C1 does not.
    c0 = np.full(len(identities), 0.9, dtype=np.float64)
    c1 = np.full(len(identities), 0.1, dtype=np.float64)
    result = paired_bootstrap_source_fpr_reduction(c0, c1, labels, identities, 0.5, seed=7, resamples=200)
    assert result["observed_reduction"] == 1.0
    assert result["ci95_low"] > 0.99
    assert result["cluster_count"] == 10


def test_p3os_noise_power_follows_rechunk_cache_source_receipt(tmp_path) -> None:
    source = tmp_path / "source.mat"
    cache = tmp_path / "cache.mat"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("config/radar/noise_power", data=np.asarray([2.5]))
    with h5py.File(cache, "w") as handle:
        handle.attrs["source_path"] = str(source)
    with h5py.File(cache, "r") as handle:
        assert _read_noise_power(handle) == 2.5
