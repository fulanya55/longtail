import numpy as np

from experiments.generate_route_a_h3_vft_three_strata_source import LENGTH, _sample_triplet


def test_h3_triplet_has_matched_background_and_vft_visibility_strata():
    signals, labels, power = _sample_triplet(3_400_000_000)
    assert len(signals) == 3
    assert all(signal.shape == (LENGTH,) for signal in signals)
    assert np.array_equal(labels[:, 4], [0, 1, 1])
    assert np.isnan(power[0, 4])
    assert np.allclose(power[1:, 4], [8.0, 20.0])
    assert np.array_equal(labels[0, [0, 1, 2, 3, 5]], labels[1, [0, 1, 2, 3, 5]])
    assert np.array_equal(labels[0, [0, 1, 2, 3, 5]], labels[2, [0, 1, 2, 3, 5]])
    assert np.allclose(power[0, [0, 1, 2, 3, 5]], power[1, [0, 1, 2, 3, 5]], equal_nan=True)
    assert np.allclose(power[0, [0, 1, 2, 3, 5]], power[2, [0, 1, 2, 3, 5]], equal_nan=True)
    assert not np.array_equal(signals[0], signals[1])
    assert not np.array_equal(signals[1], signals[2])
