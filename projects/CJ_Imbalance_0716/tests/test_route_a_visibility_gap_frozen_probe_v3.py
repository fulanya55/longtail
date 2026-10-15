import numpy as np

from experiments.run_route_a_visibility_gap_frozen_probe_v3 import _relative_power_gap


def test_relative_power_gap_handles_compounds_singletons_and_clipping():
    labels = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ], dtype=np.uint8)
    power = np.array([
        [-20, 0, np.nan, np.nan, np.nan, np.nan],
        [-100, -5, 0, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, 12, np.nan, np.nan],
    ], dtype=np.float64)
    gap = _relative_power_gap(labels, power, -40.0)
    assert gap[0, 0] == -20
    assert gap[0, 1] == 0
    assert gap[1, 0] == -40
    assert gap[1, 1] == -5
    assert gap[1, 2] == 0
    assert gap[2, 3] == 0
    assert np.isnan(gap[2, 0])

