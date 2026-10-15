import numpy as np

from experiments.audit_route_a_u2_mft_peak_guard_statistics import _holm
from experiments.audit_route_a_u2_mft_peak_guard_v10_subgroups import _margin


def test_holm_is_monotone_and_bounded():
    adjusted = _holm({"a": 0.01, "b": 0.02, "c": 0.20})
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_relative_margin_uses_strongest_other_component():
    labels = np.asarray([[1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    power = np.asarray([[20, np.nan, np.nan, 5, 10, np.nan], [np.nan, np.nan, np.nan, 12, np.nan, np.nan]], dtype=np.float32)
    margin = _margin(power, labels)
    assert margin[0] == -15.0
    assert np.isinf(margin[1])

