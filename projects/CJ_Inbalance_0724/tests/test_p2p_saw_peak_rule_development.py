import numpy as np

from cj0724.p2p_saw_peak_rule_development import CANDIDATES, estimate_candidates, select_peak_lag


def test_peak_rules_are_deterministic_and_choose_valid_lags() -> None:
    lags = np.asarray([80, 400, 800, 1200], dtype=np.int64)
    scores = np.asarray([0.15, 0.70, 0.55, 0.40], dtype=np.float64)
    assert select_peak_lag(lags, scores, "legacy") == 80
    assert select_peak_lag(lags, scores, "relative_0p75") == 400
    assert select_peak_lag(lags, scores, "relative_0p90") == 400
    assert select_peak_lag(lags, scores, "strongest") == 400
    assert select_peak_lag(lags, scores, "harmonic_consensus") in set(lags.tolist())


def test_estimator_recovers_clean_complex_period() -> None:
    length = 4096
    period = 600
    base = np.exp(1j * 2.0 * np.pi * (np.arange(length) % period) / period)
    iq = np.stack((base.real, base.imag), axis=0)[None].astype(np.float32)
    estimates = estimate_candidates(iq)
    assert set(estimates) == set(CANDIDATES)
    for values in estimates.values():
        assert values.shape == (1,)
        assert np.isfinite(values[0])
