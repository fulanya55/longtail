import numpy as np

from cj0724.p2p_harmonic_consensus import select_harmonic_consensus_lag


def test_consensus_backtracks_supported_fourth_harmonic() -> None:
    acf = np.zeros(1000, dtype=np.float64)
    acf[0] = 1.0
    for lag, height in ((50, 0.08), (100, 0.12), (150, 0.16), (200, 0.60)):
        acf[lag - 1:lag + 2] = (height * 0.2, height, height * 0.2)
    decision = select_harmonic_consensus_lag(acf, 200)
    assert decision["backtracked"] is True
    assert decision["lag"] == 50
    assert decision["divisor"] == 4


def test_consensus_keeps_isolated_baseline_peak() -> None:
    acf = np.zeros(1000, dtype=np.float64)
    acf[0] = 1.0
    acf[199:202] = (0.1, 0.7, 0.1)
    decision = select_harmonic_consensus_lag(acf, 200)
    assert decision["backtracked"] is False
    assert decision["lag"] == 200
