import numpy as np

from cj0724.p2q_cross_window_consensus import select_cross_window_lag


def test_cross_window_consensus_corrects_stable_fifth_multiple() -> None:
    acf = np.zeros(3000, dtype=np.float64)
    acf[0] = 1.0
    acf[168:173] = (0.0, 0.05, 0.20, 0.05, 0.0)
    acf[848:853] = (0.0, 0.10, 0.60, 0.10, 0.0)
    decision = select_cross_window_lag(acf, 850, np.asarray([169, 170, 171]))
    assert decision["corrected"] is True
    assert decision["lag"] == 170
    assert decision["ratio"] == 5


def test_cross_window_consensus_rejects_unstable_windows() -> None:
    acf = np.zeros(3000, dtype=np.float64)
    acf[0] = 1.0
    acf[168:173] = (0.0, 0.05, 0.20, 0.05, 0.0)
    decision = select_cross_window_lag(acf, 850, np.asarray([170, 230, 310]))
    assert decision["corrected"] is False
    assert decision["lag"] == 850
