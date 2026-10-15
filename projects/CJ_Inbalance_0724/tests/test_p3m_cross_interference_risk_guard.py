import numpy as np

from cj0724.p3m_cross_interference_risk_guard import apply_veto


def test_apply_veto_only_changes_isrj_column():
    probability = np.full((3, 6), 0.7)
    guarded, veto = apply_veto(probability, np.array([0.1, 0.9, 0.4]), 0.5)
    assert veto.tolist() == [False, True, False]
    assert guarded[1, 5] == 0.0
    assert np.allclose(guarded[:, :5], probability[:, :5])
