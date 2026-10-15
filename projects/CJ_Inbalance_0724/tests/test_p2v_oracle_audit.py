import numpy as np

from cj0724.p2v_oracle_audit import choose_oracle, risk_metrics


def test_choose_oracle_selects_smallest_error() -> None:
    target = np.asarray([0.0, 10.0, -5.0])
    first = np.asarray([1.0, 30.0, -20.0])
    second = np.asarray([8.0, 11.0, -4.0])
    prediction, choice = choose_oracle(target, [first, second])
    np.testing.assert_array_equal(prediction, [1.0, 11.0, -4.0])
    np.testing.assert_array_equal(choice, [0, 1, 1])


def test_risk_metrics_perfectly_orders_catastrophic_errors() -> None:
    error = np.asarray([100.0, 200.0, 6000.0, 8000.0])
    coherence = np.asarray([0.9, 0.8, 0.2, 0.1])
    result = risk_metrics(error, coherence)
    assert result["risk_aupr"] == 1.0
    assert result["risk_roc_auc"] == 1.0
