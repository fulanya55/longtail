import numpy as np

from cj0724.p2g_sawfm_separator_confirmation import per_power_metrics


def test_per_power_metrics_uses_identity_major_power_order():
    target = np.asarray([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    prediction = target.copy()
    result = per_power_metrics(prediction, target)
    assert set(result) == {"35", "40", "45", "50"}
    assert all(row["mae"] == 0.0 for row in result.values())
