import numpy as np

from cj0724.p2j_full_parameter_attachment import attach_saw_predictions, build_promotion_gates


def test_attach_changes_only_saw_period_at_candidates():
    old = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    uncalibrated, calibrated = attach_saw_predictions(
        old, np.asarray([1, 3]), np.asarray([10.0, 20.0]), np.asarray([11.0, 22.0]),
    )
    unchanged = np.ones((6, 2), dtype=bool)
    unchanged[1, 1] = False
    assert np.array_equal(uncalibrated[:, unchanged], old[:, unchanged])
    assert np.array_equal(calibrated[:, unchanged], old[:, unchanged])
    assert uncalibrated[1, 1, 1] == 10.0
    assert calibrated[3, 1, 1] == 22.0
    assert uncalibrated[0, 1, 1] == old[0, 1, 1]


def test_promotion_gates_enforce_low_power_guard():
    rows = [{
        "full_improvement_vs_safe": 0.06,
        "full_improvement_vs_old": 0.01,
        "miss_delta_vs_old": -0.01,
        "worst_combo_delta_vs_old": -0.01,
        "high_power_saw_reduction_vs_old": 0.40,
        "low_power_saw_ratios_vs_old": [1.0, 1.01, 1.02, 1.03, 1.04],
    } for _ in range(3)]
    bootstrap = {"ci95_high": -0.001}
    gates = build_promotion_gates(rows, bootstrap, True, True)
    assert all(gates.values())
    rows[0]["low_power_saw_ratios_vs_old"][-1] = 1.06
    gates = build_promotion_gates(rows, bootstrap, True, True)
    assert not gates["each_low_power_saw_ratio_vs_old_at_most_1p05"]
