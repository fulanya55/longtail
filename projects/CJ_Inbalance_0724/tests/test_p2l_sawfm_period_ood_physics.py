import numpy as np

from cj0724.p2l_sawfm_period_ood_physics import summarize_predictions


def test_summary_reports_expected_reductions():
    target = np.asarray([1.0, 2.0, 3.0])
    predictions = {
        "power": np.asarray([25, 35, 50]),
        "target": target,
        "constant": np.asarray([0.0, 0.0, 0.0]),
        "raw": np.asarray([0.0, 0.0, 0.0]),
        "exact_component": np.asarray([1.0, 2.0, 3.0]),
        "separated_uncalibrated": np.asarray([0.5, 1.5, 2.5]),
        "separated_calibrated": np.asarray([0.75, 1.75, 2.75]),
    }
    summary = summarize_predictions(predictions)
    assert summary["exact_reduction_vs_constant"] == 1.0
    assert summary["calibrated_reduction_vs_raw"] > 0.0
