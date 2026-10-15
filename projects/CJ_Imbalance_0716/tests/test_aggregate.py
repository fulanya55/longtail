import pytest

from cjlt.analysis.aggregate import aggregate_metric, paired_difference


def test_aggregate_metric_reports_mean_std_and_ci95():
    summary = aggregate_metric([0.70, 0.72, 0.74, 0.76, 0.78])

    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.74)
    assert summary["std"] > 0.0
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]


def test_paired_difference_aligns_runs_by_seed():
    baseline = {2026: 0.70, 2027: 0.72, 2028: 0.71}
    method = {2026: 0.74, 2027: 0.75, 2028: 0.76}

    result = paired_difference(baseline, method)

    assert result["seeds"] == [2026, 2027, 2028]
    assert result["mean_difference"] == pytest.approx(0.04)

