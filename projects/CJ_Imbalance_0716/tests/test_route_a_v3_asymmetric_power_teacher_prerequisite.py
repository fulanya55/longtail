import numpy as np

from experiments.audit_route_a_v3_asymmetric_power_teacher_prerequisite import (
    aggregate_seed_results,
    summarize_pair,
)


def test_pair_summary_reports_threshold_free_high_low_rank_gain():
    targets = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    low = np.array([0.4, 0.3, 0.8, 0.2, 0.7, 0.5])
    high = np.array([0.1, 0.2, 0.3, 1.0, 1.1, 1.2])
    result = summarize_pair(high, low, targets)
    assert result["high_50db"]["auroc"] == 1.0
    assert result["high_minus_low"]["auroc"] > 0.0
    assert result["high_50db"]["top10_lift_over_prevalence"] > 1.0


def test_aggregate_requires_all_targets_and_four_positive_seeds():
    row = {"per_target": {}}
    for name in ("MFT", "VFT", "ISRJ"):
        row["per_target"][name] = {
            "high_50db": {"auroc": 0.7, "top10_lift_over_prevalence": 2.0},
            "low_10db": {"auroc": 0.6},
            "high_minus_low": {"auroc": 0.1, "aupr": 0.1},
            "paired_positive_score_increase_fraction": 0.8,
        }
    aggregate, checks = aggregate_seed_results([row] * 5)
    assert all(checks.values())
    assert aggregate["MFT"]["positive_high_minus_low_auroc_seeds"] == 5
