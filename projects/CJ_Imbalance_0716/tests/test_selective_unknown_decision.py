import numpy as np

from experiments.run_route_a_selective_unknown_decision import (
    _search_unified_thresholds,
    _selective_metrics,
)


def test_selective_metrics_separates_positive_negative_and_unknown():
    metrics = _selective_metrics(
        np.asarray([0.9, 0.6, 0.1]),
        np.asarray([0.8, 0.4, 0.05]),
        0.2,
        0.7,
    )
    assert metrics["confirmed_positive_recall"] == 1.0 / 3.0
    assert metrics["confirmed_positive_fpr"] == 1.0 / 3.0
    assert metrics["positive_false_negative_rate"] == 1.0 / 3.0
    assert metrics["decision_coverage"] == 4.0 / 6.0


def test_unified_search_finds_feasible_nontrivial_reject_region():
    ordinary_probability = np.asarray([0.9, 0.8, 0.2, 0.1])
    ordinary_targets = np.asarray([1, 1, 0, 0], dtype=bool)
    power = {
        "40": {"positive": np.asarray([0.8, 0.6]), "negative": np.asarray([0.2, 0.1])},
        "45": {"positive": np.asarray([0.7, 0.5]), "negative": np.asarray([0.2, 0.1])},
        "50": {"positive": np.asarray([0.6, 0.4]), "negative": np.asarray([0.2, 0.1])},
    }
    selected, rows = _search_unified_thresholds(
        ordinary_probability,
        ordinary_targets,
        power,
        step=0.1,
        max_fpr=0.2,
        max_positive_fn=0.1,
        min_ordinary_coverage=0.5,
        min_power_coverage=0.4,
    )
    assert rows
    assert selected is not None
    assert selected["feasible"]
    assert selected["t_negative"] < selected["t_positive"]
