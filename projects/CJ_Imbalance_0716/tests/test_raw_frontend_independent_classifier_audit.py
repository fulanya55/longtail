import numpy as np

from experiments.run_route_a_raw_frontend_independent_classifier_audit import (
    _decision_metrics,
    _ece,
    _search_unified_threshold,
)


def test_ece_is_zero_for_perfect_binary_probabilities():
    probability = np.asarray([0.0, 1.0, 0.0, 1.0])
    targets = np.asarray([0, 1, 0, 1])
    assert _ece(probability, targets) == 0.0


def test_decision_metrics_reports_recall_and_fpr():
    probability = np.asarray([0.9, 0.4, 0.8, 0.1])
    targets = np.asarray([1, 1, 0, 0])
    metrics = _decision_metrics(probability, targets, 0.5)
    assert metrics["recall"] == 0.5
    assert metrics["fpr"] == 0.5


def test_unified_threshold_obeys_all_domain_fpr_constraint():
    ordinary_probability = np.asarray([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    ordinary_targets = np.asarray([1, 1, 1, 0, 0, 0])
    power = {
        "40": {"positive": np.asarray([0.9, 0.7]), "negative": np.asarray([0.6, 0.1])},
        "50": {"positive": np.asarray([0.8, 0.65]), "negative": np.asarray([0.55, 0.2])},
    }
    selected, _ = _search_unified_threshold(
        ordinary_probability, ordinary_targets, power, step=0.1, max_fpr=0.0
    )
    assert selected is not None
    assert selected["ordinary_fpr"] == 0.0
    assert selected["max_power_fpr"] == 0.0
    assert selected["threshold"] > 0.6
