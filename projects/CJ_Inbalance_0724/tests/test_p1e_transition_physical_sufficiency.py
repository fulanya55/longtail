import numpy as np

from cj0724.p1e_transition_physical_sufficiency import (
    binary_metrics,
    route_metrics,
    select_veto_threshold,
    stratified_identity_split,
)


def test_stratified_identity_split_is_disjoint_and_balanced():
    combinations = np.array([[int(bit) for bit in f"{value:06b}"] for value in range(1, 64)], dtype=np.uint8)
    labels = np.repeat(combinations, 20, axis=0)
    roles = stratified_identity_split(labels, 123)
    assert {name: len(values) for name, values in roles.items()} == {"fit": 630, "calibration": 315, "probe": 315}
    assert not np.intersect1d(roles["fit"], roles["calibration"]).size
    assert not np.intersect1d(roles["fit"], roles["probe"]).size
    for values, expected in ((roles["fit"], 10), (roles["calibration"], 5), (roles["probe"], 5)):
        _, counts = np.unique(labels[values], axis=0, return_counts=True)
        assert np.all(counts == expected)


def test_veto_threshold_meets_requested_recall_with_minimal_coverage():
    target = np.array([1, 1, 0, 0, 0], dtype=np.uint8)
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    selected = select_veto_threshold(target, score, 1.0)
    assert selected["recall"] == 1.0
    assert selected["veto_coverage"] == 0.4


def test_binary_metrics_reports_lift_over_prevalence():
    target = np.array([0, 0, 1, 1], dtype=np.uint8)
    score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(target, score)
    assert metrics["auroc"] == 1.0
    assert metrics["aupr_lift"] == 2.0


def test_route_metrics_rewards_helpful_expert_without_false_positives():
    labels = np.zeros((4, 6), dtype=np.uint8)
    labels[:2, 3] = 1
    main = np.zeros((4, 6), dtype=np.float32)
    main[:, 3] = np.array([0.4, 0.4, 0.1, 0.1])
    expert = np.zeros((4, 3), dtype=np.float32)
    expert[:, 0] = np.array([0.9, 0.8, 0.1, 0.1])
    metrics = route_metrics(labels, main, expert, np.ones(4, dtype=bool), 0, 0.5, alpha=0.75)
    assert metrics["recall_gain"] == 1.0
    assert metrics["fpr_delta"] == 0.0
