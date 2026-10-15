import numpy as np

from experiments.run_route_a_isrj_context_domain_gate_diagnostic_v3 import (
    _context_gates,
    _context_patterns,
    _rank_metrics,
)


def test_context_patterns_use_only_weak_isrj_positives():
    labels = np.asarray([
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
    ], dtype=np.int8)
    weak = np.zeros_like(labels, dtype=bool)
    weak[0, 5] = True
    weak[1, 5] = True
    patterns = _context_patterns(labels, weak)
    assert patterns.tolist() == [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]


def test_context_gates_have_expected_oracle_hard_and_soft_behavior():
    patterns = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
    targets = np.asarray([[1, 0], [1, 1]], dtype=np.int8)
    probabilities = np.asarray([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32)
    predictions = (probabilities >= 0.5).astype(np.int8)
    oracle, soft, hard = _context_gates(targets, probabilities, predictions, patterns)
    assert oracle.tolist() == [1.0, 0.0]
    assert hard.tolist() == [1.0, 0.0]
    assert 0.8 < soft[0] <= 1.0
    assert 0.0 < soft[1] < soft[0]


def test_rank_metrics_rewards_correct_weak_positive_ordering():
    target = np.asarray([1, 1, 0, 0], dtype=np.int8)
    weak = np.asarray([True, False, False, False])
    good = _rank_metrics(target, weak, np.asarray([0.8, 0.7, 0.2, 0.1]))
    bad = _rank_metrics(target, weak, np.asarray([0.1, 0.7, 0.8, 0.2]))
    assert good["weak_vs_negative_auroc"] > bad["weak_vs_negative_auroc"]
    assert good["aupr"] > bad["aupr"]

