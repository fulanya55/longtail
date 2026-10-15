import numpy as np

from experiments.run_route_a_h2v_b0_admission import inversion_diagnostic


def test_h2v_inversion_diagnostic_counts_only_weak_vs_absent_pairs():
    logits = np.zeros((4, 6), dtype=np.float32); logits[:, 3] = [0.2, 0.7, 0.3, 0.8]
    labels = np.zeros((4, 6), dtype=np.float32); labels[2:, 3] = 1
    weak, physics = np.array([False, False, True, False]), np.array([0.1, 0.4, 0.8, 0.7])
    result = inversion_diagnostic(logits, labels, weak, physics)
    assert result["weak_vs_absent_pair_count"] == 2
    assert result["b0_inverted_pair_count"] == 1
    assert result["physics_correct_order_within_b0_inversions"] == 1.0
