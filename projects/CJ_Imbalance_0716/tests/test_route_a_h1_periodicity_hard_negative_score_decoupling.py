import numpy as np

from experiments.run_route_a_h1_periodicity_hard_negative_score_decoupling import (
    _score_features,
    _weights,
)


def test_hard_negative_weighting_only_changes_top_delta_negatives():
    n = 100
    b0 = np.zeros((n, 6), dtype=np.float32)
    b1 = np.zeros((n, 6), dtype=np.float32)
    b2 = np.zeros((n, 6), dtype=np.float32)
    b2[:, 5] = np.linspace(-1.0, 1.0, n)
    features = _score_features(b0, b1, b2)
    targets = np.zeros((n, 6), dtype=np.uint8)
    targets[:20, 5] = 1
    weak = np.zeros((n, 6), dtype=bool)
    weak[:10, 5] = True
    control, treatment, _, hard = _weights(targets, weak, features[:, 1])
    assert np.all(control[:10] == 4.0)
    assert np.all(treatment[hard] == 4.0)
    assert np.all(treatment[~hard] == control[~hard])
    assert hard.sum() == 20

