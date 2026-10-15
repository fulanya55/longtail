import numpy as np

from experiments.run_route_a_vft_isrj_cross_expert_head import HEADS, cross_features


def test_cross_expert_controls_have_equal_dimension():
    q1 = np.zeros((8, 6), dtype=np.float32); q2 = q1.copy(); expert = q1.copy()
    aligned = cross_features(q1, q2, expert, HEADS[0])
    capacity = cross_features(q1, q2, expert, HEADS[1])
    shuffled = cross_features(q1, q2, expert, HEADS[2], np.arange(7, -1, -1))
    assert aligned.shape == capacity.shape == shuffled.shape == (8, 5)
