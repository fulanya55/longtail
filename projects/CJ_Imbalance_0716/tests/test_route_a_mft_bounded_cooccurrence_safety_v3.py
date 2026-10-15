import numpy as np

from experiments.run_route_a_mft_bounded_cooccurrence_safety_v3 import _bounded_score


def test_bounded_score_is_one_sided_and_capped():
    base = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    head = np.array([-10.0, 10.0, 0.0], dtype=np.float32)
    output = _bounded_score(base, head, 0.5)
    delta = output - base
    assert np.all(delta <= 1e-7)
    assert np.all(delta >= -0.500001)
    assert delta[1] == 0.0
