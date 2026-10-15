import numpy as np

from experiments.run_route_a_vft_sqrt_linear_gate_ablation import _features


def test_sqrt_and_linear_gates_keep_same_capacity_but_differ_off_boundary():
    logits = np.zeros((2, 6), dtype=np.float32)
    logits[:, 4] = [-2.0, 0.0]
    z = np.array([1.0, 1.0], dtype=np.float32)
    sqrt = _features(logits, z, "sqrt")
    linear = _features(logits, z, "linear")
    assert sqrt.shape == linear.shape == (2, 4)
    assert np.isclose(sqrt[1, 2], linear[1, 2])
    assert not np.isclose(sqrt[0, 2], linear[0, 2])
