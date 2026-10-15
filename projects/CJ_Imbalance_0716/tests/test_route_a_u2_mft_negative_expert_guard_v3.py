import numpy as np

from experiments.run_route_a_u2_mft_negative_expert_guard_v3 import _guarded_logits


def test_negative_guard_never_raises_mft_and_preserves_other_labels():
    base = np.zeros((3, 6), dtype=np.float32)
    expert = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    output, residual = _guarded_logits(base, expert, cap=0.5, signed=False)
    assert np.all(residual <= 0)
    assert residual[2] == 0
    assert np.array_equal(output[:, [0, 1, 2, 4, 5]], base[:, [0, 1, 2, 4, 5]])
