import numpy as np

from experiments.run_route_a_visibility_temporal_expert_v2 import _reconnect


def test_vtse_reconnection_is_bounded_and_leaves_non_targets_unchanged():
    base = np.zeros((4, 6), dtype=np.float32)
    score = np.asarray([[10.0, -10.0, 2.0]] * 4, dtype=np.float32)
    logits, residual = _reconnect(base, score, np.zeros(3), np.ones(3),
                                  np.ones(3), 0.35)
    assert np.max(np.abs(residual)) <= 0.350001
    assert np.allclose(logits[:, :3], base[:, :3])
    assert np.allclose(residual[:, :3], 0.0)
    assert np.allclose(residual[:, 3:], logits[:, 3:] - base[:, 3:])


def test_sqrt_uncertainty_gate_is_stronger_but_still_bounded():
    base = np.full((2, 6), 2.0, dtype=np.float32)
    score = np.full((2, 3), 10.0, dtype=np.float32)
    _, linear = _reconnect(base, score, np.zeros(3), np.ones(3),
                           np.ones(3), 0.35, uncertainty_power=1.0)
    _, square_root = _reconnect(base, score, np.zeros(3), np.ones(3),
                                np.ones(3), 0.35, uncertainty_power=0.5)
    assert np.abs(square_root[:, 3:]).mean() > np.abs(linear[:, 3:]).mean()
    assert np.abs(square_root).max() <= 0.350001
