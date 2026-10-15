import numpy as np

from experiments.run_route_a_vft_weighting_uncertainty_core_ablation import ARMS, _features, _weights


def test_fixed_arms_weights_and_gate_feature_dimensions():
    assert ARMS == ("A0_UniformBCE_NoGate", "A1_WeakWeight_NoGate", "A2_ThreeStratum_NoGate", "A3_WeakWeight_SqrtGate")
    labels = np.asarray([1, 1, 1, 0, 0], dtype=np.uint8)
    weak = np.asarray([1, 0, 1, 0, 0], dtype=bool)
    assert np.array_equal(_weights(labels, weak, ARMS[0]), np.ones(5))
    assert np.array_equal(_weights(labels, weak, ARMS[1]), np.asarray([4, 1, 4, 1, 1], dtype=float))
    strata = _weights(labels, weak, ARMS[2])
    assert np.isclose(strata[(labels == 1) & weak].sum(), len(labels) / 3)
    assert np.isclose(strata[(labels == 1) & ~weak].sum(), len(labels) / 3)
    assert np.isclose(strata[labels == 0].sum(), len(labels) / 3)
    logits = np.zeros((5, 6), dtype=np.float32)
    z = np.arange(5, dtype=np.float32)
    assert _features(logits, z, "constant").shape == _features(logits, z, "sqrt").shape == (5, 4)
