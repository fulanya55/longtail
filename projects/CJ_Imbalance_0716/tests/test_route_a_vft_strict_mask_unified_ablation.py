import numpy as np

from experiments.run_route_a_vft_strict_mask_unified_ablation import ARMS, _fit_ridge


def test_strict_mask_unified_arms_and_balanced_ridge_fit():
    assert ARMS == ("StrictWeak_Q4", "AllPositive_NoWeakMask")
    model = _fit_ridge(np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32), np.asarray([0, 0, 1, 1], dtype=np.uint8), 10.0)
    assert model.coef_.shape == (1, 1)
