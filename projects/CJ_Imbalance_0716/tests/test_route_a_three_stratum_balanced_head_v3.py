import numpy as np

from experiments.run_route_a_three_stratum_balanced_head_v3 import _stratum_weights


def test_stratum_weights_equalize_each_label_tier():
    targets = np.asarray([[1,1,1],[1,1,1],[1,0,0],[0,0,0],[0,0,0],[0,0,0]], dtype=np.float32)
    weak = np.asarray([[1,1,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]], dtype=bool)
    weights, counts = _stratum_weights(targets, weak)
    for local, name in enumerate(("MFT", "VFT", "ISRJ")):
        positive = targets[:, local] > .5
        masks = (positive & weak[:, local], positive & ~weak[:, local], ~positive)
        np.testing.assert_allclose([weights[mask, local].sum() for mask in masks], [1/3]*3)
        assert all(value > 0 for value in counts[name].values())
