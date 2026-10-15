import numpy as np

from cj0724.p1k_joint_level3_system import lazy_candidate_masks


def test_lazy_candidate_masks_follow_frozen_threshold_rules():
    probabilities = np.zeros((3, 6), dtype=np.float32)
    thresholds = np.full(6, 0.5, dtype=np.float32)
    probabilities[:, 5] = [0.4, 0.5, 0.6]
    probabilities[:, 1] = [0.4, 0.5, 0.6]
    probabilities[:, 4] = [0.6, 0.5, 0.4]
    isrj, saw, vft = lazy_candidate_masks(probabilities, thresholds)
    assert isrj.tolist() == [True, False, False]
    assert saw.tolist() == [False, True, True]
    assert vft.tolist() == [True, True, False]
