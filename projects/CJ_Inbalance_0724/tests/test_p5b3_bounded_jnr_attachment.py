import numpy as np
from cj0724.p5b3_bounded_jnr_attachment import TARGET_CLASSES

def test_p5b3_attachment_mask_only_allows_the_three_predicted_positive_jnr_cells():
    decision = np.array([[False, False, False, True, True, True]])
    allowed = np.zeros((1, 6, 2), bool)
    for index in TARGET_CLASSES:
        allowed[:, index, 0] = decision[:, index]
    assert allowed.sum() == 3
    assert not allowed[:, :, 1].any()
