import numpy as np

from experiments.run_route_a_dual_axis_sawfm_r2_formal import (
    CLASS_NAMES,
    weak_mask,
)


def test_weak_mask_is_sawfm_only_in_low_view_compounds():
    labels = np.asarray(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    low = weak_mask(labels, "VL")
    high = weak_mask(labels, "VH")
    assert not low[0].any()
    assert low[1, CLASS_NAMES.index("SawFM")]
    assert low.sum() == 1
    assert not high.any()


def test_unknown_visibility_is_rejected():
    try:
        weak_mask(np.zeros((2, 6), dtype=np.uint8), "unknown")
    except ValueError as error:
        assert "unknown visibility" in str(error)
    else:
        raise AssertionError("unknown visibility was accepted")
