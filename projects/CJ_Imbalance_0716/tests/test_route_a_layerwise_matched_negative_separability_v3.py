import numpy as np

from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view


def test_matched_view_keeps_only_weak_positive_and_same_other_label_negative():
    targets=np.asarray([
        [1,0,0,1,0,0],
        [1,0,0,0,0,0],
        [0,0,0,1,0,0],
        [1,1,0,0,0,0],
        [1,1,0,1,0,0],
    ],dtype=np.uint8)
    weak=np.zeros_like(targets,dtype=bool); weak[0,3]=True; weak[4,3]=True
    selected,labels,receipt=_matched_view(targets,weak,3)
    np.testing.assert_array_equal(np.flatnonzero(selected),[0,1,3,4])
    np.testing.assert_array_equal(labels,[1,0,0,1])
    assert receipt=={"original_weak_positive_count":2,"weak_positive_count":2,
                     "matched_negative_count":2,"positive_composition_count":2,
                     "matched_composition_count":2}


def test_matched_view_drops_positive_only_compositions():
    targets=np.asarray([[0,0,0,1,0,0],[0,0,0,0,0,0],[1,1,0,1,0,0]],dtype=np.uint8)
    weak=np.zeros_like(targets,dtype=bool);weak[[0,2],3]=True
    selected,labels,receipt=_matched_view(targets,weak,3)
    np.testing.assert_array_equal(np.flatnonzero(selected),[0,1])
    np.testing.assert_array_equal(labels,[1,0])
    assert receipt["original_weak_positive_count"]==2
    assert receipt["weak_positive_count"]==1
