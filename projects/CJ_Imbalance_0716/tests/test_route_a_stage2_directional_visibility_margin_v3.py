import numpy as np

from experiments.run_route_a_stage2_directional_visibility_margin_v3 import _matched_hard_negative_pairs


def test_matched_pairs_only_differ_at_target_label():
    labels=[]
    for mask in range(1,64): labels.append([(mask>>i)&1 for i in range(6)])
    labels=np.repeat(np.asarray(labels,dtype=np.uint8),2,axis=0)
    pairs=_matched_hard_negative_pairs(labels,2039)
    assert len(pairs["positive"])>0
    for pos,neg,local in zip(pairs["positive"],pairs["negative"],pairs["local"]):
        target=(3,4,5)[int(local)]
        assert labels[pos,target]==1 and labels[neg,target]==0
        other=[i for i in range(6) if i!=target]
        assert np.array_equal(labels[pos,other],labels[neg,other])

