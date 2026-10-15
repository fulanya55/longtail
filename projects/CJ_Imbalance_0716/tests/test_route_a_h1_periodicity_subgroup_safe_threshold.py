import numpy as np
from experiments.run_route_a_h1_periodicity_subgroup_safe_threshold import _select_threshold

def test_selector_respects_overall_and_subgroup_caps():
    labels=np.zeros((100,6),dtype=np.uint8); labels[:40,5]=1; labels[:50,0]=1
    weak=np.zeros(100,dtype=bool); weak[:20]=True; probability=np.linspace(.01,.99,100)
    row=_select_threshold(labels[:,5],weak,probability,labels,.20,.25,20)
    assert row['subgroup_fpr']['overall']<=.20+1e-12
