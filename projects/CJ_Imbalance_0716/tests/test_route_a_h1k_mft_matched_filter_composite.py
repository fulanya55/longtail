import numpy as np
from experiments.run_route_a_h1k_mft_matched_filter_composite import _bootstrap,_matched


def test_matched_keeps_only_shared_other_label_contexts_and_bootstrap_is_paired():
    labels=np.asarray([[0,0,0,1,0,0],[0,0,0,0,0,0],[1,0,0,1,0,0],[1,0,0,0,0,0],[0,0,0,1,0,1]],dtype=np.uint8);weak=np.zeros_like(labels,dtype=bool);weak[[0,2],3]=True
    keep,y,counts=_matched(labels,weak)
    assert keep.sum()==4 and y.sum()==2 and counts['matched_context_count']==2
    result=_bootstrap(np.asarray([0,0,0,1,1,1]),np.asarray([.1,.2,.3,.7,.8,.9]),np.asarray([.4,.9,.1,.8,.2,.6]),draws=100,seed=7)
    assert result['auroc']['point']>0 and result['aupr']['point']>0
