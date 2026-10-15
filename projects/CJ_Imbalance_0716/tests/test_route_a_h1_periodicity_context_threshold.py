import numpy as np
from experiments.run_route_a_h1_periodicity_context_threshold import _select_context

def test_context_selector_only_tightens_flagged_samples():
    target=np.r_[np.ones(40),np.zeros(60)].astype(np.uint8); weak=np.r_[np.ones(20),np.zeros(80)].astype(bool); p=np.linspace(.99,.01,100); risk=np.zeros(100,bool); risk[::2]=True
    row=_select_context(target,weak,p,risk,.5,.20,.25)
    assert row['risk_threshold']>=.5
