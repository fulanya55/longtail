import numpy as np

from experiments.audit_route_a_mft_expert_development_v8_identifiability import _auc


def test_auc_uses_weak_positive_and_selected_negative_only():
    weak=np.array([1,0,0,1],dtype=bool); negative=np.array([0,1,1,0],dtype=bool); score=np.array([0.9,0.2,0.1,0.8])
    assert _auc(weak,negative,score)==1.0
