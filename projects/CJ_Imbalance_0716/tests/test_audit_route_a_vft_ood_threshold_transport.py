import numpy as np
from experiments.audit_route_a_vft_ood_threshold_transport import _oracle_threshold

def test_oracle_threshold_respects_negative_fpr_with_ties():
    probability=np.array([.9,.8,.8,.1],dtype=np.float64);targets=np.array([0,0,0,1],dtype=np.uint8)
    threshold=_oracle_threshold(probability,targets,.20)
    assert ((probability[targets==0]>=threshold).mean()) <= .20
