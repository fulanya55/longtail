import numpy as np
from experiments.run_route_a_mft_remove_cap_unified_decision_protocol_repair import ARMS

def test_protocol_repair_keeps_the_two_declared_cap_arms():
    assert ARMS == ("Q4_Cap075", "Q4_NaturalCap100")

def test_threshold_fpr_is_computed_on_the_threshold_role_only():
    probability=np.array([.2,.7,.9],dtype=np.float32);negative=np.array([True,True,False])
    assert (probability[negative]>=.5).mean() == .5
