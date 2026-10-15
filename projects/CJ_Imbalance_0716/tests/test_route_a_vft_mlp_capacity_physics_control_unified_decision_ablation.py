import numpy as np
from experiments.run_route_a_vft_mlp_capacity_physics_control_unified_decision_ablation import ARMS, _base_only, _permute

def test_three_physics_control_arms_are_fixed():
    assert ARMS == ("MLP1_NoVTSECapacity", "MLP1_AlignedVTSE", "MLP1_ShuffledVTSE")

def test_base_only_contains_no_expert_score():
    logits=np.zeros((1,6),dtype=np.float32); logits[0,4]=2.0
    assert np.allclose(_base_only(logits), [[2.0,np.tanh(2.0),4.0,2.0]])

def test_shuffling_is_seed_deterministic():
    a,ha=_permute(np.arange(8,dtype=np.float32),7); b,hb=_permute(np.arange(8,dtype=np.float32),7)
    assert np.array_equal(a,b) and ha==hb
