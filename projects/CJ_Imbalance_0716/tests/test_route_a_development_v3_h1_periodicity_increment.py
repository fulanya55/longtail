import numpy as np
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features

def test_h1_periodicity_features_have_expected_four_columns():
    logits=np.zeros((5,6),dtype=np.float32); z=np.arange(5,dtype=np.float32)
    assert _fusion_features(logits,z).shape==(5,4)
