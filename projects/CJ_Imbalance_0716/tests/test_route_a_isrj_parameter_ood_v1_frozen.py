import numpy as np
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration

def test_frozen_calibration_uses_saved_affine_parameters():
    logits=np.zeros((3,2),dtype=np.float32); payload={"parameters":[{"coefficient":1,"intercept":0},{"coefficient":2,"intercept":0}]}
    assert np.allclose(_apply_frozen_calibration(logits,payload),.5)
