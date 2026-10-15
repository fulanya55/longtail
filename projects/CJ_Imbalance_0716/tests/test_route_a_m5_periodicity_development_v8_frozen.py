import numpy as np

from experiments.run_route_a_m5_periodicity_development_v8_frozen import _order_fpr


def test_order_fpr_uses_only_target_absent_samples():
    labels = np.array([[1,0,0,0,0,0],[1,1,1,0,0,0],[1,1,1,1,1,0]],dtype=np.uint8)
    prediction = np.zeros_like(labels)
    prediction[:,3] = 1
    result = _order_fpr(labels,prediction)
    assert result["MFT"]["1"]["fpr"] == 1.0
    assert result["MFT"]["3"]["fpr"] == 1.0
    assert "5" not in result["MFT"]
