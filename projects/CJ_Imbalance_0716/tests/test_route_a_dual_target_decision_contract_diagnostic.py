import numpy as np

from experiments.run_route_a_dual_target_decision_contract_diagnostic import MFT_INDEX, VFT_INDEX, contract_logits


def test_contract_caps_mft_and_vft_residuals():
    q1 = np.zeros((4, 6), dtype=np.float32)
    q2 = q1.copy(); q2[:, MFT_INDEX] = -0.75
    expert = q2.copy(); expert[:, VFT_INDEX] = 100.0
    output = contract_logits(q1, q2, expert)
    assert np.allclose(output[:, MFT_INDEX], -0.5)
    assert np.all(output[:, VFT_INDEX] <= 0.5 + 1e-6)
    assert np.all(output[:, VFT_INDEX] >= -0.5 - 1e-6)
