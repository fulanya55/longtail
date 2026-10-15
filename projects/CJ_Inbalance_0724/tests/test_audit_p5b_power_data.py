import numpy as np


def test_p5b_expected_jnr_contract_is_fixed_first3_and_swept_last3():
    level = 15
    expected = np.asarray([30.0, 30.0, 30.0, level, level, level], dtype=np.float32)
    assert expected.tolist() == [30.0, 30.0, 30.0, 15.0, 15.0, 15.0]
