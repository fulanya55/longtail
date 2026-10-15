import numpy as np

from cj0724.audit_fresh_multitask_data import array_sha256


def test_array_sha256_is_shape_and_value_stable():
    values = np.array([1, 2, 3], dtype=np.int64)
    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values) != array_sha256(values + 1)
