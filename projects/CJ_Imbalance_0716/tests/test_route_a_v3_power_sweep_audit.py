import hashlib
import numpy as np

from experiments.audit_route_a_v3_power_sweep import _array_sha


def test_array_sha_is_dtype_stable_and_sensitive():
    values = np.array([1, 2, 3], dtype=np.int64)
    expected = hashlib.sha256(values.astype("<i8").tobytes()).hexdigest()
    assert _array_sha(values, "<i8") == expected
    assert _array_sha(values + 1, "<i8") != expected

