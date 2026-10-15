import numpy as np

from cj0724.p3l_p1n_two_identity_confirmation import paired_bootstrap


def test_paired_bootstrap_is_deterministic_and_positive():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    first = paired_bootstrap(values, 7, 1000)
    second = paired_bootstrap(values, 7, 1000)
    assert first == second
    assert first["estimate"] > 0
    assert first["ci_lower"] > 0
