import numpy as np

from cj0724.audit_p8b_axis_stratified_longtail import inspect


def test_p8b_audit_detects_overlap_and_exact_combination_contract():
    combos = np.asarray([[int(bit) for bit in format(value, "06b")] for value in range(1, 64)], dtype=np.uint8)
    labels = np.repeat(combos, 400, axis=0)
    fresh = np.arange(len(labels), dtype=np.int64) + 1000
    clean = inspect(fresh, labels, np.arange(1000, dtype=np.int64))
    assert clean["identity_overlap_with_p8a"] == 0
    assert clean["all_63_combinations_have_400"]
    leaked = inspect(fresh, labels, np.asarray([1000, 999999], dtype=np.int64))
    assert leaked["identity_overlap_with_p8a"] == 1
