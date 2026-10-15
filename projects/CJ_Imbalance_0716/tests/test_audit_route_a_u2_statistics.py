from experiments.audit_route_a_u2_statistics import _holm


def test_holm_is_monotone_in_sorted_p_order():
    adjusted = _holm({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] >= adjusted["a"]
    assert adjusted["c"] >= adjusted["b"]
