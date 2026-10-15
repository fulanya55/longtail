from experiments.audit_route_a_final_shared_policy_statistics import _holm


def test_holm_is_monotone_in_ordered_family():
    adjusted = _holm({"a": 0.001, "b": 0.01, "c": 0.04})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert adjusted["a"] == 0.003
