from experiments.audit_route_a_final_h1_b2_statistics import _audit, _holm


def test_paired_effect_and_holm_are_deterministic():
    row = _audit([1.0, 2.0, 3.0, 4.0, 5.0])
    assert row["favorable_seed_count"] == 5
    assert row["paired_cohen_dz"] > 1.0
    adjusted = _holm({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] >= adjusted["a"]

