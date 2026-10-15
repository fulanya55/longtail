from experiments.analyze_route_a_final_shared_policy_stratified_v12 import BASELINE, SHARED_VARIANT


def test_final_stratified_variants_are_distinct():
    assert BASELINE == "Q0_H1"
    assert SHARED_VARIANT != BASELINE
