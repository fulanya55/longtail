from experiments.run_route_a_final_shared_policy_isrj_ood_frozen import SHARED_VARIANT


def test_external_candidate_is_shared_policy_variant():
    assert "SharedMFTDecision" in SHARED_VARIANT
