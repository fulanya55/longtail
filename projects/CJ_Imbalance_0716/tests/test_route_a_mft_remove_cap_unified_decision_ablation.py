from experiments.run_route_a_mft_remove_cap_unified_decision_ablation import ARMS, EVIDENCE_KEY, SOURCE_Q1, SOURCE_Q4


def test_mft_cap_arms_are_fixed():
    assert ARMS == ("Q4_Cap075", "Q4_NaturalCap100")
    assert SOURCE_Q1 == "Q1_H1B2"
    assert SOURCE_Q4 == "Q4_VFTAlignedRidgeFusion"
    assert EVIDENCE_KEY == "K1_AlignedPeakNegativeGuard"
