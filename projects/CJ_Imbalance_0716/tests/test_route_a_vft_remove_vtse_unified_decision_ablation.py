from experiments.run_route_a_vft_remove_vtse_unified_decision_ablation import ARMS, SOURCE


def test_vtse_removal_arms_are_fixed():
    assert ARMS == ("Q2_NoVTSE", "Q3_BaseOnlyHead", "Q4_AlignedVTSE")
    assert SOURCE["Q2_NoVTSE"] == "Q2_H1B2_MFTGuard"
    assert SOURCE["Q4_AlignedVTSE"] == "Q4_VFTAlignedRidgeFusion"
