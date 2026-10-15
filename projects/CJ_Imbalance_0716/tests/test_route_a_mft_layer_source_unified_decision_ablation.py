from experiments.run_route_a_mft_layer_source_unified_decision_ablation import ARMS, SOURCE

def test_all_declared_layer_sources_are_present():
    assert ARMS == ("P0_Q4_NoLayerExpert", "P1_Global", "P2_FinalTemporal", "P3_Stage2", "P4_Stage3")
    assert tuple(SOURCE.values()) == ("G_Global", "T_FinalTemporal", "S2_Physics", "S3_Physics")
