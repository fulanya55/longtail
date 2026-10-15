from experiments.run_route_a_vft_isrj_source_v1_raw_feature_diagnostic import ARMS


def test_source_v1_raw_feature_arms_are_fixed_and_include_phase_control():
    assert ARMS == ("RidgeSTFT", "RawSource", "PhaseScrambledControl")
