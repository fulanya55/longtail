from experiments.run_route_a_m5_periodicity_unified_v3 import VARIANTS


def test_candidate_and_controls_are_fixed():
    assert VARIANTS == ("Q0_C6RawSaw", "Q1_M5RawSaw", "Q2_M5AlignedPeriodicity", "Q3_M5BlockShuffled")
