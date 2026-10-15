import experiments.analyze_route_a_isrj_f3_stratified_robustness_v3 as base

def test_stratification_variant_contract_can_target_h1_outputs():
    original=base.VARIANTS
    try:
        base.VARIANTS={"F0":"B0_H1","F3":"B2_H1AlignedPeriodicity","F4":"B3_H1BlockShuffled"}
        assert base.VARIANTS["F3"].startswith("B2_")
    finally:
        base.VARIANTS=original
