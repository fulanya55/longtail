from experiments.analyze_route_a_fullalpha_heterogeneity import assess_ptv_eligibility, signed_summary


def test_signed_summary_counts_paired_seed_signs() -> None:
    summary = signed_summary([-0.01, 0.0, 0.02])
    assert summary["positive_seed_count"] == 1
    assert summary["nonnegative_seed_count"] == 2
    assert summary["seed_count"] == 3


def test_ptv_gate_cannot_be_met_by_non_target_stratified_metrics() -> None:
    result = assess_ptv_eligibility({"sources": []})
    assert result["decision"] == "STOP_PTV_DIRECTION"
    assert result["eligible_for_trb_ptv_preregistration"] is False
