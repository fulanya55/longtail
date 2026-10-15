from experiments.audit_route_a_dual_axis_next_stage0 import build_audit


def test_stage0_selects_route_r_without_deployable_second_observation():
    audit = build_audit()
    assert audit["decision"] == "no_admissible_non_oracle_observation"
    assert audit["selected_branch"] == "R_sawfm_scoped_dual_axis"
    assert audit["checks"]["admissible_non_oracle_second_observation_found"] is False


def test_stage0_preserves_all_test_boundaries():
    audit = build_audit()
    assert audit["test_v3_seal"]["clear_for_future_one_time_use"] is True
    assert audit["test_v3_seal"]["sample_label_prediction_metric_files_opened"] is False
    assert audit["locked_test_used"] is False
    assert audit["test_v2_used"] is False
    assert audit["test_v3_used"] is False


def test_stage0_uses_authoritative_b0_compute_value():
    audit = build_audit()
    b0 = audit["authoritative_baselines"]["B0"]
    assert b0["conv_linear_macs_per_sample"] == 211857536
    assert audit["checks"]["b0_authoritative_macs_equal_211857536"] is True
