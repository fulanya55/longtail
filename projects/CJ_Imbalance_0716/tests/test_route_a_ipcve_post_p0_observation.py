from experiments.audit_route_a_ipcve_post_p0_observation import build_audit


def test_post_p0_admission_closes_overlapping_local_families():
    audit = build_audit()
    assert audit["checks"]["all_evidence_artifacts_present"] is True
    assert audit["checks"]["all_evidence_test_flags_clear"] is True
    assert audit["checks"]["all_existing_local_candidate_families_closed"] is True
    assert audit["checks"]["new_local_h0_candidate_admitted"] is False
    assert audit["decision"] == "no_run_close_local_ipcve_mechanism_external_observation_required"


def test_post_p0_audit_never_uses_test_or_trains_model():
    audit = build_audit()
    assert audit["new_data_generated"] is False
    assert audit["model_trained"] is False
    assert audit["locked_test_used"] is False
    assert audit["test_v2_used"] is False
    assert audit["test_v3_used"] is False

