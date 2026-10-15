from experiments.audit_route_a_core_14_ablation_coverage import build


def test_core_14_coverage_audit_is_boundary_safe_and_complete():
    payload = build()
    assert len(payload["entries"]) == 14
    assert payload["test_v3_used"] is False
    assert payload["all_14_direct_current_protocol"] is True
    assert payload["partial_or_missing"] == []
