from experiments.audit_route_a_dual_axis_sawfm_preflight import build_audit, overlap


def test_closed_interval_overlap():
    assert overlap((1, 3), (3, 5)) is True
    assert overlap((1, 2), (3, 5)) is False


def test_preflight_admits_new_paired_identity_range():
    audit = build_audit()
    assert audit["new_identity_range"] == [26080702, 26105901]
    assert audit["paired_view_identity_range_equal"] is True
    assert audit["identity_collisions"] == []
    assert audit["decision"] == "admit_formal_generation"
    assert audit["checks"]["formal_generation_admitted"] is True
    assert audit["test_v3_used"] is False
