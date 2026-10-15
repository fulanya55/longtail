from experiments.audit_route_a_ipcve_stage_minus1 import (
    CONFIRMATION_BASE_SEED,
    DEV_BASE_SEED,
    _mechanism_matrix,
    _reserved_range,
)


def test_closed_counterfactual_lines_are_not_reusable():
    rows = {row["mechanism"]: row for row in _mechanism_matrix()}
    assert rows["H0e_CVMC"]["formal_status"] == "fail_closed"
    assert rows["H0f_CDP"]["formal_status"] == "fail_closed"
    assert rows["H0v_NaturalVisibilityGraph"]["formal_status"] == "fail_closed"
    assert rows["H0e_CVMC"]["reuse_allowed"] is False
    assert rows["H0f_CDP"]["reuse_allowed"] is False
    assert rows["H0v_NaturalVisibilityGraph"]["reuse_allowed"] is False


def test_only_p0_is_conditionally_admissible_before_new_evidence():
    rows = {row["mechanism"]: row for row in _mechanism_matrix()}
    assert rows["IP_CVE_P0"]["reuse_allowed"] is True
    assert rows["IP_CVE_P1"]["reuse_allowed"] is False
    assert "not_authorized_until_P0" in rows["IP_CVE_P1"]["formal_status"]


def test_reserved_identity_ranges_are_disjoint_and_uint32_safe():
    development = _reserved_range(DEV_BASE_SEED)
    confirmation = _reserved_range(CONFIRMATION_BASE_SEED)
    assert development[1] < confirmation[0]
    assert confirmation[1] < 2**32
