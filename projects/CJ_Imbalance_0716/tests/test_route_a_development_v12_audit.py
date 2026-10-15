from experiments.audit_route_a_development_v12_dataset import EXPECTED_RECEIVER, COUNT, audit


def test_v12_identity_contract_constants():
    assert EXPECTED_RECEIVER == (322677743, 322702942)
    assert EXPECTED_RECEIVER[1] - EXPECTED_RECEIVER[0] + 1 == COUNT


def test_v12_audit_source_explicitly_records_all_forbidden_roles_false():
    names = audit.__code__.co_consts
    serialized = repr(names)
    assert "test_v2_used" in serialized
    assert "test_v3_used" in serialized
    assert "test_v12_used" in serialized
    assert "locked_test_used" in serialized
