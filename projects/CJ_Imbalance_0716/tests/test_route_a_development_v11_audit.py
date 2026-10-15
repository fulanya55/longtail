from experiments.audit_route_a_development_v11_dataset import EXPECTED_RECEIVER, COUNT


def test_v11_identity_contract_constants():
    assert EXPECTED_RECEIVER == (312677743, 312702942)
    assert EXPECTED_RECEIVER[1] - EXPECTED_RECEIVER[0] + 1 == COUNT
