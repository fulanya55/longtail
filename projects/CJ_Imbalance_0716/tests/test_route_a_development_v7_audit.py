from experiments.audit_route_a_development_v7_dataset import EXPECTED_RECEIVER, REFERENCES


def test_v7_receiver_contract_is_disjoint():
    assert all(EXPECTED_RECEIVER[1] < low or EXPECTED_RECEIVER[0] > high for low, high in REFERENCES.values())

