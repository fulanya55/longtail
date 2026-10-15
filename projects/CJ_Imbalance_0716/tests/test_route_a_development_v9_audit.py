from experiments.audit_route_a_development_v9_dataset import EXPECTED_RECEIVER, REFERENCES


def test_v9_receiver_range_is_disjoint_from_all_prior_contracts():
    assert all(
        EXPECTED_RECEIVER[1] < value[0] or EXPECTED_RECEIVER[0] > value[1]
        for value in REFERENCES.values()
    )

