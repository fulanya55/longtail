from experiments.audit_route_a_0716_generator_lineage import contains_all


def test_generator_semantic_tokens_require_all_contract_terms():
    assert contains_all("a b c", "a", "b", "c")
    assert not contains_all("a b", "a", "b", "c")
