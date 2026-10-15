from experiments.audit_route_a_rawsaw_topk_within_run_protocol_repair import TOPKS

def test_within_run_pooling_values_are_complete():
    assert TOPKS == (4, 8, 16)
