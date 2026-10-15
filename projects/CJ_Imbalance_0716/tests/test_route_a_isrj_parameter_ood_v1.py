from experiments.audit_route_a_isrj_parameter_ood_v1 import REFERENCES

def test_reserved_identity_contracts_do_not_overlap_each_other():
    rows=list(REFERENCES.values())
    for index,a in enumerate(rows):
        for b in rows[index+1:]:
            assert a[1] < b[0] or a[0] > b[1]
