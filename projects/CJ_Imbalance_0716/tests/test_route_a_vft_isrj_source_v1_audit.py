import numpy as np

from experiments.audit_route_a_vft_isrj_source_v1 import _combination_code, _roles_for_seed


def test_source_roles_split_each_combination_without_overlap():
    combinations = np.array([[int((code >> bit) & 1) for bit in range(6)] for code in range(1, 64)], dtype=np.uint8)
    labels = np.repeat(combinations, 80, axis=0)
    roles = _roles_for_seed(labels, 2039)
    assert {name: len(values) for name, values in roles.items()} == {"fit": 3024, "validation": 1008, "early": 1008}
    assert len(np.unique(np.concatenate(list(roles.values())))) == 5040
    codes = _combination_code(labels)
    for indices in roles.values():
        assert set(np.unique(codes[indices], return_counts=True)[1]) == {16 if len(indices) == 1008 else 48}
