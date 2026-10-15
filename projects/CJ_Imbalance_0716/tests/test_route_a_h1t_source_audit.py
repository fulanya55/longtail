import numpy as np

from experiments.audit_route_a_h1t_source import make_roles


def test_h1t_roles_are_disjoint_and_composition_stratified():
    labels = np.array([[int((i >> bit) & 1) for bit in range(6)] for i in range(63) for _ in range(128)], dtype=np.uint8)
    roles = make_roles(labels, 2101)
    assert {name: len(values) for name, values in roles.items()} == {"fit": 4032, "validation": 1008, "early": 3024}
    assert len(np.unique(np.concatenate(list(roles.values())))) == len(labels)
    for role, indices in roles.items():
        packed = np.packbits(labels[indices], axis=1, bitorder="little").reshape(-1)
        expected = {"fit": 64, "validation": 16, "early": 48}[role]
        assert all((packed == key).sum() == expected for key in range(63))
