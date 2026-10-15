import numpy as np

from experiments.audit_route_a_dual_axis_sawfm_r1_data import build_roles, validate_roles


def test_role_builder_is_complete_disjoint_and_deterministic():
    labels = np.zeros((800, 6), dtype=np.uint8)
    labels[:400, 0] = 1
    labels[400:, 1] = 1
    first = build_roles(labels)
    second = build_roles(labels)
    audit = validate_roles(first, 800)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert audit["counts"] == {
        "optimization": 480,
        "early": 80,
        "calibration": 80,
        "threshold": 80,
        "evaluation": 80,
    }
    assert audit["covers_all_identities"] is True
    assert audit["all_pairwise_disjoint"] is True
