from __future__ import annotations

import numpy as np

from experiments.run_route_a_h0v_visibility_graph_observability import NODE_COUNT, _paired_rows, degree_entropy


def test_natural_visibility_degree_entropy_has_known_three_node_cases() -> None:
    rising = np.linspace(0.0, 1.0, NODE_COUNT) ** 2
    valley = np.tile(np.asarray((0.0, 1.0, 0.0)), NODE_COUNT // 3 + 1)[:NODE_COUNT]
    assert degree_entropy(rising) == 0.0
    assert 0.0 < degree_entropy(valley) <= 1.0


def test_natural_visibility_degree_entropy_rejects_wrong_node_count() -> None:
    try:
        degree_entropy(np.zeros(NODE_COUNT - 1))
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("wrong H0v node count was accepted")


def test_pair_contract_rejects_mismatched_receiver_identity() -> None:
    pair = np.asarray((1, 1)); target = np.asarray((3, 3)); side = np.asarray((0, 1))
    labels = np.zeros((2, 6), dtype=np.uint8); labels[1, 3] = 1
    try:
        _paired_rows(pair, target, side, labels, np.asarray((10, 11)), np.asarray((2, 2)))
    except ValueError as error:
        assert "identity contract" in str(error)
    else:
        raise AssertionError("mismatched receiver identity was accepted")
