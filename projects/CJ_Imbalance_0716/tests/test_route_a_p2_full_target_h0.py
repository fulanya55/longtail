from __future__ import annotations

import numpy as np

from experiments.generate_route_a_p1_observability_adequate_pairs import _scale
from experiments.generate_route_a_p2_full_target_pairs import _isrj_high_replica
from experiments.run_route_a_p2_full_target_h0 import _block_shuffle, _isrj_joint_periodicity_score


def test_p2_high_replica_periodicity_exceeds_its_block_shuffle_control_without_noise() -> None:
    signal = _scale(_isrj_high_replica(29), 8.0)
    assert _isrj_joint_periodicity_score(signal) > _isrj_joint_periodicity_score(_block_shuffle(signal))


def test_p2_block_shuffle_preserves_values_but_changes_order() -> None:
    x = np.arange(30_000)
    shuffled = _block_shuffle(x)
    assert np.array_equal(np.sort(shuffled), x)
    assert not np.array_equal(shuffled, x)
