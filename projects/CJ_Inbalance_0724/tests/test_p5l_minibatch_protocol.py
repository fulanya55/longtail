import json
from pathlib import Path

import numpy as np
import pytest

from cj0724.p5l_minibatch_shared_dense_execution import fixed_benchmark_indices


def test_p5l_protocol_freezes_p5k_and_pre_registers_fixed_batch_cost_unit():
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / "configs/p5l_minibatch_shared_dense_preregister_20260803.json").read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    for key in ("P5e_confirmation_result", "P5e_candidate_freeze", "P5g_preregister", "P5g_outcome_freeze", "P5h_preregister", "P5h_outcome_freeze", "P5i_preregister", "P5i_outcome_freeze", "P5j_preregister", "P5j_outcome_freeze", "P5k_preregister", "P5k_outcome_freeze"):
        assert key in parent and f"{key}_sha256" in parent
    assert protocol["execution"]["batch_size"] == 126
    assert protocol["execution"]["benchmark_batch"].startswith("first complete")
    all_seeds = protocol["development_data"]["generator_seeds"] + protocol["confirmation_if_and_only_if_development_passes"]["generator_seeds"]
    assert len(all_seeds) == len(set(all_seeds))
    assert all(0 <= seed <= 2**32 - 1 for seed in all_seeds)


def test_p5l_fixed_benchmark_indices_are_ordered_and_data_independent():
    np.testing.assert_array_equal(fixed_benchmark_indices(3780, 126), np.arange(126))
    with pytest.raises(ValueError):
        fixed_benchmark_indices(125, 126)
