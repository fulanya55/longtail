import json
from pathlib import Path

from cj0724.p5b1_jnr_physical_oracle import sha256_file
from cj0724.p5m_authorized_window_shared_dense_execution import fixed_benchmark_indices


def test_p5n_is_a_fresh_identity_only_successor_of_the_frozen_p5m_contract():
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / "configs/p5n_verified_idle_window_shared_dense_preregister_20260803.json").read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    assert protocol["execution"] == json.loads((project / "configs/p5m_authorized_window_shared_dense_preregister_20260803.json").read_text(encoding="utf-8"))["execution"]
    all_seeds = protocol["development_data"]["generator_seeds"] + protocol["confirmation_if_and_only_if_development_passes"]["generator_seeds"]
    assert len(all_seeds) == len(set(all_seeds))
    assert all(0 <= seed <= 2**32 - 1 for seed in all_seeds)
    assert fixed_benchmark_indices(3780, 126).tolist() == list(range(126))
    for key in ("P5e_candidate_freeze", "P5m_preregister", "P5m_outcome_freeze", "P5m_runner", "P5m_generator", "audit_runner"):
        assert sha256_file(project / parent[key]) == parent[f"{key}_sha256"]
    assert "P3o-U" in protocol["authorized_environment"]["additional_launch_precondition"]
