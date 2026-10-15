import json
from pathlib import Path

from cj0724.p5m_authorized_window_shared_dense_execution import fixed_benchmark_indices, restore_default_if_changed


def test_p5m_protocol_keeps_p5l_timing_contract_and_uses_fresh_uint32_seeds():
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / "configs/p5m_authorized_window_shared_dense_preregister_20260803.json").read_text(encoding="utf-8"))
    assert protocol["execution"]["batch_size"] == 126
    assert protocol["execution"]["warmup_runs"] == 5
    assert protocol["execution"]["timed_repetitions"] == 15
    all_seeds = protocol["development_data"]["generator_seeds"] + protocol["confirmation_if_and_only_if_development_passes"]["generator_seeds"]
    assert len(all_seeds) == len(set(all_seeds))
    assert all(0 <= seed <= 2**32 - 1 for seed in all_seeds)
    assert fixed_benchmark_indices(3780, 126).tolist() == list(range(126))


def test_p5m_does_not_restore_a_mode_that_was_not_changed(monkeypatch):
    def forbidden(_args):
        raise AssertionError("restore should not run")
    monkeypatch.setattr("cj0724.p5m_authorized_window_shared_dense_execution._command", forbidden)
    assert restore_default_if_changed({"mode_attempt": {"returncode": 1}})["skipped"]
