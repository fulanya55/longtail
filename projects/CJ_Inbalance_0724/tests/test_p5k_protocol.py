import json
from pathlib import Path


def test_p5k_protocol_freezes_prior_execution_artifacts_and_compute_guard_contract():
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / "configs/p5k_shared_dense_compute_guard_preregister_20260803.json").read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    for key in ("P5e_confirmation_result", "P5g_preregister", "P5g_outcome_freeze", "P5h_preregister", "P5h_outcome_freeze", "P5i_preregister", "P5i_outcome_freeze", "P5j_preregister", "P5j_outcome_freeze"):
        assert key in parent and f"{key}_sha256" in parent
    assert protocol["execution"]["warmup_runs"] == 5
    assert protocol["execution"]["timed_repetitions"] == 15
    assert "pmon" in protocol["execution"]["timing"]
