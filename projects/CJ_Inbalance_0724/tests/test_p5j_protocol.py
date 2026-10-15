import json
from pathlib import Path


def test_p5j_protocol_includes_all_prior_frozen_execution_artifacts():
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / "configs/p5j_shared_dense_exclusive_gpu_preregister_20260803.json").read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    for key in ("P5e_confirmation_result", "P5g_preregister", "P5g_outcome_freeze", "P5h_preregister", "P5h_outcome_freeze", "P5i_preregister", "P5i_outcome_freeze"):
        assert key in parent and f"{key}_sha256" in parent
    assert protocol["execution"]["warmup_runs"] == 5
    assert protocol["execution"]["timed_repetitions"] == 15
