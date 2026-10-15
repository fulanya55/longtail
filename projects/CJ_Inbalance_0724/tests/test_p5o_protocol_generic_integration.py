import json
import subprocess
import sys
from pathlib import Path


def test_p5o_protocol_validation_executes_parent_hash_checks_without_cuda_or_data():
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/p5o_protocol_generic_idle_window_preregister_20260803.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    command = [sys.executable, "-m", "cj0724.p5o_protocol_generic_idle_window_execution", "--preregister", str(protocol_path.relative_to(project)), "--p5e-freeze", protocol["base_candidate"]["freeze_path"], "--validate-protocol"]
    completed = subprocess.run(command, cwd=project, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload["protocol_valid"] is True
    assert len(payload["validated_artifacts"]) == len(protocol["frozen_artifacts"])
