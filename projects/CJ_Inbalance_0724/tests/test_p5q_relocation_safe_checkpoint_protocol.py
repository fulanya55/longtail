import json
import os
import subprocess
import sys
from pathlib import Path


def test_p5q_protocol_is_relocation_safe_and_checkpoint_recovery_is_cuda_free(tmp_path):
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/p5q_relocation_safe_checkpoint_preregister_20260805.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    common = [
        sys.executable, "-m", "cj0724.p5q_relocation_safe_checkpoint_execution",
        "--preregister", str(protocol_path),
        "--p5e-freeze", str(project / protocol["base_candidate"]["freeze_path"]),
    ]
    unrelated_cwd = tmp_path / "unrelated" / "cwd"
    unrelated_cwd.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project)
    validated = subprocess.run([*common, "--validate-protocol"], cwd=unrelated_cwd, env=environment, capture_output=True, text=True, check=True)
    payload = json.loads(validated.stdout)
    assert payload["protocol_valid"] is True
    assert Path(payload["project_root"]) == project
    receipt_root = tmp_path / "receipts"
    tested = subprocess.run([*common, "--self-test-checkpoints", "--receipt-root", str(receipt_root)], cwd=unrelated_cwd, env=environment, capture_output=True, text=True, check=True)
    inspection = json.loads(tested.stdout)["inspection"]
    assert inspection["analytic_roundtrip_exact"] is True
    assert inspection["classification"] == "recoverable_unobserved_unit"
    assert inspection["data_opened"] is False
    assert inspection["cuda_opened"] is False
    assert (receipt_root / "pool_000_analytic.npz").is_file()
    assert (receipt_root / "pool_000_model_3041_started.json").is_file()
    assert not (receipt_root / "pool_000_model_3041_completed.json").exists()
