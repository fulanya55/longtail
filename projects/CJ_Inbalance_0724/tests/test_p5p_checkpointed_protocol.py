import json
import subprocess
import sys
from pathlib import Path


def test_p5p_protocol_and_receipt_recovery_are_data_and_cuda_free(tmp_path):
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/p5p_checkpointed_idle_window_preregister_20260804.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    common = [sys.executable, "-m", "cj0724.p5p_checkpointed_idle_window_execution", "--preregister", str(protocol_path.relative_to(project)), "--p5e-freeze", protocol["base_candidate"]["freeze_path"]]
    validated = subprocess.run([*common, "--validate-protocol"], cwd=project, capture_output=True, text=True, check=True)
    assert json.loads(validated.stdout)["protocol_valid"] is True
    receipt_root = tmp_path / "receipts"
    tested = subprocess.run([*common, "--self-test-receipts", "--receipt-root", str(receipt_root)], cwd=project, capture_output=True, text=True, check=True)
    payload = json.loads(tested.stdout)
    assert payload["receipt_self_test_valid"] is True
    assert payload["inspection"]["classification"] == "interrupted_before_pool_completion_no_resume"
    assert (receipt_root / "launch_receipt.json").is_file()
    assert (receipt_root / "pool_000_started.json").is_file()
    assert not (receipt_root / "pool_000_completed.json").exists()
    assert (receipt_root / "recovery_validation.json").is_file()
