import json
import os
import subprocess
import sys
from pathlib import Path


def test_p5s_chunk_protocol_and_detached_supervisor_are_data_cuda_free(tmp_path):
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/p5s_detached_chunked_preregister_20260805.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    environment = os.environ.copy(); environment["PYTHONPATH"] = str(project)
    unrelated = tmp_path / "unrelated"; unrelated.mkdir()
    common = [sys.executable, "-m", "cj0724.p5s_detached_chunked_execution", "--preregister", str(protocol_path), "--p5e-freeze", str(project / protocol["base_candidate"]["freeze_path"])]
    assert json.loads(subprocess.run([*common, "--validate-protocol"], cwd=unrelated, env=environment, capture_output=True, text=True, check=True).stdout)["protocol_valid"] is True
    receipt_root = tmp_path / "chunk_receipts"
    self_test = json.loads(subprocess.run([*common, "--self-test-chunks", "--receipt-root", str(receipt_root)], cwd=unrelated, env=environment, capture_output=True, text=True, check=True).stdout)
    assert self_test["inspection"]["chunk_roundtrip_exact"] is True
    assert self_test["inspection"]["data_opened"] is False and self_test["inspection"]["cuda_opened"] is False

    execution_root = tmp_path / "supervised"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "P5s-detached-supervisor-v1", "working_directory": str(project), "execution_root": str(execution_root), "command": [sys.executable, "-c", "raise SystemExit(0)"]}), encoding="utf-8")
    completed = subprocess.run([sys.executable, "-m", "cj0724.p5s_detached_supervisor", "--manifest", str(manifest)], cwd=unrelated, env=environment, capture_output=True, text=True)
    assert completed.returncode == 0
    terminal = json.loads((execution_root / "supervisor_terminal.json").read_text(encoding="utf-8"))
    assert terminal["returncode"] == 0
    assert (execution_root / "supervisor_child_started.json").is_file()
