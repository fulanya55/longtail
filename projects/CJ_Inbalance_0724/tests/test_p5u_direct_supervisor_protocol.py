import json
import os
import subprocess
import sys
from pathlib import Path


def test_p5u_supervisor_binds_terminal_to_direct_child_after_preflight(tmp_path):
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / "execution"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "P5u-preflight-supervisor-v1", "working_directory": str(project), "execution_root": str(root),
        "preflight_command": [sys.executable, "-c", "raise SystemExit(0)"],
        "child_command": [sys.executable, "-c", "raise SystemExit(0)"],
    }), encoding="utf-8")
    environment = os.environ.copy(); environment["PYTHONPATH"] = str(project)
    result = subprocess.run([sys.executable, "-m", "cj0724.p5u_preflight_supervisor", "--manifest", str(manifest)], cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0
    preflight = json.loads((root / "supervisor_preflight.json").read_text(encoding="utf-8"))
    child = json.loads((root / "supervisor_child_started.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "supervisor_terminal.json").read_text(encoding="utf-8"))
    assert preflight["returncode"] == 0
    assert terminal["stage"] == "child_terminal" and terminal["returncode"] == 0
    assert terminal["child_pid"] == child["child_pid"]
