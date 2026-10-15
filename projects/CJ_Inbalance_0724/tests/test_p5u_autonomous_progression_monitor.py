import json
from pathlib import Path

from cj0724.p5u_autonomous_progression_monitor import run_monitor


def test_p5u_monitor_once_waits_without_generating_confirmation_data(tmp_path):
    project = Path(__file__).resolve().parents[1]
    source = project / "configs" / "p5u_autonomous_progression_monitor_20260806.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["development_execution_root"] = str(tmp_path / "no-development-result")
    manifest["monitor_root"] = str(tmp_path / "monitor")
    path = tmp_path / "monitor.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert run_monitor(project, path, poll_seconds=0.0, once=True) == 0
    receipt = json.loads((tmp_path / "monitor" / "monitor_started.json").read_text(encoding="utf-8"))
    assert receipt["cuda_opened"] is False
    assert not any((project / root).exists() for root in manifest["confirmation_data_roots"])
