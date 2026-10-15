import pytest
import json
from pathlib import Path

from cj0724 import p5h_exclusive_gpu_guard as guard


def test_guard_rejects_other_python_process(monkeypatch):
    monkeypatch.setattr(guard, "_powershell_python_processes", lambda pid: [{"pid": 9, "name": "python.exe", "command_line": "train.py"}])
    monkeypatch.setattr(guard, "_gpu_compute_pids", lambda: {9})
    monkeypatch.setattr(guard, "_nvidia_snapshot", lambda: "gpu")
    with pytest.raises(RuntimeError, match="exclusive-GPU"):
        guard.assert_exclusive_gpu(current_pid=7)


def test_guard_records_empty_external_process_list(monkeypatch):
    monkeypatch.setattr(guard, "_powershell_python_processes", lambda pid: [])
    monkeypatch.setattr(guard, "_gpu_compute_pids", lambda: set())
    monkeypatch.setattr(guard, "_nvidia_snapshot", lambda: "gpu")
    assert guard.assert_exclusive_gpu(current_pid=7) == {"current_pid": 7, "external_python_processes": [], "external_python_processes_observed_but_not_using_gpu": [], "gpu_compute_pids": [], "gpu_snapshot": "gpu"}


def test_conda_activation_helper_is_not_treated_as_compute_work(monkeypatch):
    monkeypatch.setattr(guard.subprocess, "run", lambda *args, **kwargs: type("Done", (), {"stdout": '[{"ProcessId":9,"Name":"python.exe","CommandLine":"D:\\\\anaconda3\\\\python.exe D:\\\\anaconda3\\\\Scripts\\\\conda-script.py shell.powershell activate base"}]'})())
    assert guard._powershell_python_processes(7) == []


def test_conda_hook_helper_is_not_treated_as_compute_work(monkeypatch):
    monkeypatch.setattr(guard.subprocess, "run", lambda *args, **kwargs: type("Done", (), {"stdout": '[{"ProcessId":9,"Name":"python.exe","CommandLine":"D:\\\\anaconda3\\\\python.exe D:\\\\anaconda3\\\\Scripts\\\\conda-script.py shell.powershell hook"}]'})())
    assert guard._powershell_python_processes(7) == []


def test_cpu_only_python_is_recorded_but_does_not_fail_exclusive_gpu_guard(monkeypatch):
    monkeypatch.setattr(guard, "_powershell_python_processes", lambda pid: [{"pid": 9, "name": "python.exe", "command_line": "p3om_data_audit"}])
    monkeypatch.setattr(guard, "_gpu_compute_pids", lambda: set())
    monkeypatch.setattr(guard, "_nvidia_snapshot", lambda: "gpu")
    receipt = guard.assert_exclusive_gpu(current_pid=7)
    assert receipt["external_python_processes"] == []
    assert receipt["external_python_processes_observed_but_not_using_gpu"][0]["pid"] == 9


def test_p5h_protocol_exposes_the_runner_timing_contract():
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / "configs/p5h_shared_dense_exclusive_gpu_preregister_20260802.json").read_text(encoding="utf-8"))
    assert protocol["execution"]["warmup_runs"] == 5
    assert protocol["execution"]["timed_repetitions"] == 15
