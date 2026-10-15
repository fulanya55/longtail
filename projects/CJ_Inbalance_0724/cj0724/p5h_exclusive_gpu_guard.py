"""P5h process-level exclusive-GPU guard for timing claims.

Windows user-process enumeration is used because `nvidia-smi` may hide process
names behind insufficient-permission placeholders.  The guard intentionally
rejects any other live Python process, a conservative proxy for competing
training/inference work on the single experiment GPU.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _powershell_python_processes(current_pid: int) -> list[dict[str, object]]:
    script = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(\\.exe)?$' } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script], check=True,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    import json
    raw = completed.stdout.strip()
    rows = [] if not raw else json.loads(raw)
    if isinstance(rows, dict):
        rows = [rows]
    def is_noncomputing_conda_shell(command_line: str) -> bool:
        lower = command_line.lower()
        return (
            "conda-script.py shell.powershell activate" in lower
            or "conda-script.py shell.powershell hook" in lower
        )

    return [
        {"pid": int(row["ProcessId"]), "name": str(row["Name"]), "command_line": str(row.get("CommandLine") or "")}
        for row in rows
        if int(row["ProcessId"]) != int(current_pid)
        and not is_noncomputing_conda_shell(str(row.get("CommandLine") or ""))
    ]


def _nvidia_snapshot() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return completed.stdout.strip()


def _gpu_compute_pids() -> set[int]:
    """Return CUDA/compute PIDs from the driver process monitor.

    `--query-compute-apps` is unreliable on this Windows driver when desktop
    graphics applications are present.  `pmon` exposes the process type; only
    C/C+G Python processes can perturb this CUDA inference benchmark.
    """
    completed = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1"], check=True,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    result: set[int] = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1].isdigit() and "C" in fields[2]:
            result.add(int(fields[1]))
    return result


def assert_exclusive_gpu(current_pid: int | None = None) -> dict[str, object]:
    """Fail only when another Python process holds a GPU compute context."""
    pid = os.getpid() if current_pid is None else int(current_pid)
    python_processes = _powershell_python_processes(pid)
    compute_pids = _gpu_compute_pids()
    external = [row for row in python_processes if int(row["pid"]) in compute_pids]
    receipt = {
        "current_pid": pid,
        "external_python_processes": external,
        "external_python_processes_observed_but_not_using_gpu": [row for row in python_processes if int(row["pid"]) not in compute_pids],
        "gpu_compute_pids": sorted(compute_pids),
        "gpu_snapshot": _nvidia_snapshot(),
    }
    if external:
        raise RuntimeError(f"exclusive-GPU timing guard rejected external Python processes: {external}")
    return receipt


def append_exclusive_receipt(receipts: list[dict[str, object]]) -> None:
    receipts.append(assert_exclusive_gpu())
