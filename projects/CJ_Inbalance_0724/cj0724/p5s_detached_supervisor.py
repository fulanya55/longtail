"""Detached supervisor for a frozen P5 execution command on Windows."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cj0724.p5b1_jnr_physical_oracle import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"P5s supervisor receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _validate_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "P5s-detached-supervisor-v1":
        raise RuntimeError("P5s supervisor manifest schema mismatch")
    command = manifest.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise RuntimeError("P5s supervisor command must be a non-empty string list")
    working = Path(manifest["working_directory"])
    execution = Path(manifest["execution_root"])
    if not working.is_dir(): raise RuntimeError(f"P5s supervisor working directory missing: {working}")
    if execution.exists() and (execution / "result.json").exists():
        raise RuntimeError("P5s supervisor refuses to launch a completed execution root")
    return manifest


def run_manifest(manifest_path: Path) -> int:
    manifest = _validate_manifest(manifest_path)
    execution = Path(manifest["execution_root"])
    execution.mkdir(parents=True, exist_ok=True)
    base = {"schema": "P5s-detached-supervisor-v1", "created_utc": _utc_now(), "supervisor_pid": os.getpid(), "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path)}
    _atomic_write_once(execution / "supervisor_started.json", {**base, "stage": "supervisor_started"})
    stdout_path, stderr_path = execution / "supervisor_child_stdout.log", execution / "supervisor_child_stderr.log"
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        child = subprocess.Popen(manifest["command"], cwd=manifest["working_directory"], stdout=stdout, stderr=stderr, creationflags=flags)
        _atomic_write_once(execution / "supervisor_child_started.json", {**base, "stage": "child_started", "child_pid": child.pid, "command": manifest["command"]})
        code = int(child.wait())
    _atomic_write_once(execution / "supervisor_terminal.json", {**base, "stage": "child_terminal", "child_pid": child.pid, "returncode": code})
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    raise SystemExit(run_manifest(Path(args.manifest)))


if __name__ == "__main__":
    main()
