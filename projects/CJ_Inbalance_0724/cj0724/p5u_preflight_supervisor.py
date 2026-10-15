"""P5u direct-child supervisor: run source preflight, then supervise P5s executor."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from cj0724.p5b1_jnr_physical_oracle import sha256_file


def _utc_now() -> str: return datetime.now(timezone.utc).isoformat()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists(): raise RuntimeError(f"P5u supervisor receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "P5u-preflight-supervisor-v1": raise RuntimeError("P5u supervisor manifest schema mismatch")
    for key in ("working_directory", "execution_root", "preflight_command", "child_command"):
        if key not in manifest: raise RuntimeError(f"P5u manifest missing {key}")
    if not Path(manifest["working_directory"]).is_dir(): raise RuntimeError("P5u working directory missing")
    if not all(isinstance(item, str) for item in manifest["preflight_command"] + manifest["child_command"]): raise RuntimeError("P5u commands must be string lists")
    return manifest


def run_manifest(path: Path) -> int:
    manifest = _load_manifest(path); root = Path(manifest["execution_root"]); root.mkdir(parents=True, exist_ok=True)
    base = {"schema": "P5u-preflight-supervisor-v1", "created_utc": _utc_now(), "supervisor_pid": os.getpid(), "manifest": str(path.resolve()), "manifest_sha256": sha256_file(path)}
    _atomic_write_once(root / "supervisor_started.json", {**base, "stage": "supervisor_started"})
    preflight = subprocess.run(manifest["preflight_command"], cwd=manifest["working_directory"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    _atomic_write_once(root / "supervisor_preflight.json", {**base, "stage": "preflight_completed", "returncode": int(preflight.returncode), "stdout": preflight.stdout, "stderr": preflight.stderr})
    if preflight.returncode != 0:
        _atomic_write_once(root / "supervisor_terminal.json", {**base, "stage": "preflight_terminal", "returncode": int(preflight.returncode)})
        return int(preflight.returncode)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    stdout_path, stderr_path = root / "supervisor_child_stdout.log", root / "supervisor_child_stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        child = subprocess.Popen(manifest["child_command"], cwd=manifest["working_directory"], stdout=stdout, stderr=stderr, creationflags=flags)
        _atomic_write_once(root / "supervisor_child_started.json", {**base, "stage": "child_started", "child_pid": child.pid, "command": manifest["child_command"]})
        code = int(child.wait())
    _atomic_write_once(root / "supervisor_terminal.json", {**base, "stage": "child_terminal", "child_pid": child.pid, "returncode": code})
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--manifest", required=True); args = parser.parse_args()
    raise SystemExit(run_manifest(Path(args.manifest)))


if __name__ == "__main__": main()
