"""Gate-driven P5u progression monitor with no CUDA work.

The monitor waits for the already-launched three-pool P5u executor.  It never
interprets partial checkpoints.  Only a successful direct-supervisor terminal
receipt plus a passing final development result unlocks the separately frozen
two-pool confirmation workflow.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cj0724.p5b1_jnr_physical_oracle import sha256_file
from cj0724.p5u_two_pool_confirmation_execution import validate_confirmation_context


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if not path.exists():
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _load_monitor(project: Path, path: Path) -> dict:
    monitor = _load_json(path)
    if monitor.get("schema") != "P5u-autonomous-progression-monitor-v1":
        raise RuntimeError("P5u monitor manifest schema mismatch")
    for artifact in monitor.get("frozen_artifacts", []):
        artifact_path = _resolve(project, artifact["path"])
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"P5u monitor frozen artifact hash mismatch: {artifact['path']}")
    required = ("development_execution_root", "confirmation_preregister", "confirmation_method_freeze", "confirmation_data_roots", "confirmation_audits", "confirmation_execution_root", "historical_baseline", "historical_baseline_sha256")
    missing = [key for key in required if key not in monitor]
    if missing:
        raise RuntimeError(f"P5u monitor manifest missing {missing}")
    if monitor["confirmation_data_roots"] != [
        "generated_data/p5u_two_pool_confirmation_seed3340000000",
        "generated_data/p5u_two_pool_confirmation_seed3350000000",
    ]:
        raise RuntimeError("P5u monitor confirmation data-root contract changed")
    return monitor


def _run(command: list[str], cwd: Path) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": int(completed.returncode), "stdout": completed.stdout, "stderr": completed.stderr}


def _completion_status(project: Path, monitor: dict) -> tuple[str, dict | None]:
    root = _resolve(project, monitor["development_execution_root"])
    terminal, result = root / "supervisor_terminal.json", root / "result.json"
    if not terminal.exists():
        return "running", None
    if not result.exists():
        return "stopped_without_result", _load_json(terminal)
    return "completed", {"terminal": _load_json(terminal), "result": _load_json(result)}


def _confirmation_launch_manifest(project: Path, monitor: dict, root: Path) -> dict:
    data_roots = monitor["confirmation_data_roots"]
    audits = monitor["confirmation_audits"]
    return {
        "schema": "P5u-preflight-supervisor-v1",
        "experiment": "P5u conditional two-new-identity confirmation",
        "working_directory": str(project),
        "execution_root": str(root),
        "preflight_command": [
            monitor["python"], "-m", "cj0724.p5t_legacy_portable_execution",
            "--preregister", "configs/p5t_legacy_portable_preregister_20260806.json",
            "--p5e-freeze", "configs/p5e_five_model_seed_confirmation_freeze_20260801.json",
            "--self-test-frozen-models",
        ],
        "child_command": [
            monitor["python"], "-m", "cj0724.p5u_two_pool_confirmation_execution",
            "--preregister", "configs/p5u_direct_supervision_preregister_20260806.json",
            "--confirmation-preregister", monitor["confirmation_preregister"],
            "--p5e-freeze", "configs/p5e_five_model_seed_confirmation_freeze_20260801.json",
            "--development-result", "outputs/p5u_direct_supervision_20260806/execution/result.json",
            "--development-child-started", "outputs/p5u_direct_supervision_20260806/execution/supervisor_child_started.json",
            "--development-terminal", "outputs/p5u_direct_supervision_20260806/execution/supervisor_terminal.json",
            "--data-roots", *data_roots,
            "--audits", *audits,
            "--output-root", monitor["confirmation_execution_root"],
        ],
        "audit_sha256": {str(seed): sha256_file(_resolve(project, audit)) for seed, audit in zip((3340000000, 3350000000), audits)},
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }


def _generate_and_audit(project: Path, monitor: dict, root: Path) -> None:
    dryrun_receipt = root / "confirmation_generation_dry_run.json"
    matlab_code = "cd('{}'); addpath(fullfile(pwd,'matlab')); generate_p5u_two_pool_confirmation_data({});".format(str(project).replace("\\", "/"), "{}")
    if not dryrun_receipt.exists():
        before = [str(_resolve(project, item)) for item in monitor["confirmation_data_roots"] if _resolve(project, item).exists()]
        if before:
            raise RuntimeError(f"confirmation roots exist before conditional dry-run: {before}")
        dry = _run([monitor["matlab"], "-batch", matlab_code.format("false")], project)
        after = [str(_resolve(project, item)) for item in monitor["confirmation_data_roots"] if _resolve(project, item).exists()]
        _atomic_write_once(dryrun_receipt, {"created_utc": _utc_now(), "stage": "dry_run", "before_roots": before, "after_roots": after, **dry})
        if dry["returncode"] != 0 or after:
            raise RuntimeError("confirmation MATLAB dry-run failed or created data")
    generation_receipt = root / "confirmation_generation.json"
    if not generation_receipt.exists():
        generated = _run([monitor["matlab"], "-batch", matlab_code.format("true")], project)
        _atomic_write_once(generation_receipt, {"created_utc": _utc_now(), "stage": "generation", **generated})
        if generated["returncode"] != 0:
            raise RuntimeError("confirmation MATLAB generation failed")
    for data_root, audit_arg in zip(monitor["confirmation_data_roots"], monitor["confirmation_audits"]):
        root_path, audit_path = _resolve(project, data_root), _resolve(project, audit_arg)
        if not (root_path / "generation_receipt.mat").is_file():
            raise RuntimeError(f"confirmation generation receipt missing: {root_path}")
        if audit_path.exists():
            audit = _load_json(audit_path)
            if not audit.get("valid_for_frozen_inference"):
                raise RuntimeError(f"existing confirmation audit invalid: {audit_path}")
            continue
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_command = [
            monitor["python"], "-m", "cj0724.audit_p5r_relocation_safe_data",
            "--power-root", data_root,
            "--historical-baseline", monitor["historical_baseline"],
            "--historical-baseline-sha256", monitor["historical_baseline_sha256"],
            "--output", audit_arg,
        ]
        audit_run = _run(audit_command, project)
        _atomic_write_once(root / f"{audit_path.stem}_run.json", {"created_utc": _utc_now(), "stage": "audit", **audit_run})
        if audit_run["returncode"] != 0 or not audit_path.exists() or not _load_json(audit_path).get("valid_for_frozen_inference"):
            raise RuntimeError(f"confirmation audit failed: {audit_path}")


def _launch_confirmation(project: Path, monitor: dict, root: Path) -> None:
    execution_root = _resolve(project, monitor["confirmation_execution_root"])
    launch_manifest = root / "confirmation_launch_manifest.json"
    launch_receipt = root / "confirmation_supervisor_started.json"
    if launch_receipt.exists():
        return
    if execution_root.exists():
        raise RuntimeError(f"confirmation execution root already exists before launch: {execution_root}")
    manifest = _confirmation_launch_manifest(project, monitor, execution_root)
    _atomic_write_once(launch_manifest, manifest)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    child = subprocess.Popen([monitor["python"], "-m", "cj0724.p5u_preflight_supervisor", "--manifest", str(launch_manifest)], cwd=project, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    _atomic_write_once(launch_receipt, {"created_utc": _utc_now(), "stage": "confirmation_supervisor_started", "supervisor_pid": child.pid, "launch_manifest": str(launch_manifest), "launch_manifest_sha256": sha256_file(launch_manifest)})


def run_monitor(project: Path, monitor_path: Path, poll_seconds: float, once: bool) -> int:
    monitor = _load_monitor(project, monitor_path)
    root = _resolve(project, monitor["monitor_root"])
    _atomic_write_once(root / "monitor_started.json", {"schema": monitor["schema"], "created_utc": _utc_now(), "monitor_pid": os.getpid(), "monitor_manifest": str(monitor_path), "monitor_manifest_sha256": sha256_file(monitor_path), "cuda_opened": False, "test_generated": False, "test_opened": False, "locked_test_used": False})
    while True:
        status, payload = _completion_status(project, monitor)
        _atomic_write_once(root / "development_status_first_observed.json", {"created_utc": _utc_now(), "status": status})
        if status == "running":
            if once:
                return 0
            time.sleep(poll_seconds)
            continue
        if status == "stopped_without_result":
            _atomic_write_once(root / "development_stopped_without_result.json", {"created_utc": _utc_now(), "status": status, "terminal": payload, "confirmation_generated": False, "locked_test_used": False})
            return 0
        try:
            development_root = _resolve(project, monitor["development_execution_root"])
            context = validate_confirmation_context(project, _resolve(project, monitor["parent_preregister"]), _resolve(project, monitor["confirmation_preregister"]), development_root / "result.json", development_root / "supervisor_child_started.json", development_root / "supervisor_terminal.json")
            _atomic_write_once(root / "development_promoted.json", {"created_utc": _utc_now(), "status": "development_passed", "context": context, "locked_test_used": False})
            _generate_and_audit(project, monitor, root)
            _launch_confirmation(project, monitor, root)
        except Exception as error:
            _atomic_write_once(root / "development_or_promotion_stopped.json", {"created_utc": _utc_now(), "status": "stopped", "error_type": type(error).__name__, "error": str(error), "confirmation_generated": any(_resolve(project, item).exists() for item in monitor["confirmation_data_roots"]), "locked_test_used": False})
            return 0
        confirmation_root = _resolve(project, monitor["confirmation_execution_root"])
        terminal, result = confirmation_root / "supervisor_terminal.json", confirmation_root / "result.json"
        if terminal.exists():
            _atomic_write_once(root / "confirmation_terminal_observed.json", {"created_utc": _utc_now(), "terminal": _load_json(terminal), "result_present": result.exists(), "result": _load_json(result) if result.exists() else None, "locked_test_used": False})
            return 0
        if once:
            return 0
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-manifest", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    raise SystemExit(run_monitor(project, _resolve(project, args.monitor_manifest), args.poll_seconds, args.once))


if __name__ == "__main__":
    main()
