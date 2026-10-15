"""Read-only terminal freezer for P5u development and confirmation evidence."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cj0724.p5b1_jnr_physical_oracle import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, payload: dict) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        if not path.exists():
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _terminal_pair(root: Path) -> tuple[dict, dict] | None:
    started, terminal = root / "supervisor_child_started.json", root / "supervisor_terminal.json"
    if not started.exists() or not terminal.exists():
        return None
    return _load(started), _load(terminal)


def _resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _validate_freezer_preregister(project: Path, preregister: Path, parent: Path, confirmation: Path, development: Path, confirmation_root: Path, output: Path) -> None:
    protocol = _load(preregister)
    if protocol.get("experiment") != "P5u terminal evidence freezer":
        raise RuntimeError("P5u final evidence freezer preregistration mismatch")
    expected = {
        "parent_preregister": parent,
        "confirmation_preregister": confirmation,
        "development_root": development,
        "confirmation_root": confirmation_root,
        "output": output,
    }
    for key, value in expected.items():
        if _resolve(project, protocol[key]) != value:
            raise RuntimeError(f"P5u final evidence freezer path mismatch: {key}")
    for artifact in protocol.get("frozen_artifacts", []):
        artifact_path = _resolve(project, artifact["path"])
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"P5u final evidence freezer artifact mismatch: {artifact['path']}")


def _valid_direct_terminal(root: Path) -> tuple[bool, dict]:
    pair = _terminal_pair(root)
    if pair is None:
        return False, {"reason": "terminal_or_child_started_missing"}
    started, terminal = pair
    valid = terminal.get("stage") == "child_terminal" and terminal.get("returncode") == 0 and terminal.get("child_pid") == started.get("child_pid")
    return valid, {"started": started, "terminal": terminal}


def _all_gates(result: dict, role: str, expected_pools: int) -> bool:
    pools = result.get(role, {})
    return len(pools) == expected_pools and all(all(item.get("gates", {}).values()) for item in pools.values())


def _latency_summary(result: dict, role: str) -> dict[str, float]:
    ratios = [float(row["timing"]["median_ratio_shared_dense_over_reference"]) for pool in result[role].values() for row in pool["rows"]]
    return {"count": len(ratios), "minimum": float(np.min(ratios)), "median": float(np.median(ratios)), "maximum": float(np.max(ratios))}


def inspect(project: Path, development_root: Path, confirmation_root: Path) -> tuple[str, dict | None]:
    development_result = development_root / "result.json"
    development_pair = _terminal_pair(development_root)
    if development_pair is None:
        return "development_running", None
    if not development_result.exists():
        return "development_stopped_without_result", {"development_terminal": development_pair[1]}
    dev = _load(development_result)
    dev_terminal_ok, dev_terminal = _valid_direct_terminal(development_root)
    if not dev.get("development_passed") or not dev_terminal_ok or not _all_gates(dev, "development", 3):
        return "development_failed", {"development": dev, "development_terminal": dev_terminal}
    confirmation_result = confirmation_root / "result.json"
    confirmation_pair = _terminal_pair(confirmation_root)
    if confirmation_pair is None:
        return "confirmation_running_or_not_started", None
    if not confirmation_result.exists():
        return "confirmation_stopped_without_result", {"confirmation_terminal": confirmation_pair[1]}
    confirmation = _load(confirmation_result)
    confirmation_terminal_ok, confirmation_terminal = _valid_direct_terminal(confirmation_root)
    if not confirmation.get("confirmation_passed") or not confirmation_terminal_ok or not _all_gates(confirmation, "confirmation", 2):
        return "confirmation_failed", {"development": dev, "confirmation": confirmation, "development_terminal": dev_terminal, "confirmation_terminal": confirmation_terminal}
    return "p5_final_confirmed", {"development": dev, "confirmation": confirmation, "development_terminal": dev_terminal, "confirmation_terminal": confirmation_terminal}


def freeze(project: Path, parent_preregister: Path, confirmation_preregister: Path, development_root: Path, confirmation_root: Path, output: Path) -> bool:
    state, payload = inspect(project, development_root, confirmation_root)
    if payload is None:
        return False
    base = {
        "experiment": "P5u direct-supervised frozen shared-forward execution",
        "status": state,
        "created_utc": _utc_now(),
        "parent_preregister": str(parent_preregister),
        "parent_preregister_sha256": sha256_file(parent_preregister),
        "confirmation_preregister": str(confirmation_preregister),
        "confirmation_preregister_sha256": sha256_file(confirmation_preregister),
        "development_execution_root": str(development_root),
        "confirmation_execution_root": str(confirmation_root),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    if state == "p5_final_confirmed":
        development, confirmation = payload["development"], payload["confirmation"]
        base.update({
            "evidence_role": "five-model three-development-pool plus two-confirmation-pool strict-output-equivalent execution-cost evidence",
            "development_result_sha256": sha256_file(development_root / "result.json"),
            "confirmation_result_sha256": sha256_file(confirmation_root / "result.json"),
            "development_direct_terminal": payload["development_terminal"],
            "confirmation_direct_terminal": payload["confirmation_terminal"],
            "development_latency_ratio_summary": _latency_summary(development, "development"),
            "confirmation_latency_ratio_summary": _latency_summary(confirmation, "confirmation"),
            "frozen_claim": "For the fixed first 126-example batch, the shared frozen base-forward plus unchanged dense JNR attachment was exactly output-equivalent to the two-forward reference and satisfied the pre-registered latency gate for every reported model and pool. This is not a classification, OOD, composition-disjoint, long-tail or general-acceleration claim.",
        })
    else:
        base.update({"evidence_role": "stopped_or_failed_execution_protocol", "payload": payload, "frozen_claim": "No P5u execution-performance claim is supported; the consumed identities are not eligible for rerun, selection, calibration, confirmation reuse or locked-test use."})
    _write_once(output, base)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-preregister", required=True)
    parser.add_argument("--freezer-preregister", required=True)
    parser.add_argument("--confirmation-preregister", required=True)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    paths = [_resolve(project, value) for value in (args.parent_preregister, args.confirmation_preregister, args.development_root, args.confirmation_root, args.output)]
    _validate_freezer_preregister(project, _resolve(project, args.freezer_preregister), *paths)
    while True:
        if freeze(project, *paths):
            return
        if not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
