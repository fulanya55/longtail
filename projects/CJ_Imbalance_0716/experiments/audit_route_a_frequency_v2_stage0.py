"""Stage-0 immutable-evidence and seed-lineage audit for frequency v2."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "route_a_frequency_reconstruction_v2" / "stage0"
GENERATOR = ROOT / "experiments" / "generate_route_a_frequency_development_v2.m"
PREREG = ROOT / "docs" / "plans" / "route_a_frequency_reconstruction_v2_preregistration_20260808.md"
GOAL = ROOT / "docs" / "prompts" / "route_a_frequency_reconstruction_next_stage_autonomous_goal_prompt_20260808.md"
DRY_RECEIPT = ROOT / "outputs" / "route_a_frequency_reconstruction_v2" / "stage1_data_generation" / "dry_run_generation_receipt.json"
TERMINAL_N_RECEIPT = ROOT / "outputs" / "route_a_dual_axis_next" / "terminal_N_closure" / "run_receipt.json"
TERMINAL_N_SUMMARY = ROOT / "outputs" / "route_a_dual_axis_next" / "terminal_N_closure" / "closure_summary.json"
R2_DATA_AUDIT = ROOT / "outputs" / "route_a_dual_axis_next" / "r1_data_audit" / "data_audit.json"
SEAL = ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json"
NEW_RANGE = (3_930_000_001, 3_930_025_200)
SEED_PATTERN = re.compile(r"(?:cfg\.seed|baseSeed)\s*=\s*([0-9]+)")
RANGE_KEYS = {
    "receiver_noise_seed_range",
    "observed_receiver_noise_seed_range",
    "new_identity_range",
    "receiver_range",
}
GENERATOR_ROOTS = (
    ROOT / "experiments",
    Path(r"E:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def generator_inventory() -> list[dict[str, Any]]:
    rows = []
    for root in GENERATOR_ROOTS:
        if not root.exists():
            continue
        for path in root.glob("**/generate*.m"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in SEED_PATTERN.finditer(text):
                seed = int(match.group(1))
                rows.append({
                    "path": str(path),
                    "base_seed": seed,
                    "assumed_train_range": [seed + 1, seed + 25_200],
                    "assumed_test_range": [seed + 3_000_001, seed + 3_025_200],
                })
    unique = {(row["path"], row["base_seed"]): row for row in rows}
    return sorted(unique.values(), key=lambda row: (row["base_seed"], row["path"]))


def walk_ranges(value: Any, path: str = "") -> list[dict[str, Any]]:
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in RANGE_KEYS and isinstance(child, list) and len(child) == 2:
                if all(isinstance(item, (int, float)) for item in child):
                    rows.append({"json_key": child_path, "range": [int(child[0]), int(child[1])]})
            rows.extend(walk_ranges(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(walk_ranges(child, f"{path}[{index}]"))
    return rows


def audit_json_ranges() -> list[dict[str, Any]]:
    rows = []
    output_root = ROOT / "outputs"
    for path in output_root.glob("**/*.json"):
        if "route_a_frequency_reconstruction_v2" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            continue
        for row in walk_ranges(payload):
            rows.append({"path": str(path), **row})
    return rows


def git_status() -> list[str]:
    result = subprocess.run(
        ["git", "-c", "safe.directory=E:/LJQ_code/CJ_Imbalance_0716", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    terminal_receipt = json.loads(TERMINAL_N_RECEIPT.read_text(encoding="utf-8"))
    terminal_summary = json.loads(TERMINAL_N_SUMMARY.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    r2_data = json.loads(R2_DATA_AUDIT.read_text(encoding="utf-8"))
    dry = json.loads(DRY_RECEIPT.read_text(encoding="utf-8"))
    inventory = generator_inventory()
    json_ranges = audit_json_ranges()

    generator_collisions = []
    for row in inventory:
        if Path(row["path"]).resolve() == GENERATOR.resolve():
            continue
        for role in ("assumed_train_range", "assumed_test_range"):
            protected = tuple(row[role])
            if overlap(NEW_RANGE, protected):
                generator_collisions.append({**row, "role": role, "protected_range": list(protected)})
    json_collisions = [row for row in json_ranges if overlap(NEW_RANGE, tuple(row["range"]))]

    statuses = git_status()
    test_v3 = seal["test_v3"]
    checks = {
        "py10_python_310": sys.version_info[:2] == (3, 10),
        "cuda_available": bool(torch.cuda.is_available()),
        "terminal_n_complete": (
            terminal_receipt["status"] == "completed"
            and terminal_summary["status"] == "completed"
            and terminal_summary["goal_terminal"] == "N"
        ),
        "terminal_n_hash_intact": sha256(TERMINAL_N_SUMMARY) == terminal_receipt["closure_summary_sha256"],
        "test_v3_seal_intact": (
            seal["status"] == "sealed"
            and test_v3["access_authorized"] is False
            and test_v3["analysis_allowed"] is False
            and test_v3["used"] is False
        ),
        "dry_run_completed_without_generation": dry["status"] == "completed" and dry["executed"] is False,
        "dry_run_contract": (
            int(dry["base_seed"]) == 3_930_000_000
            and dry["levels_dB"] == [20, 30]
            and dry["swept_codes"] == ["SawFM"]
            and dry["split_selection"] == "train_only"
            and dry["test_views_generated"] is False
            and dry["paired_seed_verified"] is True
        ),
        "dry_run_identity_range": all(
            [int(item["first"]), int(item["last"])] == list(NEW_RANGE)
            for item in dry["train_seed_signatures"]
        ),
        "generator_ranges_disjoint": len(generator_collisions) == 0,
        "audited_json_ranges_disjoint": len(json_collisions) == 0,
        "r2_source_hashes_present": all(
            r2_data["files"][name].get("sha256")
            for name in ("low_visibility", "high_visibility")
        ),
        "new_data_absent_before_formal_generation": not Path(dry["sweep_root"]).exists(),
        "worktree_audited_without_cleanup": True,
    }
    admitted = all(checks.values())
    audit = {
        "status": "completed",
        "experiment": "RouteA_FrequencyReconstructionV2_Stage0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "admit_formal_generation" if admitted else "stop_before_generation",
        "new_identity_range": list(NEW_RANGE),
        "generator_inventory_count": len(inventory),
        "audited_json_range_count": len(json_ranges),
        "generator_collisions": generator_collisions,
        "json_range_collisions": json_collisions,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "worktree": {
            "dirty": bool(statuses),
            "entry_count": len(statuses),
            "entries": statuses,
            "policy": "preserve all pre-existing changes; add only Goal-scoped files",
        },
        "artifacts": {
            "goal": {"path": str(GOAL), "sha256": sha256(GOAL)},
            "preregistration": {"path": str(PREREG), "sha256": sha256(PREREG)},
            "generator": {"path": str(GENERATOR), "sha256": sha256(GENERATOR)},
            "dry_run_receipt": {"path": str(DRY_RECEIPT), "sha256": sha256(DRY_RECEIPT)},
            "terminal_n_receipt": {"path": str(TERMINAL_N_RECEIPT), "sha256": sha256(TERMINAL_N_RECEIPT)},
            "terminal_n_summary": {"path": str(TERMINAL_N_SUMMARY), "sha256": sha256(TERMINAL_N_SUMMARY)},
            "r2_data_audit": {"path": str(R2_DATA_AUDIT), "sha256": sha256(R2_DATA_AUDIT)},
            "test_v3_seal_metadata": {"path": str(SEAL), "sha256": sha256(SEAL)},
        },
        "r2_source_hashes": {
            name: r2_data["files"][name]["sha256"]
            for name in ("low_visibility", "high_visibility")
        },
        "checks": checks,
        "new_data_generated": False,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT / "stage0_audit.json"
    write_json(audit_path, audit)
    receipt = {
        "status": "completed" if admitted else "stopped",
        "decision": audit["decision"],
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "new_data_generated": False,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    write_json(OUTPUT / "run_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "decision": audit["decision"], "checks": checks, "collisions": {"generator": len(generator_collisions), "json": len(json_collisions)}}, ensure_ascii=False, indent=2))
    if not admitted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

