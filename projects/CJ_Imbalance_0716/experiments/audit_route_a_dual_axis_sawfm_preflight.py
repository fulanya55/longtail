"""Preflight gate for formal dual-axis SawFM development-v1 generation."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "experiments/generate_route_a_dual_axis_sawfm_development_v1.m"
DRY_RECEIPT = ROOT / "outputs/route_a_dual_axis_next/r1_data_generation/dry_run_generation_receipt.json"
OUTPUT_DIR = ROOT / "outputs/route_a_dual_axis_next/r1_data_generation"
KNOWN_GENERATOR_ROOTS = [
    ROOT / "experiments",
    Path(r"E:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成"),
]
SEED_PATTERN = re.compile(r"(?:cfg\.seed|baseSeed)\s*=\s*([0-9]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def seed_inventory() -> list[dict[str, Any]]:
    rows = []
    for root in KNOWN_GENERATOR_ROOTS:
        for path in root.glob("**/generate*.m"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in SEED_PATTERN.finditer(text):
                seed = int(match.group(1))
                rows.append({
                    "path": str(path),
                    "base_seed": seed,
                    "assumed_train_range": [seed + 1, seed + 25200],
                    "assumed_test_range": [seed + 3000001, seed + 3025200],
                })
    unique = {}
    for row in rows:
        unique[(row["path"], row["base_seed"])] = row
    return sorted(unique.values(), key=lambda row: (row["base_seed"], row["path"]))


def build_audit() -> dict[str, Any]:
    dry = json.loads(DRY_RECEIPT.read_text(encoding="utf-8"))
    signatures = dry["train_seed_signatures"]
    ranges = [(int(row["first"]), int(row["last"])) for row in signatures]
    paired = len(ranges) == 2 and ranges[0] == ranges[1]
    new_range = ranges[0]
    inventory = seed_inventory()
    collisions = []
    for row in inventory:
        if Path(row["path"]).resolve() == GENERATOR.resolve():
            continue
        for role in ("assumed_train_range", "assumed_test_range"):
            protected = tuple(row[role])
            if overlap(new_range, protected):
                collisions.append({
                    "path": row["path"],
                    "base_seed": row["base_seed"],
                    "role": role,
                    "protected_range": list(protected),
                })

    passed = bool(
        dry["status"] == "completed"
        and dry["executed"] is False
        and dry["paired_seed_verified"] is True
        and paired
        and int(dry["base_seed"]) == 26080701
        and dry["levels_dB"] == [20, 30]
        and dry["swept_codes"] == ["SawFM"]
        and dry["split_selection"] == "train_only"
        and dry["test_views_generated"] is False
        and not collisions
    )
    return {
        "status": "completed",
        "experiment": "RouteA_DualAxis_SawFM_DevelopmentV1_Preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "generator": {"path": str(GENERATOR), "sha256": sha256(GENERATOR)},
            "dry_run_receipt": {"path": str(DRY_RECEIPT), "sha256": sha256(DRY_RECEIPT)},
        },
        "new_identity_range": list(new_range),
        "paired_view_identity_range_equal": paired,
        "known_generator_seed_inventory": inventory,
        "identity_collisions": collisions,
        "checks": {
            "dry_run_completed_without_generation": dry["executed"] is False,
            "paired_seed_verified": dry["paired_seed_verified"] is True and paired,
            "sawfm_only_visibility_intervention": dry["swept_codes"] == ["SawFM"],
            "train_only_no_test_view": (
                dry["split_selection"] == "train_only"
                and dry["test_views_generated"] is False
            ),
            "known_generator_seed_ranges_disjoint": not collisions,
            "formal_generation_admitted": passed,
        },
        "decision": "admit_formal_generation" if passed else "stop_before_generation",
        "new_data_generated": False,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    audit = build_audit()
    audit_path = OUTPUT_DIR / "preflight_identity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    receipt = {
        "status": "completed" if audit["checks"]["formal_generation_admitted"] else "stopped",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": audit["decision"],
        "audit_path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
        "audit_sha256": sha256(audit_path),
        "script_sha256": sha256(Path(__file__)),
        "new_data_generated": False,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    receipt_path = OUTPUT_DIR / "preflight_identity_audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "decision": audit["decision"],
        "new_identity_range": audit["new_identity_range"],
        "collision_count": len(audit["identity_collisions"]),
        "checks": audit["checks"],
    }, indent=2, ensure_ascii=False))
    if not audit["checks"]["formal_generation_admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
