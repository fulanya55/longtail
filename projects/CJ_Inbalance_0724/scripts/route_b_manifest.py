"""Create a reproducible hash manifest for the active Route-B code/protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    files = [
        "cj0724/__init__.py", "cj0724/data.py", "cj0724/models.py", "cj0724/multitask.py",
        "cj0724/multitask_benchmark.py", "scripts/route_b_r1_train.py",
        "scripts/route_b_r1_aggregate.py", "scripts/route_b_r2_r3_preflight.py",
        "scripts/route_b_r3_evaluate.py", "scripts/route_b_r4_preflight.py",
        "scripts/route_b_resource_benchmark.py", "scripts/route_b_finalize_report.py",
        "scripts/route_b_write_stop_receipt.py", "scripts/route_b_r0_readiness.py",
        "configs/route_b_r1_preregister.json", "configs/route_b_r4_preregister.json",
    ]
    entries = {item: sha256(project / item) for item in files if (project / item).is_file()}
    manifest = {
        "project_root": str(project),
        "code_version": "local_active_route_b_20260826",
        "files": entries,
        "config_hash": hashlib.sha256(json.dumps({k: entries[k] for k in entries if k.startswith("configs/")}, sort_keys=True).encode()).hexdigest(),
        "manifest_hash": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        "locked_test_used": False,
    }
    receipt = project / "docs" / "receipts" / "route_b_manifest_receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
