"""Verify the frozen P2n receipt before P2o confirmation execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_FREEZE_SHA256 = "ab1d5efee7f0a35278a3e9e3b53d68e70d9c40adcd21a676a27195bd6a5f0f10"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze(project: Path, freeze_path: Path) -> dict[str, object]:
    errors = []
    freeze_hash = sha256_file(freeze_path)
    if freeze_hash != EXPECTED_FREEZE_SHA256:
        errors.append("P2n method-freeze hash changed")
    receipt = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, artifact in receipt["artifacts"].items():
        path = Path(artifact["path"])
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            errors.append(f"frozen artifact changed: {name}")
    if receipt.get("locked_test_used", True):
        errors.append("P2n receipt reports locked_test_used=true")
    return {
        "freeze_path": str(freeze_path.resolve()),
        "freeze_sha256": freeze_hash,
        "artifact_count": len(receipt["artifacts"]),
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default="outputs/p2n_analytic_saw_full_system_20260801/method_freeze.json")
    parser.add_argument("--output", default="outputs/p2o_analytic_ood_confirmation_20260801/preflight.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = verify_freeze(project, project / args.freeze)
    output = project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
