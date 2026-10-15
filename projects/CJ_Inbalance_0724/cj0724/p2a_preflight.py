"""G0/P2a preflight: audit the frozen P1n reference without opening waveform data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


EXPECTED_MODEL_SEEDS = (2041, 2042, 2043)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference(method_freeze: dict, summary: dict, freeze_hash: str, registered_hash: str) -> list[str]:
    errors: list[str] = []
    if freeze_hash != registered_hash:
        errors.append("P1n method-freeze hash differs from the P2a preregistration")
    if summary.get("method_freeze_sha256") != freeze_hash:
        errors.append("P1n summary does not reference the current method-freeze hash")
    if not summary.get("level3_validated", False):
        errors.append("P1n reference is not marked level3_validated")
    if summary.get("locked_test_used", True):
        errors.append("P1n summary unexpectedly reports locked-test use")
    if not summary.get("test_used_once", False):
        errors.append("P1n fresh test is not recorded as used once")
    if summary.get("test_fit_or_selection", True):
        errors.append("P1n fresh test was used for fitting or selection")
    if method_freeze.get("test_used_at_freeze", True):
        errors.append("P1n method freeze was written after test access")
    seeds = tuple(int(row["seed"]) for row in method_freeze.get("seed_freezes", []))
    if seeds != EXPECTED_MODEL_SEEDS:
        errors.append(f"Unexpected P1n seed order: {seeds}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1n-root", default="outputs/p1n_full_blend_joint_20260801")
    parser.add_argument("--preregister", default="configs/p2a_nine_power_generalization_preregister_20260731.json")
    parser.add_argument("--fresh-raw-root", default="generated_data/p2a_nine_power_generalization_seed1200000000")
    parser.add_argument("--output-root", default="outputs/p2a_g0_preflight_20260731")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    p1n_root = project / args.p1n_root
    preregister_path = project / args.preregister
    fresh_raw_root = project / args.fresh_raw_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)

    method_freeze_path = p1n_root / "method_freeze.json"
    summary_path = p1n_root / "summary.json"
    method_freeze = json.loads(method_freeze_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    preregister = json.loads(preregister_path.read_text(encoding="utf-8"))
    freeze_hash = sha256_file(method_freeze_path)
    registered_hash = preregister["reference_only"]["method_freeze_sha256"]
    errors = validate_reference(method_freeze, summary, freeze_hash, registered_hash)

    checkpoint_receipts = []
    for row in method_freeze["seed_freezes"]:
        for field in ("main_checkpoint", "unet_checkpoint", "saw_checkpoint"):
            path = Path(row[field])
            if not path.is_file():
                errors.append(f"Missing frozen checkpoint: {path}")
                continue
            checkpoint_receipts.append({
                "seed": int(row["seed"]), "field": field, "path": str(path),
                "bytes": int(path.stat().st_size), "sha256": sha256_file(path),
            })

    source_receipts = []
    for relative in (
        "cj0724/models.py",
        "cj0724/multitask.py",
        "cj0724/p1k_joint_level3_system.py",
        "cj0724/p1k_finalize_saved_predictions.py",
        "matlab/generate_p2a_nine_power_generalization_data.m",
    ):
        path = project / relative
        if not path.is_file():
            errors.append(f"Missing P2a source: {path}")
            continue
        source_receipts.append({
            "path": relative, "bytes": int(path.stat().st_size), "sha256": sha256_file(path),
        })

    # Inventory only: do not import h5py and do not open any MAT waveform file.
    fresh_inventory = []
    if fresh_raw_root.is_dir():
        for path in sorted(fresh_raw_root.rglob("*")):
            if path.is_file():
                fresh_inventory.append({
                    "path": str(path.relative_to(project)),
                    "bytes": int(path.stat().st_size),
                    "suffix": "".join(path.suffixes),
                })

    ledger = {
        "experiment": "G0 frozen P1n evidence ledger and P2a preflight",
        "created_unix": time.time(),
        "valid": not errors,
        "errors": errors,
        "p1n_reference": {
            "method_freeze": str(method_freeze_path),
            "method_freeze_sha256": freeze_hash,
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "level3_validated": bool(summary.get("level3_validated", False)),
            "test_used_once": bool(summary.get("test_used_once", False)),
            "test_fit_or_selection": bool(summary.get("test_fit_or_selection", False)),
            "locked_test_used": bool(summary.get("locked_test_used", False)),
            "identity_audit": summary.get("identity_audit"),
        },
        "preregister": {
            "path": str(preregister_path),
            "sha256": sha256_file(preregister_path),
        },
        "checkpoint_receipts": checkpoint_receipts,
        "source_receipts": source_receipts,
        "fresh_data_inventory_without_waveform_read": fresh_inventory,
        "historical_waveform_or_prediction_read_by_preflight": False,
        "fresh_waveform_read_by_preflight": False,
        "locked_test_used": False,
    }
    (output_root / "evidence_ledger.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps({
        "valid": True, "method_freeze_sha256": freeze_hash,
        "checkpoint_count": len(checkpoint_receipts),
        "source_count": len(source_receipts),
        "fresh_inventory_count": len(fresh_inventory),
        "output": str(output_root / "evidence_ledger.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
