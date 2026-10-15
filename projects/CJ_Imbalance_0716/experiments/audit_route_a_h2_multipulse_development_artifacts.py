"""Create a hash-addressed audit manifest only after all H2 formal runs exist.

This script never opens the external H2 source or test-v3.  It audits the
development artifacts produced by the already-frozen C0--C4 runner, so it is
safe to use after the serial queue has completed and cannot alter selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODES = ("C0-mp", "C1-mp", "C2-mp", "C3-mp", "C4-mp")
SEEDS = (20260803, 20260811, 20260819, 20260827, 20260904)
REQUIRED_RUN_FILES = ("checkpoint.pt", "development_predictions.npz", "calibration_and_thresholds.json")
SOURCE_RECEIPTS = {
    "train-mp-v1": "train_mp_v1_r1_generation_receipt.json",
    "development-mp-v1": "development_mp_v1_r1_generation_receipt.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(root: Path, mode: str, seed: int) -> tuple[dict, dict]:
    receipt_path = root / "run_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"missing formal receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed_development_unaggregated" or receipt.get("smoke"):
        raise ValueError(f"not a completed formal run: {receipt_path}")
    if any(receipt.get(key, True) for key in ("locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError(f"test boundary violation: {receipt_path}")
    if receipt.get("modes") != [mode] or receipt.get("seeds") != [seed] or len(receipt.get("results", [])) != 1:
        raise ValueError(f"run identity mismatch: {receipt_path}")
    result = receipt["results"][0]
    if result.get("mode") != mode or int(result.get("seed")) != seed:
        raise ValueError(f"result identity mismatch: {receipt_path}")
    source = receipt.get("source", {})
    if "train-mp-v1" not in str(source.get("train", "")) or "development-mp-v1" not in str(source.get("development", "")):
        raise ValueError(f"unexpected H2 source roles: {receipt_path}")
    return receipt, result


def _source_receipt(generation_root: Path, role: str, name: str) -> dict:
    path = generation_root / name
    if not path.is_file():
        raise ValueError(f"missing source generation receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed" or value.get("role") != role:
        raise ValueError(f"unexpected source generation receipt: {path}")
    if any(value.get(key, True) for key in ("test_files_opened", "locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError(f"test boundary violation in source receipt: {path}")
    source_sha = value.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError(f"source SHA-256 missing: {path}")
    return {"role": role, "receipt": str(path), "receipt_sha256": sha256(path), "source_path": value.get("source_path"), "source_sha256": source_sha, "sample_count": value.get("sample_count"), "pair_count": value.get("pair_count"), "identity_range": value.get("sample_seed_range")}


def audit(root_template: str, output: Path, generation_root: Path) -> dict:
    records: list[dict] = []
    parameter_counts: dict[str, list[int]] = {mode: [] for mode in MODES}
    source_provenance = {role: _source_receipt(generation_root, role, name) for role, name in SOURCE_RECEIPTS.items()}
    for seed in SEEDS:
        for mode in MODES:
            root = Path(root_template.format(mode=mode.lower().replace("-mp", ""), seed=seed))
            receipt, result = _receipt(root, mode, seed)
            artifact_root = root / f"{mode}_seed{seed}"
            files = {"run_receipt.json": {"bytes": (root / "run_receipt.json").stat().st_size, "sha256": sha256(root / "run_receipt.json")}}
            for name in REQUIRED_RUN_FILES:
                path = artifact_root / name
                if not path.is_file():
                    raise ValueError(f"missing required artifact: {path}")
                files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            parameter_counts[mode].append(int(result["parameter_count"]))
            records.append({
                "mode": mode,
                "seed": seed,
                "root": str(root),
                "best_epoch": int(result["best_epoch"]),
                "parameter_count": int(result["parameter_count"]),
                "files": files,
            })
    c3_c4_equal = parameter_counts["C3-mp"] == parameter_counts["C4-mp"]
    if not c3_c4_equal:
        raise ValueError("C3/C4 parameter counts differ")
    result = {
        "status": "completed_artifact_audit",
        "experiment": "RouteA_H2_MultiPulseSixLabel",
        "record_count": len(records),
        "source_provenance": source_provenance,
        "records": records,
        "parameter_counts_per_mode": parameter_counts,
        "c3_c4_equal_parameter_count": c3_c4_equal,
        "external_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-template", default="outputs/tapc/route_a_h2_multipulse_{mode}_seed{seed}")
    parser.add_argument("--output", type=Path, default=Path("outputs/tapc/route_a_h2_multipulse_artifact_audit.json"))
    parser.add_argument("--generation-root", type=Path, default=Path("outputs/tapc/route_a_h2_multipulse_generation"))
    args = parser.parse_args()
    value = audit(args.root_template, args.output, args.generation_root)
    print(json.dumps({key: value[key] for key in ("status", "record_count", "c3_c4_equal_parameter_count")}, ensure_ascii=False))
