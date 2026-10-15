"""Run and record Route-B R0 readiness without opening any locked test data.

This script is intentionally limited to active-code checks, the six-class
training MAT contract, unit/full tests, and a small real-data smoke.  It does
not train a model, fit a threshold, or discover/open a test MAT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.multitask import PARAMETER_SPECS


PROJECT = Path(__file__).resolve().parents[1]
DOCS = PROJECT / "docs"
REPORTS = DOCS / "reports"
RESULTS = DOCS / "results"
RECEIPTS = DOCS / "receipts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=PROJECT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "returncode": int(process.returncode),
        "passed": bool(process.returncode == 0),
        "elapsed_seconds": float(time.perf_counter() - started),
        "stdout_tail": process.stdout[-12000:],
        "stderr_tail": process.stderr[-12000:],
    }


def audit_training_mat(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        required = {"data_legacy", "label_selected", "parameter_values", "parameter_mask", "receiver_noise_seed"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"missing datasets: {missing}")
        data_shape = tuple(handle["data_legacy"].shape)
        label_shape = tuple(handle["label_selected"].shape)
        value_shape = tuple(handle["parameter_values"].shape)
        mask_shape = tuple(handle["parameter_mask"].shape)
        seed_shape = tuple(handle["receiver_noise_seed"].shape)
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        values = np.asarray(handle["parameter_values"][:, :], dtype=np.float64).T
        masks = np.asarray(handle["parameter_mask"][:, :], dtype=np.uint8).T.astype(bool)
        receiver_seeds = np.asarray(handle["receiver_noise_seed"][:, :], dtype=np.int64).reshape(-1)

    if data_shape != (6000, 25200):
        raise ValueError(f"unexpected data shape: {data_shape}")
    if label_shape != (6, 25200) or labels.shape != (25200, 6):
        raise ValueError(f"unexpected label shape: {label_shape}")
    if value_shape != (79, 25200) or mask_shape != (79, 25200):
        raise ValueError(f"unexpected parameter shape: {value_shape}, {mask_shape}")
    if seed_shape != (1, 25200) or receiver_seeds.size != 25200:
        raise ValueError(f"unexpected receiver_noise_seed shape: {seed_shape}")

    selected_columns = [spec.column_index for spec in PARAMETER_SPECS]
    selected_values = values[:, selected_columns].reshape(len(labels), len(CLASS_NAMES), 2)
    selected_masks = masks[:, selected_columns].reshape(len(labels), len(CLASS_NAMES), 2)
    valid = selected_masks & labels.astype(bool)[:, :, None] & np.isfinite(selected_values)
    negative_mask_count = int(np.sum(selected_masks & ~labels.astype(bool)[:, :, None]))
    finite_valid = bool(np.isfinite(selected_values[valid]).all())
    parameter_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for parameter_index, spec in enumerate(PARAMETER_SPECS[2 * class_index:2 * class_index + 2]):
            current = selected_values[:, class_index, parameter_index][valid[:, class_index, parameter_index]]
            parameter_rows.append({
                "class_name": class_name,
                "parameter": spec.name,
                "valid_count": int(current.size),
                "finite": bool(np.isfinite(current).all()),
                "minimum": float(np.min(current)),
                "maximum": float(np.max(current)),
                "range": float(np.ptp(current)),
            })
    combo_ids = labels.dot(1 << np.arange(labels.shape[1], dtype=np.int64))
    combo_values, combo_counts = np.unique(combo_ids, return_counts=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": {
            "data_legacy": list(data_shape),
            "label_selected": list(label_shape),
            "parameter_values": list(value_shape),
            "parameter_mask": list(mask_shape),
            "receiver_noise_seed": list(seed_shape),
        },
        "labels": {
            "class_names": list(CLASS_NAMES),
            "sample_count": int(len(labels)),
            "positive_counts": labels.sum(axis=0).astype(int).tolist(),
            "combination_count": int(len(combo_values)),
            "minimum_combination_count": int(combo_counts.min()),
            "maximum_combination_count": int(combo_counts.max()),
        },
        "identity": {
            "count": int(receiver_seeds.size),
            "unique": int(np.unique(receiver_seeds).size),
            "sha256": hashlib.sha256(receiver_seeds.astype("<i8").tobytes()).hexdigest(),
        },
        "parameter_contract": {
            "selected_parameter_count": len(PARAMETER_SPECS),
            "valid_rule": "label=1 AND parameter_mask=1 AND target finite",
            "valid_count": int(valid.sum()),
            "negative_mask_count": negative_mask_count,
            "all_valid_finite": finite_valid,
            "rows": parameter_rows,
        },
        "locked_test_used": False,
    }


def write_pending_csv(path: Path, fields: list[str], row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    for directory in (REPORTS, RESULTS, RECEIPTS):
        directory.mkdir(parents=True, exist_ok=True)
    data_audit = audit_training_mat(Path(DEFAULT_TRAIN_MAT))
    commands = [
        run_command([sys.executable, "-m", "compileall", "-q", "cj0724", "scripts"]),
        run_command([sys.executable, "-m", "pytest", "-q"]),
        run_command([sys.executable, "scripts/smoke_real_data.py"]),
    ]
    all_passed = bool(data_audit["parameter_contract"]["all_valid_finite"] and all(item["passed"] for item in commands))
    receipt = {
        "stage": "R0",
        "status": "verified" if all_passed else "stopped/failed",
        "project": str(PROJECT),
        "data_audit": data_audit,
        "commands": commands,
        "locked_test_used": False,
        "test_generated": False,
        "test_opened": False,
        "notes": [
            "No model training or threshold fitting was performed.",
            "The active default MAT path was repaired from a historical C: path to the E: research root.",
            "Historical D-drive path tests now construct paths without opening archived files; execution still validates file existence.",
        ],
    }
    (RECEIPTS / "route_b_terminal_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    result_receipt = {
        "stage": "R0",
        "status": receipt["status"],
        "evidence": {
            "data_contract": "verified" if data_audit["parameter_contract"]["all_valid_finite"] else "failed",
            "compile": "verified" if commands[0]["passed"] else "failed",
            "full_pytest": "verified" if commands[1]["passed"] else "failed",
            "gpu_smoke": "verified" if commands[2]["passed"] else "failed",
        },
        "locked_test_used": False,
    }
    (RECEIPTS / "route_b_result_receipt.json").write_text(json.dumps(result_receipt, indent=2), encoding="utf-8")

    report = f"""# Route B R0 readiness\n\n- Status: **{receipt['status']}**\n- Project root: `{PROJECT}`\n- Data role: development/training MAT only\n- Locked test opened: `False`\n- Model training: `False`\n\n## Data contract\n\n- MAT: `{data_audit['path']}`\n- File SHA-256: `{data_audit['sha256']}`\n- Input `data_legacy`: `{data_audit['shape']['data_legacy']}`; sample contract `[N, 2, 3000]` after reshape\n- Labels: `{data_audit['labels']['class_names']}`\n- Samples: `{data_audit['labels']['sample_count']}`; combinations: `{data_audit['labels']['combination_count']}`; per-combination count range: `{data_audit['labels']['minimum_combination_count']}—{data_audit['labels']['maximum_combination_count']}`\n- Unique `receiver_noise_seed`: `{data_audit['identity']['unique']}/{data_audit['identity']['count']}`\n- Parameter rule: `{data_audit['parameter_contract']['valid_rule']}`\n- Valid selected parameter entries: `{data_audit['parameter_contract']['valid_count']}`\n- Mask-on-negative entries: `{data_audit['parameter_contract']['negative_mask_count']}`\n- All valid targets finite: `{data_audit['parameter_contract']['all_valid_finite']}`\n\n## Executed gates\n\n| Gate | Status |\n|---|---|\n| Python compile | {result_receipt['evidence']['compile']} |\n| Full pytest | {result_receipt['evidence']['full_pytest']} |\n| GPU/real-data smoke | {result_receipt['evidence']['gpu_smoke']} |\n| Data shape/label/mask/finite audit | {result_receipt['evidence']['data_contract']} |\n\n## Boundary\n\nThis receipt proves R0 readiness only. It does not prove R1 baseline performance, R2 conditional-head improvement, multi-seed stability, long-tail generalization, or open-world performance. No test-v2 role has been created or opened.\n"""
    (REPORTS / "route_b_r0_readiness.md").write_text(report, encoding="utf-8")

    pending = {"status": "pending", "stage": "not_started", "note": "R0 only; no result rows generated"}
    write_pending_csv(RESULTS / "route_b_r1_baseline.csv", ["status", "stage", "note"], pending)
    write_pending_csv(RESULTS / "route_b_r2_conditional_head.csv", ["status", "stage", "note"], pending)
    write_pending_csv(RESULTS / "route_b_r3_ablation.csv", ["status", "stage", "note"], pending)
    for name in ("route_b_per_parameter_metrics.csv", "route_b_visibility_audit.csv", "route_b_paired_power_consistency.csv"):
        write_pending_csv(RESULTS / name, ["status", "stage", "note"], pending)
    evidence_index = {
        "goal": "Route B conditional parameter estimation",
        "r0": str(RECEIPTS / "route_b_result_receipt.json"),
        "r0_report": str(REPORTS / "route_b_r0_readiness.md"),
        "r1_r4_status": "pending",
        "locked_test_used": False,
    }
    (RECEIPTS / "route_b_final_evidence_index.json").write_text(json.dumps(evidence_index, indent=2), encoding="utf-8")
    print(json.dumps(result_receipt, indent=2))


if __name__ == "__main__":
    main()
