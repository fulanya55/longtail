"""Build the reproducible Terminal-N evidence ledger and closure receipts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/route_a_frequency_reconstruction_v2"
CLOSURE_ROOT = OUTPUT_ROOT / "terminal_N_closure"
FORMAL_ROOT = OUTPUT_ROOT / "stage2_formal"
GATE_ROOT = OUTPUT_ROOT / "stage2_frequency_gate"
REPORT = ROOT / "docs/reports/route_a_frequency_reconstruction_v2_terminal_n_report_20260808.md"
NEXT_GOAL = ROOT / "docs/prompts/route_a_frequency_representation_v3_autonomous_goal_prompt_20260808.md"
PREREG = ROOT / "docs/plans/route_a_frequency_reconstruction_v2_preregistration_20260808.md"
SEEDS = (260821, 260822, 260823, 260824, 260825)
REGIMES = ("FB", "LabelIF20", "ComboIF20", "LabelIF10", "ComboIF10", "LabelIF50", "ComboIF50")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def validate_formal() -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for regime in REGIMES:
            run_root = FORMAL_ROOT / f"seed_{seed}" / regime
            receipt_path = run_root / "run_receipt.json"
            summary_path = run_root / "summary.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["status"] != "completed" or receipt["summary_sha256"] != sha256(summary_path):
                raise ValueError(f"formal receipt mismatch: {seed}/{regime}")
            rows.append({
                "seed": seed,
                "regime": regime,
                "receipt": str(receipt_path.relative_to(ROOT)).replace("\\", "/"),
                "receipt_sha256": sha256(receipt_path),
                "summary_sha256": receipt["summary_sha256"],
            })
    return rows


def artifact_manifest() -> dict[str, Any]:
    entries = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if not path.is_file() or CLOSURE_ROOT in path.parents:
            continue
        entries.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for path in (REPORT, NEXT_GOAL, PREREG):
        entries.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return {
        "status": "completed",
        "artifact_count": len(entries),
        "total_bytes": int(sum(entry["bytes"] for entry in entries)),
        "artifacts": entries,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def main():
    CLOSURE_ROOT.mkdir(parents=True, exist_ok=True)
    formal_rows = validate_formal()
    gate_path = GATE_ROOT / "frequency_gate_summary.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["decision"] != "frequency_reconstruction_failed":
        raise ValueError("Terminal N requires a failed frozen frequency gate")
    cost_path = OUTPUT_ROOT / "stage2_cost_audit/cost_summary.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    data_receipt = json.loads((OUTPUT_ROOT / "stage1_data_audit/run_receipt.json").read_text(encoding="utf-8"))
    formal_terminal = json.loads((FORMAL_ROOT / "terminal_receipt.json").read_text(encoding="utf-8"))
    if data_receipt["status"] != "completed" or formal_terminal["run_count"] != 35 or cost["status"] != "completed":
        raise ValueError("required upstream completion evidence is missing")

    ledger = {
        "status": "completed",
        "terminal": "N",
        "decision": "frequency_reconstruction_failed",
        "evidence_scope": "new_identity_development_only",
        "evidence_items": [
            {"id": "E0", "status": "verified", "claim": "PY10/CUDA/worktree/seal/seed lineage audited before formal generation", "artifact": "outputs/route_a_frequency_reconstruction_v2/stage0/stage0_audit.json"},
            {"id": "E1", "status": "verified", "claim": "25,200 paired VH/VL identities generated and atomically reloaded", "artifact": "outputs/route_a_frequency_reconstruction_v2/stage1_data_generation/formal_generation_receipt.json"},
            {"id": "E2", "status": "verified", "claim": "Roles are complete/disjoint and FB/IF10/20/50 manifests meet tolerance and trainability", "artifact": "outputs/route_a_frequency_reconstruction_v2/stage1_data_audit/data_audit.json"},
            {"id": "E3", "status": "verified", "claim": "35/35 five-seed formal cells completed with checkpoints, histories, predictions, and receipts", "artifact": "outputs/route_a_frequency_reconstruction_v2/stage2_formal/run_index.json"},
            {"id": "E4", "status": "failed", "claim": "F3 does not establish the frequency main effect", "value": gate["primary_if20_macro_delta"], "artifact": "outputs/route_a_frequency_reconstruction_v2/stage2_frequency_gate/frequency_gate_summary.json"},
            {"id": "E5", "status": "failed", "claim": "Receiver bootstrap is strictly negative", "value": gate["receiver_bootstrap"]["combined_primary_if20"]},
            {"id": "E6", "status": "failed", "claim": "Co-occurrence matched pairs do not beat random pairs", "value": gate["controls"]["vs_random_all_primary"]},
            {"id": "E7", "status": "verified", "claim": "FB safety passes but does not rescue efficacy", "value": gate["fb_macro_delta"]},
            {"id": "E8", "status": "not_executed_by_gate", "claim": "RawSaw visibility replication prohibited after frequency failure"},
            {"id": "E9", "status": "not_executed_by_gate", "claim": "2x2 joint/interaction prohibited after frequency failure"},
            {"id": "E10", "status": "not_executed_by_gate", "claim": "Fresh confirmation prohibited after frequency failure"},
            {"id": "E11", "status": "bounded_protocol_interpretation", "claim": "Final prereg operationalized F2 as label-balanced head-only cRT and used equal normalized bias anchoring; parent Goal wording mentioned plain BCE and 0.1 bias factor. Main negative comparison remains against best F0/F1/F2 and random-pair."},
            {"id": "E12", "status": "verified", "claim": "Inference cost equals B0 because F3 replaces, rather than adds, the 774-parameter linear head", "value": {"parameters": cost["parameter_count_total"], "macs": cost["macs_per_sample_conv_linear_only"], "batch1_ms": cost["batch_1"]["mean_sample_latency_ms"], "batch64_throughput": cost["batch_64"]["throughput_samples_per_second"]}},
        ],
        "failed_gates": gate["failed_gates"],
        "formal_cells": formal_rows,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    ledger_path = CLOSURE_ROOT / "evidence_ledger.json"
    write_json(ledger_path, ledger)

    closure = {
        "status": "completed",
        "terminal": "N",
        "decision": "frequency_reconstruction_failed",
        "scientific_conclusion": "Frozen B0 linear RB-C cRT did not reconstruct label-frequency robustness and did not beat the matched random-pair control.",
        "primary_if20_macro_delta": gate["primary_if20_macro_delta"],
        "receiver_bootstrap": gate["receiver_bootstrap"]["combined_primary_if20"],
        "failed_gate_count": len(gate["failed_gates"]),
        "failed_gates": gate["failed_gates"],
        "next_actions": {
            "rawsaw": "closed_not_executed",
            "joint_2x2": "closed_not_executed",
            "confirmation": "closed_not_executed",
            "same_source_retuning": "prohibited",
            "future_route": "new_identity representation-level adaptation only, governed by the v3 Goal Prompt",
        },
        "report": {"path": str(REPORT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(REPORT)},
        "next_goal_prompt": {"path": str(NEXT_GOAL.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(NEXT_GOAL)},
        "evidence_ledger_sha256": sha256(ledger_path),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    closure_path = CLOSURE_ROOT / "closure_summary.json"
    write_json(closure_path, closure)

    manifest = artifact_manifest()
    manifest_path = CLOSURE_ROOT / "artifact_manifest.json"
    write_json(manifest_path, manifest)
    receipt = {
        "status": "completed",
        "terminal": "N",
        "goal_complete": True,
        "decision": "frequency_reconstruction_failed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "closure_summary_sha256": sha256(closure_path),
        "evidence_ledger_sha256": sha256(ledger_path),
        "artifact_manifest_sha256": sha256(manifest_path),
        "report_sha256": sha256(REPORT),
        "next_goal_prompt_sha256": sha256(NEXT_GOAL),
        "script_sha256": sha256(Path(__file__)),
        "formal_run_count": 35,
        "bootstrap_repetitions": 5000,
        "rawsaw_trained": False,
        "joint_2x2_run": False,
        "confirmation_run": False,
        "same_source_retuning_authorized": False,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    receipt_path = CLOSURE_ROOT / "run_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "terminal": receipt["terminal"],
        "decision": receipt["decision"],
        "formal_run_count": receipt["formal_run_count"],
        "artifact_count": manifest["artifact_count"],
        "failed_gates": gate["failed_gates"],
        "closure_summary_sha256": receipt["closure_summary_sha256"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
