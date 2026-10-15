"""Validate and seal the Route-A dual-axis Goal as terminal N."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "route_a_dual_axis_next"
FORMAL_ROOT = BASE / "r2_formal_primary"
BOOT_ROOT = BASE / "r2_receiver_bootstrap_audit"
COST_ROOT = BASE / "r2_cost_audit"
OUTPUT_ROOT = BASE / "terminal_N_closure"
SCRIPT_PATH = Path(__file__).resolve()
SEEDS = (260811, 260812, 260813, 260814, 260815)
CELLS = ("FB-VH", "FB-VL", "LT-VH", "LT-VL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_no_test_use(payload: dict, source: str) -> None:
    for key in ("locked_test_used", "test_v2_used", "test_v3_used"):
        if bool(payload.get(key)):
            raise ValueError(f"{source} reports forbidden {key}")


def validate_receipted_summary(root: Path, summary_name: str) -> tuple[dict, dict]:
    receipt_path = root / "run_receipt.json"
    summary_path = root / summary_name
    receipt = read_json(receipt_path)
    summary = read_json(summary_path)
    if receipt["status"] != "completed" or summary["status"] != "completed":
        raise ValueError(f"incomplete receipted summary: {root}")
    if sha256(summary_path) != receipt["summary_sha256"]:
        raise ValueError(f"summary hash mismatch: {root}")
    script_path = Path(receipt["script_path"])
    if sha256(script_path) != receipt["script_sha256"]:
        raise ValueError(f"script hash mismatch: {root}")
    assert_no_test_use(receipt, str(receipt_path))
    assert_no_test_use(summary, str(summary_path))
    return receipt, summary


def main() -> None:
    started = time.time()
    stage0 = read_json(BASE / "stage0_admission" / "admission_audit.json")
    stage0_receipt = read_json(BASE / "stage0_admission" / "run_receipt.json")
    if stage0_receipt["status"] != "completed" or stage0["decision"] != "no_admissible_non_oracle_observation":
        raise ValueError("Stage0 does not support terminal N")
    assert_no_test_use(stage0_receipt, "Stage0 receipt")

    formal_terminal_path = FORMAL_ROOT / "terminal_receipt.json"
    formal_summary_path = FORMAL_ROOT / "formal_primary_summary.json"
    formal_terminal = read_json(formal_terminal_path)
    formal = read_json(formal_summary_path)
    if formal_terminal["status"] != "completed" or formal["status"] != "completed":
        raise ValueError("formal matrix is incomplete")
    if sha256(formal_summary_path) != formal_terminal["summary_sha256"]:
        raise ValueError("formal summary hash mismatch")
    assert_no_test_use(formal_terminal, "formal terminal")
    assert_no_test_use(formal, "formal summary")
    gates = formal["aggregate"]["gates"]
    expected_gates = {
        "MF_frequency_main_effect": False,
        "MV_visibility_main_effect": True,
        "MFV_point_effect": False,
        "non_saw_safety": True,
        "controls_admitted": False,
    }
    if gates != expected_gates:
        raise ValueError(f"unexpected formal gate state: {gates}")
    if formal["next_decision"] != "stop_primary_mechanism_before_controls":
        raise ValueError("formal stop action drifted")

    formal_runner = ROOT / "experiments" / "run_route_a_dual_axis_sawfm_r2_formal.py"
    formal_runner_sha = sha256(formal_runner)
    cell_audit = []
    for seed in SEEDS:
        for cell in CELLS:
            cell_root = FORMAL_ROOT / f"seed_{seed}" / cell
            receipt = read_json(cell_root / "run_receipt.json")
            summary = read_json(cell_root / "cell_summary.json")
            if receipt["status"] != "completed" or summary["status"] != "completed":
                raise ValueError(f"incomplete cell: {seed}/{cell}")
            if receipt["script_sha256"] != formal_runner_sha:
                raise ValueError(f"formal runner hash drift: {seed}/{cell}")
            if sha256(cell_root / "cell_summary.json") != receipt["summary_sha256"]:
                raise ValueError(f"cell summary hash mismatch: {seed}/{cell}")
            if sha256(cell_root / "evaluation_predictions.npz") != summary["predictions_sha256"]:
                raise ValueError(f"prediction hash mismatch: {seed}/{cell}")
            for name, checkpoint in summary["checkpoints"].items():
                if sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
                    raise ValueError(f"checkpoint hash mismatch: {seed}/{cell}/{name}")
            assert_no_test_use(receipt, f"cell receipt {seed}/{cell}")
            assert_no_test_use(summary, f"cell summary {seed}/{cell}")
            cell_audit.append({
                "seed": seed,
                "cell": cell,
                "status": "completed",
                "summary_sha256": receipt["summary_sha256"],
                "predictions_sha256": summary["predictions_sha256"],
            })
    if len(cell_audit) != 20:
        raise ValueError("formal cell count is not 20")

    boot_receipt, boot = validate_receipted_summary(BOOT_ROOT, "receiver_bootstrap_summary.json")
    cost_receipt, cost = validate_receipted_summary(COST_ROOT, "cost_summary.json")
    boot_gates = boot["gates"]
    if boot_gates["MF_frequency_main_effect"] or not boot_gates["MV_visibility_main_effect"]:
        raise ValueError("bootstrap axis gates drifted")
    if boot_gates["MFV_independent_increment"] or boot_gates["controls_admitted"]:
        raise ValueError("bootstrap unexpectedly admits combined/controls")

    report_path = ROOT / "docs" / "reports" / "route_a_dual_axis_r2_terminal_n_negative_closure_20260807.md"
    ledger_path = BASE / "evidence_ledger.json"
    ledger = read_json(ledger_path)
    if ledger["status"] != "terminal_N_complete" or ledger["goal_terminal"] != "N":
        raise ValueError("evidence ledger is not terminal N")
    assert_no_test_use(ledger, "evidence ledger")

    bundle_paths = [
        ROOT / "docs" / "prompts" / "route_a_dual_axis_next_stage_autonomous_goal_prompt_20260807.md",
        ROOT / "docs" / "plans" / "route_a_dual_axis_next_r1_sawfm_protocol_preregistration_20260807.md",
        ROOT / "docs" / "plans" / "route_a_dual_axis_next_r2_formal_execution_addendum_20260807.md",
        ROOT / "docs" / "plans" / "route_a_dual_axis_r2_receiver_bootstrap_audit_protocol_20260807.md",
        ROOT / "cjlt" / "models" / "dual_axis_sawfm.py",
        formal_runner,
        ROOT / "experiments" / "readiness_route_a_dual_axis_sawfm_r2_formal.py",
        ROOT / "experiments" / "audit_route_a_dual_axis_r2_receiver_bootstrap.py",
        ROOT / "experiments" / "audit_route_a_dual_axis_r2_cost.py",
        ROOT / "tests" / "test_route_a_dual_axis_sawfm_r2_formal.py",
        BASE / "r1_data_audit" / "data_audit.json",
        BASE / "r1_data_audit" / "role_indices.npz",
        FORMAL_ROOT / "resolved_config.json",
        formal_summary_path,
        formal_terminal_path,
        BOOT_ROOT / "receiver_bootstrap_summary.json",
        BOOT_ROOT / "paired_bootstrap_samples.npz",
        COST_ROOT / "cost_summary.json",
        report_path,
        ledger_path,
    ]
    missing = [str(path) for path in bundle_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"reproducibility bundle missing files: {missing}")
    bundle = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in bundle_paths
    }

    closure = {
        "status": "completed",
        "goal_terminal": "N",
        "active_branch": "R_sawfm_scoped_dual_axis",
        "terminal_stage": "terminal_N_negative_closure",
        "criteria": {
            "N1_new_observation_candidates_inadmissible": True,
            "N2_route_R_independent_dual_axis_effect_failed": True,
            "N3_failure_controls_statistics_receipts_boundaries_complete": True,
            "N4_unified_route_closed_and_scope_contracted": True,
        },
        "scientific_outcome": {
            "frequency_axis": "failed_minimum_effect_gate",
            "visibility_axis": "passed_on_new_development",
            "combined_independent_increment": "failed",
            "positive_claim_controls": "not_run_preregistered_stop",
            "confirmation": "not_created_gate_failure",
            "retained_scope": "SawFM_specific_RawSaw_visibility_evidence_only",
            "closed_scope": "unified_six_label_dual_axis_model",
        },
        "key_metrics": {
            "frequency_lt_macro_aupr_delta": boot["metrics"]["frequency_lt_macro_aupr_delta"],
            "visibility_weak_auroc_delta": boot["metrics"]["visibility_weak_auroc_delta"],
            "visibility_weak_recall_delta": boot["metrics"]["visibility_weak_recall_delta"],
            "visibility_candidate_fpr": boot["metrics"]["visibility_candidate_fpr"],
            "combined_macro_aupr_delta_vs_best": boot["metrics"]["combined_macro_aupr_delta_vs_best"],
            "combined_weak_recall_delta_vs_best": boot["metrics"]["combined_weak_recall_delta_vs_best"],
        },
        "cost": {
            "parameters": cost["parameters"],
            "macs_per_sample": cost["macs_per_sample"],
            "formal_terminal_wall_seconds": cost["training_cost"]["terminal_wall_seconds"],
            "max_peak_cuda_memory_bytes": cost["training_cost"]["max_peak_cuda_memory_bytes"],
        },
        "formal_cell_audit": cell_audit,
        "reproducibility_bundle_sha256": bundle,
        "report_path": str(report_path),
        "report_sha256": sha256(report_path),
        "ledger_path": str(ledger_path),
        "ledger_sha256": sha256(ledger_path),
        "bootstrap_receipt_sha256": sha256(BOOT_ROOT / "run_receipt.json"),
        "cost_receipt_sha256": sha256(COST_ROOT / "run_receipt.json"),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    closure_path = OUTPUT_ROOT / "closure_summary.json"
    write_json(closure_path, closure)
    receipt = {
        "status": "completed",
        "goal_terminal": "N",
        "elapsed_seconds": time.time() - started,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "closure_summary_path": str(closure_path),
        "closure_summary_sha256": sha256(closure_path),
        "python": sys.version,
        "platform": platform.platform(),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    write_json(OUTPUT_ROOT / "run_receipt.json", receipt)
    print(json.dumps({
        "status": "completed",
        "goal_terminal": "N",
        "criteria": closure["criteria"],
        "formal_cells_verified": len(cell_audit),
        "bundle_files_verified": len(bundle),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
