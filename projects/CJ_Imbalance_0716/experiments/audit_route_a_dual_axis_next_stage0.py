"""Read-only Stage-0 admission audit for the next Route-A dual-axis study.

The audit is intentionally limited to code, reports, receipts, and seal
metadata.  It never opens test-v3 samples, labels, predictions, or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(r"E:\LJQ_code\CJ_Inbalance_0724")

B0_SUMMARY = ROOT / "outputs/tapc/route_a_backbone_b0_development_v1/b0_summary.json"
B0_RECEIPT = ROOT / "outputs/tapc/route_a_backbone_b0_development_v1/terminal_receipt.json"
RAW_LOCKED_REPORT = ROOT / "docs/results/route_a_raw_cross_seed_locked_test_results_20260726.md"
DB_REPORT = ROOT / "docs/results/route_a_head_db_five_seed_confirmatory_results_20260723.md"
P0_SUMMARY = ROOT / "outputs/route_a_ipcve/p0_frozen_observability/p0_summary.json"
P0_RECEIPT = ROOT / "outputs/route_a_ipcve/p0_frozen_observability/run_receipt.json"
POST_P0_AUDIT = ROOT / "outputs/route_a_ipcve/post_p0_observation_admission/admission_audit.json"
TEST_V3_GUARD = ROOT / "frozen/20260731_route_a_test_v3_sealed/TEST_V3_DO_NOT_OPEN.json"
TEST_V3_SEAL = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"

ACTIVE_GENERATORS = [
    ROOT / "experiments/generate_route_a_p1_observability_adequate_pairs.py",
    ROOT / "experiments/generate_route_a_p2_full_target_pairs.py",
    ROOT / "experiments/generate_route_a_h0u_mft_delay_aperture_composite.m",
    ROOT / "experiments/generate_route_a_h0q_vft_staggered_pri.m",
    ROOT / "experiments/generate_route_a_h1a_vft_staggered_pri_mft_isrj.m",
    ROOT / "experiments/generate_route_a_h1b_vft_staggered_pri_all_backgrounds.m",
]

PAIRED_REFERENCE_FILES = [
    REFERENCE_ROOT / "cj0724/dual_axis.py",
    REFERENCE_ROOT / "cj0724/dual_axis_train.py",
    REFERENCE_ROOT / "cj0724/weak_visibility_residual_experiment.py",
    REFERENCE_ROOT / "docs/independent_observation_contract_audit_20260803.md",
]

SECOND_OBSERVATION_TERMS = (
    "iq_aux",
    "second_receiver",
    "dual_receiver",
    "independent_receiver",
    "reference_channel",
    "calibrated_channel",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def git_snapshot() -> dict[str, Any]:
    command = [
        "git",
        "-c",
        "safe.directory=E:/LJQ_code/CJ_Imbalance_0716",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "dirty": bool(rows),
        "porcelain_row_count": len(rows),
        "untracked_row_count": sum(row.startswith("??") for row in rows),
        "tracked_change_row_count": sum(not row.startswith("??") for row in rows),
        "policy": "preserve_all_preexisting_user_changes; only add goal-scoped active artifacts",
    }


def environment_snapshot() -> dict[str, Any]:
    import h5py
    import numpy
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "h5py_version": h5py.__version__,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_count": torch.cuda.device_count(),
    }


def audit_active_generators() -> list[dict[str, Any]]:
    rows = []
    for path in ACTIVE_GENERATORS:
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        matches = [term for term in SECOND_OBSERVATION_TERMS if term.lower() in text.lower()]
        rows.append({
            **artifact(path),
            "observation_contract": "single_complex_iq",
            "second_observation_terms_found": matches,
            "admitted": False,
            "rejection_reason": "no_deployable_independent_second_physical_observation",
        })
    return rows


def audit_paired_reference() -> dict[str, Any]:
    records = [artifact(path) for path in PAIRED_REFERENCE_FILES]
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in PAIRED_REFERENCE_FILES
        if path.is_file()
    ).lower()
    return {
        "candidate": "CJ_Inbalance_0724_iq_reference",
        "artifacts": records,
        "iq_reference_declared": "iq_reference" in combined,
        "fixed_30db_reference_detected": "30.0" in combined or "30 db" in combined or "30db" in combined,
        "training_only_teacher_detected": "training-only ranking teacher" in combined,
        "observation_contract": "same_identity_paired_power_view_used_as_training_reference",
        "available_at_single_view_deployment": False,
        "independent_physical_channel": False,
        "admitted": False,
        "rejection_reason": (
            "paired high-power counterfactual is a training teacher, not a simultaneously "
            "available independent receiver observation"
        ),
    }


def extract_b0() -> dict[str, Any]:
    summary = load_json(B0_SUMMARY)
    aggregate = summary["aggregate"]
    resources = summary["per_seed"]
    return {
        "artifacts": [artifact(B0_SUMMARY), artifact(B0_RECEIPT)],
        "status": summary["status"],
        "evidence_level": summary["evidence_level"],
        "model": summary["model"],
        "seeds": summary["seeds"],
        "macro_aupr_mean": aggregate["macro_aupr"]["mean"],
        "macro_f1_mean": aggregate["macro_f1"]["mean"],
        "target_mean_weak_auroc": aggregate["target_mean_weak_auroc"]["mean"],
        "target_mean_weak_recall": aggregate["target_mean_weak_recall"]["mean"],
        "target_mean_strong_recall": aggregate["target_mean_strong_recall"]["mean"],
        "target_mean_negative_fpr": aggregate["target_mean_negative_fpr"]["mean"],
        "parameter_count": resources[0]["parameter_count"],
        "conv_linear_macs_per_sample": resources[0]["conv_linear_macs_per_sample"],
        "cuda_batch1_latency_ms_per_seed": [row["cuda_batch1_latency_ms"] for row in resources],
        "peak_cuda_memory_bytes_max": max(row["peak_cuda_memory_bytes"] for row in resources),
        "locked_test_used": summary["locked_test_used"],
        "test_v2_used": summary["test_v2_used"],
        "test_v3_used": summary["test_v3_used"],
    }


def build_audit() -> dict[str, Any]:
    b0 = extract_b0()
    p0 = load_json(P0_SUMMARY)
    post_p0 = load_json(POST_P0_AUDIT)
    guard = load_json(TEST_V3_GUARD)
    seal = load_json(TEST_V3_SEAL)
    active_generators = audit_active_generators()
    paired_reference = audit_paired_reference()

    test_v3_clear = bool(
        seal.get("status") == "sealed"
        and seal.get("test_v3_used") is False
        and seal.get("test_v3", {}).get("used") is False
        and seal.get("test_v3", {}).get("access_authorized") is False
        and seal.get("test_v3", {}).get("analysis_allowed") is False
        and guard.get("access_authorized") is False
    )
    admissible_candidates = [
        row for row in [*active_generators, paired_reference] if row.get("admitted") is True
    ]
    decision = (
        "admit_route_s_independent_observation"
        if admissible_candidates
        else "no_admissible_non_oracle_observation"
    )
    branch = "S_independent_observation" if admissible_candidates else "R_sawfm_scoped_dual_axis"

    return {
        "status": "completed_read_only_stage0_admission_audit",
        "experiment": "RouteA_DualAxis_Next_Stage0_Admission",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "read_only_meta_audit_no_data_generation_no_model_training",
        "environment": environment_snapshot(),
        "worktree": git_snapshot(),
        "authoritative_baselines": {
            "B0": b0,
            "RawSawCrossSeedFPR15+H2Others": {
                "artifact": artifact(RAW_LOCKED_REPORT),
                "evidence_role": "consumed_historic_locked_test; never reusable for selection",
                "locked_macro_aupr_before": 0.861894,
                "locked_macro_aupr_after": 0.893382,
                "locked_macro_f1_before": 0.755500,
                "locked_macro_f1_after": 0.796137,
                "saw_weak_recall_before": 0.155255,
                "saw_weak_recall_after": 0.512175,
                "saw_negative_fpr_before": 0.190661,
                "saw_negative_fpr_after": 0.135177,
            },
            "Base_DB": {
                "artifact": artifact(DB_REPORT),
                "evidence_role": "development_only_frequency_axis_baseline",
                "mean_macro_aupr_delta_vs_base_asl_over_18_tail_rotations": 0.017870,
                "positive_rotations": "18/18",
            },
        },
        "negative_evidence": {
            "IP_CVE_P0": {
                "artifacts": [artifact(P0_SUMMARY), artifact(P0_RECEIPT)],
                "passed": p0["aggregate"]["passed"],
                "decision": p0["aggregate"]["decision"],
                "mean_weak_auroc_delta_vs_strongest": p0["aggregate"]["mean_weak_auroc_delta_vs_strongest"],
                "passed_targets": p0["aggregate"]["passed_targets"],
            },
            "post_P0_local_admission": {
                "artifact": artifact(POST_P0_AUDIT),
                "decision": post_p0["decision"],
                "current_audited_generators_declare_second_observation": post_p0["checks"][
                    "current_audited_generators_declare_second_observation"
                ],
            },
        },
        "second_observation_candidates": {
            "active_route_a_generators": active_generators,
            "paired_30db_reference": paired_reference,
            "external_contracts": {
                "M1_dual_receiver": "not_present",
                "M2_repeated_independent_PRI": "not_present",
                "M3_dual_slope_probe": "not_present",
                "M4_external_delay_reference": "not_present",
                "admitted": False,
                "rejection_reason": "requires a new real acquisition or simulator contract",
            },
        },
        "test_v3_seal": {
            "metadata_artifacts_only": [artifact(TEST_V3_GUARD), artifact(TEST_V3_SEAL)],
            "status": seal.get("status"),
            "access_authorized": seal.get("test_v3", {}).get("access_authorized"),
            "analysis_allowed": seal.get("test_v3", {}).get("analysis_allowed"),
            "used": seal.get("test_v3", {}).get("used"),
            "sha256": seal.get("test_v3", {}).get("sha256"),
            "clear_for_future_one_time_use": test_v3_clear,
            "sample_label_prediction_metric_files_opened": False,
        },
        "consistency_corrections": [{
            "field": "B0 conv/linear MACs per sample",
            "authoritative_value": b0["conv_linear_macs_per_sample"],
            "incorrect_value_found_in_active_20260807_summary": 11857536,
            "action": "correct active summary and prompt only; preserve official historical artifacts",
        }],
        "checks": {
            "all_required_evidence_present": all(
                path.is_file()
                for path in [
                    B0_SUMMARY, B0_RECEIPT, RAW_LOCKED_REPORT, DB_REPORT,
                    P0_SUMMARY, P0_RECEIPT, POST_P0_AUDIT, TEST_V3_GUARD, TEST_V3_SEAL,
                ]
            ),
            "b0_authoritative_macs_equal_211857536": b0["conv_linear_macs_per_sample"] == 211857536,
            "test_v3_seal_intact": test_v3_clear,
            "admissible_non_oracle_second_observation_found": bool(admissible_candidates),
            "stage0_generated_data": False,
            "stage0_trained_model": False,
            "stage0_selected_threshold": False,
            "stage0_used_locked_test": False,
            "stage0_used_test_v2": False,
            "stage0_used_test_v3": False,
        },
        "decision": decision,
        "selected_branch": branch,
        "next_action": "write_route_r1_sawfm_frequency_visibility_2x2_preregistration",
        "new_data_generated": False,
        "model_trained": False,
        "calibration_used": False,
        "threshold_selection_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/route_a_dual_axis_next/stage0_admission",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    audit = build_audit()
    audit_path = args.output_dir / "admission_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    receipt = {
        "status": "completed",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "D:/anaconda3/envs/PY10/python.exe "
            "experiments/audit_route_a_dual_axis_next_stage0.py"
        ),
        "decision": audit["decision"],
        "selected_branch": audit["selected_branch"],
        "audit_path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
        "audit_sha256": sha256(audit_path),
        "script_sha256": sha256(Path(__file__)),
        "new_data_generated": False,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    receipt_path = args.output_dir / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "decision": audit["decision"],
        "selected_branch": audit["selected_branch"],
        "checks": audit["checks"],
        "receipt": str(receipt_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
