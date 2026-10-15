"""Read-only post-P0 admission audit for a genuinely new observation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EVIDENCE = {
    "ipcve_p0": "outputs/route_a_ipcve/p0_frozen_observability/p0_summary.json",
    "mft_h0u": "outputs/tapc/route_a_h0u_mft_delay_aperture/summary.json",
    "mft_h1u": "outputs/tapc/route_a_h1u_mft_delay_aperture_learning/summary.json",
    "mft_h2v": "outputs/tapc/route_a_h2v_b0_admission/summary.json",
    "vft_h0q": "outputs/tapc/route_a_h0q_vft_staggered_pri/route_a_h0q_vft_staggered_pri_summary.json",
    "vft_h1a": "outputs/tapc/route_a_h1a_vft_staggered_pri/route_a_h1a_vft_staggered_pri_summary.json",
    "vft_h1b": "outputs/tapc/route_a_h1b_vft_staggered_pri/route_a_h1b_vft_staggered_pri_summary.json",
    "vft_h2": "outputs/tapc/route_a_h2_multipulse_development_summary.json",
    "all_target_p1": "outputs/tapc/route_a_p1_observability_adequate_h0/formal_summary.json",
    "all_target_p2": "outputs/tapc/route_a_p2_full_target_h0/formal_summary.json",
    "isrj_root_cause": "outputs/tapc/route_a_p1_p2_isrj_h0_failure_diagnosis/summary.json",
}

AUDITED_GENERATORS = [
    "experiments/generate_route_a_p1_observability_adequate_pairs.py",
    "experiments/generate_route_a_p2_full_target_pairs.py",
    "experiments/generate_route_a_h0u_mft_delay_aperture_composite.m",
    "experiments/generate_route_a_h0q_vft_staggered_pri.m",
    "experiments/generate_route_a_h1a_vft_staggered_pri_mft_isrj.m",
    "experiments/generate_route_a_h1b_vft_staggered_pri_all_backgrounds.m",
]

SECOND_OBSERVATION_PATTERN = re.compile(
    r"aux(?:iliary)?_channel|second_receiver|dual_receiver|reference_channel|"
    r"calibrated_channel|controlled_receiver_setting|iq_aux|iq_reference",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _test_flags_clear(payload: dict[str, Any]) -> bool:
    return all(payload.get(name) is False for name in (
        "locked_test_used", "test_v2_used", "test_v3_used"
    ))


def build_audit() -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    artifact_rows = []
    for name, relative in EVIDENCE.items():
        path = ROOT / relative
        payload = _load(relative)
        loaded[name] = payload
        artifact_rows.append({
            "name": name,
            "path": relative,
            "sha256": _sha256(path),
            "status": payload.get("status"),
            "test_flags_clear": _test_flags_clear(payload),
        })

    generator_rows = []
    for relative in AUDITED_GENERATORS:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        generator_rows.append({
            "path": relative,
            "sha256": _sha256(path),
            "declares_second_observation_channel": bool(SECOND_OBSERVATION_PATTERN.search(text)),
        })

    p0 = loaded["ipcve_p0"]["aggregate"]
    h0u = loaded["mft_h0u"]
    h1u = loaded["mft_h1u"]
    h2v = loaded["mft_h2v"]
    h1b = loaded["vft_h1b"]
    h2 = loaded["vft_h2"]
    p1 = loaded["all_target_p1"]
    p2 = loaded["all_target_p2"]
    diagnosis = loaded["isrj_root_cause"]

    mft = {
        "physical_h0_passed": bool(h0u["passed"]),
        "h0_candidate_minus_control_auroc": h0u["candidate_minus_control"]["auroc"],
        "learned_reconnection_passed": bool(h1u["passed"]),
        "h2v_admitted": bool(h2v["admitted_to_separate_correction_preregistration"]),
        "h2v_candidate_trained": bool(h2v["candidate_correction_trained"]),
        "overlap_decision": "closed_existing_delay_aperture_and_lfm_residual_family",
    }
    vft = {
        "all_background_physical_gate_passed": bool(h1b["gate"]["passed"]),
        "h1b_candidate_minus_control_auroc": h1b["metrics"]["candidate_minus_control_auroc"]["mean"],
        "learned_true_timestamp_gate_passed": bool(h2["gate"]["passed"]),
        "c3_minus_c0_weak_auroc": h2["comparison"]["C3_minus_C0_vft_weak_auroc"]["mean"],
        "c3_minus_c4_weak_auroc": h2["comparison"]["C3_minus_C4_vft_weak_auroc"]["mean"],
        "c3_minus_c0_macro_aupr": h2["comparison"]["C3_minus_C0_macro_aupr"]["mean"],
        "overlap_decision": "closed_existing_staggered_pri_timestamp_expert_family",
    }

    def target_row(summary: dict[str, Any], target: str) -> dict[str, Any]:
        row = summary["metrics_by_target"][target]
        return {
            "true_auroc": row["true_auroc"],
            "control_auroc": row["control_auroc"],
            "true_minus_control_auroc": row["true_minus_control_auroc"],
            "paired_weak_gt_absent_fraction": row["paired_weak_gt_absent_fraction"],
        }

    isrj = {
        "p1": target_row(p1, "ISRJ"),
        "p2": target_row(p2, "ISRJ"),
        "weak_minus_absent_by_nuisance_count_p1": diagnosis["p1"]["by_nuisance_label_count"],
        "weak_minus_absent_by_nuisance_count_p2": diagnosis["p2"]["by_nuisance_label_count"],
        "failure_layer": "single_channel_weak_label_attribution_in_compound_background",
        "overlap_decision": "closed_existing_single_channel_periodicity_and_correlation_family",
    }

    all_artifacts_present = len(artifact_rows) == len(EVIDENCE)
    all_test_flags_clear = all(row["test_flags_clear"] for row in artifact_rows)
    current_generators_have_second_observation = any(
        row["declares_second_observation_channel"] for row in generator_rows
    )
    all_local_families_closed = (
        mft["physical_h0_passed"]
        and not mft["learned_reconnection_passed"]
        and not mft["h2v_admitted"]
        and vft["all_background_physical_gate_passed"]
        and not vft["learned_true_timestamp_gate_passed"]
        and not p1["gate"]["passed"]
        and not p2["gate"]["passed"]
        and not p0["passed"]
    )

    admitted = bool(
        all_artifacts_present
        and all_test_flags_clear
        and current_generators_have_second_observation
        and not all_local_families_closed
    )
    decision = (
        "admit_new_nonoverlap_h0"
        if admitted
        else "no_run_close_local_ipcve_mechanism_external_observation_required"
    )
    return {
        "status": "completed_read_only_post_p0_admission_audit",
        "experiment": "RouteA_IPCVE_PostP0ExtendedObservationAdmission",
        "evidence_role": "read_only_meta_audit_no_data_generation_no_model_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_rows": artifact_rows,
        "generator_rows": generator_rows,
        "ipcve_p0": {
            "passed": p0["passed"],
            "decision": p0["decision"],
            "selected_source": p0["selected_source"],
            "mean_weak_auroc_delta_vs_strongest": p0["mean_weak_auroc_delta_vs_strongest"],
            "passed_targets": p0["passed_targets"],
        },
        "mechanism_audit": {"MFT": mft, "VFT": vft, "ISRJ": isrj},
        "checks": {
            "all_evidence_artifacts_present": all_artifacts_present,
            "all_evidence_test_flags_clear": all_test_flags_clear,
            "all_existing_local_candidate_families_closed": all_local_families_closed,
            "current_audited_generators_declare_second_observation": current_generators_have_second_observation,
            "new_local_h0_candidate_admitted": admitted,
        },
        "decision": decision,
        "next_only_action": (
            "independently_specify_and_acquire_a_non_oracle_additional_calibrated_channel_"
            "controlled_receiver_setting_or_real_hil_protocol_then_restart_stage_minus1"
        ),
        "goal_conclusion": "ipcve_current_local_single_view_mechanism_reliably_denied",
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
        default=ROOT / "outputs/route_a_ipcve/post_p0_observation_admission",
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
        "command": "D:/anaconda3/envs/PY10/python.exe experiments/audit_route_a_ipcve_post_p0_observation.py",
        "decision": audit["decision"],
        "admission_audit": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
        "admission_audit_sha256": _sha256(audit_path),
        "new_data_generated": False,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    (args.output_dir / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": audit["status"],
        "decision": audit["decision"],
        "checks": audit["checks"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
