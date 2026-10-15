"""Produce an evidence-boundary audit for the current Route-A candidate.

This is deliberately an audit, not a model selector.  It reads only already
completed development artifacts and the sealed test-v3 manifest.  In
particular, it never opens test-v3 data and cannot authorize its use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result(source: Path) -> dict[str, Any]:
    if not source.exists():
        return {"exists": False, "path": str(source), "sha256": None}
    payload = _read_json(source)
    effective_passed = payload.get("passed")
    if effective_passed is None:
        effective_passed = payload.get("confirmation_passed")
    if effective_passed is None and isinstance(payload.get("gate"), dict):
        effective_passed = payload["gate"].get("passed")
    if effective_passed is None and isinstance(payload.get("construction_gate"), dict):
        effective_passed = payload["construction_gate"].get("passed")
    if effective_passed is None and isinstance(payload.get("unified_promotion_gate"), dict):
        effective_passed = payload["unified_promotion_gate"].get("passed")
    if effective_passed is None and isinstance(payload.get("execution_gate"), dict):
        effective_passed = payload["execution_gate"].get("passed")
    return {
        "exists": True,
        "path": str(source),
        "sha256": _sha256(source),
        "experiment": payload.get("experiment"),
        "passed": effective_passed,
        "locked_test_used": payload.get("locked_test_used"),
        "test_v3_used": payload.get("test_v3_used"),
    }


def build_audit(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    sources = {
        "mainstream_losses": root / "outputs/tapc/route_a_development_v3_mainstream_loss_baselines/route_a_development_v3_mainstream_loss_baselines_summary.json",
        "capacity_heads": root / "outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines/route_a_development_v3_frozen_head_capacity_baselines_summary.json",
        "equal_budget_two_stage": root / "outputs/tapc/route_a_development_v3_equal_budget_two_stage_baseline/route_a_development_v3_equal_budget_two_stage_baseline_summary.json",
        "frozen_component_ablations": root / "outputs/tapc/route_a_final_candidate_frozen_component_ablations/route_a_final_candidate_frozen_component_ablations_summary.json",
        "weak_threshold_sensitivity": root / "outputs/tapc/route_a_final_candidate_weak_threshold_sensitivity/route_a_final_candidate_weak_threshold_sensitivity_summary.json",
        "sqrt_linear_gate_ablation": root / "outputs/tapc/route_a_vft_sqrt_linear_gate_ablation/route_a_vft_sqrt_linear_gate_ablation_summary.json",
        "mft_residual_cap_removal": root / "outputs/tapc/route_a_mft_residual_cap1_ablation/route_a_h1_b2_mft_peak_guard_v3_summary.json",
        "vft_weak_mixed_positive_ablation": root / "outputs/tapc/route_a_vft_weak_mixed_positive_ablation/route_a_vft_weak_mixed_positive_ablation_summary.json",
        "isrj_periodicity": root / "outputs/tapc/route_a_development_v3_h1_periodicity_increment/route_a_development_v3_h1_periodicity_increment_summary.json",
        "unified_v12": root / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen/route_a_h1_b2_mft_vft_development_v12_frozen_summary.json",
        "statistics": root / "outputs/tapc/route_a_final_shared_policy_evidence/statistical_audit.json",
        "stratified_v12": root / "outputs/tapc/route_a_final_shared_policy_evidence/stratified_v12.json",
        "engineering": root / "outputs/tapc/route_a_final_shared_policy_engineering/engineering_audit.json",
        "external_isrj_ood": root / "outputs/tapc/route_a_final_shared_policy_isrj_ood_frozen/route_a_final_shared_policy_isrj_ood_frozen_summary.json",
        "dual_target_contract": root / "outputs/tapc/route_a_dual_target_decision_contract_diagnostic/route_a_dual_target_decision_contract_diagnostic_summary.json",
        "vft_isrj_cross_head": root / "outputs/tapc/route_a_vft_isrj_cross_expert_head/route_a_vft_isrj_cross_expert_head_summary.json",
        "latest_mllt_reproducibility_audit": root / "outputs/tapc/route_a_latest_mllt_reproducibility_audit/latest_mllt_reproducibility_audit.json",
        "vft_isrj_source_v1": root / "outputs/tapc/route_a_vft_isrj_source_v1_raw_feature_diagnostic/route_a_vft_isrj_source_v1_raw_feature_summary.json",
        "vft_isrj_source_v1_stft_raw_complementarity": root / "outputs/tapc/route_a_vft_isrj_source_v1_stft_raw_complementarity/route_a_vft_isrj_source_v1_stft_raw_complementarity_summary.json",
        "vft_isrj_source_v1_highres_stft_projection": root / "outputs/tapc/route_a_vft_isrj_source_v1_highres_stft_projection/route_a_vft_isrj_source_v1_highres_stft_projection_summary.json",
        "vft_hard_soft_power_error": root / "outputs/tapc/route_a_vft_hard_soft_power_error_ablation/route_a_vft_hard_soft_power_error_ablation_summary.json",
        "vft_weighting_uncertainty_core_ablation": root / "outputs/tapc/route_a_vft_weighting_uncertainty_core_ablation/route_a_vft_weighting_uncertainty_core_ablation_summary.json",
        "vft_strict_mask_unified_ablation": root / "outputs/tapc/route_a_vft_strict_mask_unified_ablation/route_a_vft_strict_mask_unified_ablation_summary.json",
        "vft_remove_vtse_unified_decision_ablation": root / "outputs/tapc/route_a_vft_remove_vtse_unified_decision_ablation/route_a_vft_remove_vtse_unified_decision_ablation_summary.json",
        "mft_remove_cap_unified_decision_ablation": root / "outputs/tapc/route_a_mft_remove_cap_unified_decision_ablation/route_a_mft_remove_cap_unified_decision_ablation_summary.json",
        "dr_loss_fair_baseline": root / "outputs/tapc/route_a_development_v3_dr_loss_baseline/route_a_development_v3_dr_loss_baseline_summary.json",
        "rawsaw_topk_ablation": root / "outputs/tapc/route_a_rawsaw_topk_4_8_16_ablation/route_a_rawsaw_topk_4_8_16_ablation_summary.json",
        "core14_coverage": root / "outputs/tapc/route_a_core_14_ablation_coverage/core_14_ablation_coverage.json",
        "vft_ood_threshold_transport": root / "outputs/tapc/route_a_vft_ood_threshold_transport_diagnostic/route_a_vft_ood_threshold_transport_diagnostic_summary.json",
    }
    evidence = {name: _result(path) for name, path in sources.items()}
    seal_path = root / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
    seal = _read_json(seal_path)
    sealed_test = seal["test_v3"]

    # These are requirement-level judgements, not an aggregate metric.  A true
    # entry means the requirement has direct source evidence; a false entry is
    # intentionally retained as an actionable gap.
    requirements = {
        "sealed_test_v3_intact": bool(
            sealed_test.get("access_authorized") is False
            and sealed_test.get("analysis_allowed") is False
            and sealed_test.get("used") is False
        ),
        "development_v12_unified_promotion": evidence["unified_v12"].get("passed") is True,
        "paired_statistics_and_effect_sizes": evidence["statistics"].get("passed") is True,
        "stratified_development_robustness": evidence["stratified_v12"].get("passed") is True,
        "engineering_cost_accounting": evidence["engineering"].get("passed") is True,
        "equal_budget_two_stage_baseline": evidence["equal_budget_two_stage"].get("passed") is True,
        "frozen_vft_mft_isrj_component_lesions": evidence["frozen_component_ablations"].get("passed") is True,
        "weak_threshold_5_10_15_sensitivity": evidence["weak_threshold_sensitivity"].get("passed") is True,
        "sqrt_vs_linear_vft_gate_ablation": evidence["sqrt_linear_gate_ablation"].get("passed") is True,
        "mft_residual_cap_removal_ablation": evidence["mft_residual_cap_removal"].get("passed") is True,
        # This is an executed, fixed-protocol mechanism ablation.  It is
        # intentionally evaluated as existence/completeness rather than a
        # candidate-promotion result because it compares supervision regimes.
        "vft_weak_only_supervision_ablation": bool(
            evidence["vft_weak_mixed_positive_ablation"].get("exists")
            and evidence["vft_weak_mixed_positive_ablation"].get("test_v3_used") is False
        ),
        "complete_candidate_external_ood": evidence["external_isrj_ood"].get("passed") is True,
        "dual_target_contract_repair": evidence["dual_target_contract"].get("passed") is True,
        "vft_isrj_cross_head_repair": evidence["vft_isrj_cross_head"].get("passed") is True,
        "all_14_current_candidate_ablations": bool(
            evidence["core14_coverage"].get("exists")
            and _read_json(sources["core14_coverage"]).get("all_14_direct_current_protocol") is True
            and _read_json(sources["core14_coverage"]).get("test_v3_used") is False
        ),
        "latest_multilabel_longtail_equal_protocol_baseline": bool(
            evidence["latest_mllt_reproducibility_audit"].get("exists")
            and _read_json(sources["latest_mllt_reproducibility_audit"]).get("equal_protocol_numerical_baseline_completed") is True
        ),
        "fresh_project_unseen_confirmation_after_current_model_change": False,
        "one_shot_test_v3_complete": False,
    }
    return {
        "status": "completed",
        "experiment": "RouteA_FinalInnovationEvidenceAudit",
        "locked_test_used": False,
        "test_v3_used": False,
        "test_v3_manifest": {
            "path": str(seal_path),
            "sha256": _sha256(seal_path),
            "access_authorized": sealed_test.get("access_authorized"),
            "analysis_allowed": sealed_test.get("analysis_allowed"),
            "used": sealed_test.get("used"),
        },
        "evidence": evidence,
        "requirement_checks": requirements,
        "completed_requirement_count": sum(requirements.values()),
        "total_requirement_count": len(requirements),
        "blocking_gaps": [name for name, value in requirements.items() if not value],
        "interpretation": (
            "The v12 candidate has positive development, statistics, stratified, and engineering evidence. "
            "It is not eligible for test-v3: the complete candidate failed external ISRJ-parameter OOD, "
            "the read-only OOD threshold diagnostic rejects threshold-only VFT repair, core ablations are development-only, the latest-MLLT comparator is incomplete, "
            "and v12 was previously consumed by the FC-PTV line so it is not project-globally fresh."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/tapc/route_a_final_innovation_evidence/audit.json",
    )
    args = parser.parse_args()
    payload = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "blocking_gaps": payload["blocking_gaps"], "test_v3_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
