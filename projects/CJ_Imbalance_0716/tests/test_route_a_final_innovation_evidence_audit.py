import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_innovation_audit_preserves_test_v3_and_exposes_blockers(tmp_path):
    output = tmp_path / "audit.json"
    subprocess.run(
        [sys.executable, "experiments/audit_route_a_final_innovation_evidence.py", "--output", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["test_v3_used"] is False
    assert payload["test_v3_manifest"]["access_authorized"] is False
    assert payload["test_v3_manifest"]["analysis_allowed"] is False
    assert payload["test_v3_manifest"]["used"] is False
    assert payload["requirement_checks"]["development_v12_unified_promotion"] is True
    assert payload["requirement_checks"]["equal_budget_two_stage_baseline"] is True
    assert payload["requirement_checks"]["frozen_vft_mft_isrj_component_lesions"] is True
    assert payload["requirement_checks"]["weak_threshold_5_10_15_sensitivity"] is True
    assert payload["requirement_checks"]["sqrt_vs_linear_vft_gate_ablation"] is True
    assert payload["requirement_checks"]["mft_residual_cap_removal_ablation"] is True
    assert payload["requirement_checks"]["complete_candidate_external_ood"] is False
    assert payload["requirement_checks"]["vft_isrj_cross_head_repair"] is False
    assert payload["requirement_checks"]["one_shot_test_v3_complete"] is False
