from __future__ import annotations

from pathlib import Path

from experiments.audit_route_a_goal_completion_h3 import PROJECT_ROOT, build_audit


def test_h3_goal_audit_keeps_causality_and_confirmation_separate() -> None:
    audit = build_audit(PROJECT_ROOT)
    checks = audit["requirement_checks"]
    assert checks["full_frozen_candidate_fresh_identity_confirmation"] is True
    assert checks["c3_beats_same_capacity_c4_on_visibility_causal_gate"] is False
    assert checks["classic_and_recent_loss_baselines_same_protocol"] is True
    assert checks["full_candidate_parameter_or_noise_ood_robustness"] is False
    assert audit["test_v3_authorization"] is False


def test_h3_goal_audit_only_reads_public_seal_metadata() -> None:
    audit = build_audit(PROJECT_ROOT)
    seal = audit["test_v3_seal"]
    assert Path(seal["path"]).name == "test_v3_sealed_manifest.json"
    assert seal["intact"] is True
    assert audit["test_v3_used"] is False
