from __future__ import annotations

from pathlib import Path

from experiments.select_route_a_h5o_frozen_unified_ood_identities import BASE_SEED, COUNT, RECEIVER_OFFSET, build_payload


ROOT = Path(__file__).resolve().parents[1]


def test_h5o_preregisters_new_identity_interval_and_no_test_split() -> None:
    plan = (ROOT / "docs/plans/route_a_h5o_frozen_unified_combined_ood_20260803.md").read_text(encoding="utf-8")
    generator = (ROOT / "experiments/generate_route_a_h5o_frozen_unified_ood_base.m").read_text(encoding="utf-8")
    assert "850000000" in plan and "rho=0.65" in plan
    assert "cfg.seed = 850000000" in generator
    assert "[25e-6,30e-6]" in generator and "[9,9]" in generator
    assert "'train'" in generator and "test_split_generated" in generator


def test_h5o_identity_contract_is_explicit_and_not_a_test_contract() -> None:
    assert BASE_SEED == 850_000_000 and COUNT == 5_040
    assert BASE_SEED + RECEIVER_OFFSET == 1_048_677_743
    source = (ROOT / "experiments/audit_route_a_h5o_frozen_unified_ood_source.py").read_text(encoding="utf-8")
    assert "test-v3 is not sealed" in source
    assert "identity_audit_passed" in source


def test_h5o_selection_builder_is_read_only_by_contract() -> None:
    payload = build_payload(ROOT / "does_not_exist")
    assert payload["identity_overlap_count"] == 0
    assert payload["test_files_opened"] is False
