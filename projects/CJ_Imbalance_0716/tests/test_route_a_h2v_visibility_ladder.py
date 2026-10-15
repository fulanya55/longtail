import json
from pathlib import Path

from experiments.audit_route_a_h2v_visibility_ladder import BASE_SEED, COUNT


ROOT = Path(__file__).resolve().parents[1]


def test_h2v_plan_and_generator_bind_the_new_visibility_ladder():
    plan = (ROOT / "docs/plans/route_a_h2v_conditional_complementarity_admission_20260803.md").read_text(encoding="utf-8")
    generator = (ROOT / "experiments/generate_route_a_h2v_mft_visibility_ladder.m").read_text(encoding="utf-8")
    for token in ("3,920,000,000", "0, 4, and 8", "test-v3", "0.93"):
        assert token in plan
    for token in ("baseSeed=3920000000", "repetition<=14", "repetition<=27", "repetition<=40", "test_v3_used", "NoOverwrite"):
        assert token in generator


def test_h2v_identity_selection_is_isolated_and_never_opens_a_test_file():
    path = ROOT / "outputs/tapc/route_a_h2v_visibility_ladder_generation/pre_generation_identity_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sample_seed_base"] == BASE_SEED
    assert payload["sample_count"] == COUNT
    assert payload["identity_overlap_count"] == 0
    assert payload["test_files_opened"] is False


def test_h2v_dry_run_receipt_does_not_generate_or_open_test_data():
    path = ROOT / "outputs/tapc/route_a_h2v_visibility_ladder_generation/dry_run_generation_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["executed"] is False
    assert payload["test_split_generated"] is False
    assert payload["test_files_opened"] is False
    assert payload["test_v3_used"] is False
