import json
from pathlib import Path

from experiments.audit_route_a_frequency_v2_stage0 import NEW_RANGE, overlap


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "experiments" / "generate_route_a_frequency_development_v2.m"
PREREG = ROOT / "docs" / "plans" / "route_a_frequency_reconstruction_v2_preregistration_20260808.md"


def test_generator_contract_is_fresh_paired_train_only():
    text = GENERATOR.read_text(encoding="utf-8")
    assert "baseSeed = 3930000000" in text
    assert "dual-axis-frequency-development-v2" in text
    assert "requiredLevels = [20, 30]" in text
    assert "'train', existingPolicy" in text
    assert "existingPolicy = 'error'" in text
    assert "{'SawFM'}" in text
    assert "test_views_generated = false" in text
    assert "test_v3_used = false" in text


def test_preregistration_freezes_primary_counts_and_gates():
    text = PREREG.read_text(encoding="utf-8")
    assert "`[1500,75,824,453,249,137]`" in text
    assert "Label-IF20" in text and "Combo-IF20" in text
    assert "`>=+0.005`" in text
    assert "5,000" in text
    assert "不启用 optional stage4 adapter" in text


def test_new_seed_range_basic_contract():
    assert NEW_RANGE == (3_930_000_001, 3_930_025_200)
    assert not overlap(NEW_RANGE, (3_920_000_001, 3_923_025_200))
    assert not overlap(NEW_RANGE, (3_990_000_001, 3_993_025_200))
    assert overlap(NEW_RANGE, NEW_RANGE)


def test_terminal_n_metadata_remains_test_free():
    receipt = json.loads(
        (ROOT / "outputs" / "route_a_dual_axis_next" / "terminal_N_closure" / "run_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "completed"
    assert receipt["locked_test_used"] is False
    assert receipt["test_v2_used"] is False
    assert receipt["test_v3_used"] is False

