import json
from pathlib import Path

from experiments.select_route_a_h3_frozen_confirmation_identities import BASE_SEED, COUNT


ROOT = Path(__file__).resolve().parents[1]


def test_h3_protocol_uses_a_new_standard_train_only_identity_interval():
    plan = (ROOT / "docs/plans/route_a_h3_frozen_unified_external_confirmation_20260803.md").read_text(encoding="utf-8")
    generator = (ROOT / "experiments/generate_route_a_h3_frozen_confirmation_dataset.m").read_text(encoding="utf-8")
    assert "134,000,000" in plan and "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2" in plan
    for token in ("cfg.seed=134000000", "'train'", "test_split_generated", "NoOverwrite"):
        assert token in generator


def test_h3_pre_generation_identity_selection_and_dry_run_preserve_test_boundary():
    selection = json.loads((ROOT / "outputs/tapc/route_a_h3_frozen_confirmation_generation/pre_generation_identity_selection.json").read_text(encoding="utf-8"))
    dry = json.loads((ROOT / "outputs/tapc/route_a_h3_frozen_confirmation_generation/dry_run_generation_receipt.json").read_text(encoding="utf-8"))
    assert selection["base_sample_seed"] == BASE_SEED and selection["sample_count"] == COUNT
    assert selection["identity_overlap_count"] == 0 and selection["test_files_opened"] is False
    assert dry["executed"] is False and dry["test_split_generated"] is False and dry["test_v3_used"] is False
