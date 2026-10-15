from pathlib import Path

import pytest

from experiments.audit_route_a_backbone_development_v1 import (
    RECEIVER_RANGE,
    SAMPLE_SEED_RANGE,
    preflight,
    ranges_disjoint,
    receiver_seed,
)
from experiments.smoke_route_a_backbone_b0_development_v1 import weighted_focal_loss
from experiments.run_route_a_backbone_b0_development_v1 import MODEL_SPECS
from cjlt.losses.classification import MultiLabelClassificationLoss
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_registered_receiver_range_matches_generator_derivation():
    assert (
        receiver_seed(SAMPLE_SEED_RANGE[0]),
        receiver_seed(SAMPLE_SEED_RANGE[1]),
    ) == RECEIVER_RANGE


def test_candidate_receiver_range_is_disjoint_from_nearby_registered_ranges():
    assert ranges_disjoint(RECEIVER_RANGE, (2498677742, 2498678381))
    assert ranges_disjoint(RECEIVER_RANGE, (2798677742, 2798678381))
    assert not ranges_disjoint(RECEIVER_RANGE, RECEIVER_RANGE)


def test_preflight_passes_without_opening_train_or_test_data():
    seal = (
        ROOT
        / "frozen"
        / "20260731_route_a_test_v3_sealed"
        / "test_v3_sealed_manifest.json"
    )
    result = preflight(seal)
    assert result["identity_audit_passed"] is True
    assert result["identity_overlap_count"] == 0
    assert result["test_files_opened"] is False
    assert result["test_v3_used"] is False


def test_train_audit_refuses_test_named_mat():
    from experiments.audit_route_a_backbone_development_v1 import audit_train

    with pytest.raises(ValueError, match="test-named"):
        audit_train(
            Path(r"E:\example\closedset_6class_test.mat"),
            ROOT / "outputs" / "never_created",
            ROOT
            / "frozen"
            / "20260731_route_a_test_v3_sealed"
            / "test_v3_sealed_manifest.json",
        )


def test_config_and_generator_are_e_drive_train_only_contracts():
    config = (ROOT / "configs/student_a/route_a_backbone_development_v1.yaml").read_text(
        encoding="utf-8"
    )
    generator = (
        ROOT / "experiments/generate_route_a_backbone_development_v1.m"
    ).read_text(encoding="utf-8")
    assert "E:\\LJQ_code" in config
    assert "generated_test_split: false" in config
    assert "test_v3_access_authorized: false" in config
    assert "build_closedset_manifest(cfg, registry, 'train')" in generator
    assert "build_closedset_manifest(cfg, registry, 'test')" not in generator


def test_b0_smoke_loss_uses_registered_non_saw_plus_point_two_saw_weighting():
    criterion = MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0)
    logits = torch.zeros(3, 6)
    targets = torch.tensor(
        [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    element = criterion.elementwise(logits, targets)
    expected = element[:, [0, 2, 3, 4, 5]].mean() + 0.2 * element[:, 1].mean()
    torch.testing.assert_close(weighted_focal_loss(criterion, logits, targets), expected)


def test_stage2_runner_specs_register_one_backbone_mechanism_per_candidate():
    b0 = MODEL_SPECS["b0"]
    m1 = MODEL_SPECS["m1"]
    m2 = MODEL_SPECS["m2"]
    m3 = MODEL_SPECS["m3"]
    m4k8 = MODEL_SPECS["m4k8"]
    m4k16 = MODEL_SPECS["m4k16"]
    assert b0["backbone_name"] == "dfms_iq"
    assert m1["backbone_name"] == "dfms_iq_anti_alias"
    assert b0["summary_name"] == "b0_summary.json"
    assert m1["summary_name"] == "m1_summary.json"
    assert m2["backbone_name"] == "dfms_iq_csfp"
    assert m2["summary_name"] == "m2_summary.json"
    assert m3["backbone_name"] == "dfms_iq_epb"
    assert m3["summary_name"] == "m3_summary.json"
    assert m4k8["backbone_name"] == "dfms_iq_local_topk8"
    assert m4k16["backbone_name"] == "dfms_iq_local_topk16"
    assert b0["display_name"] != m1["display_name"]
    assert b0["display_name"] != m2["display_name"]
    assert b0["display_name"] != m3["display_name"]
