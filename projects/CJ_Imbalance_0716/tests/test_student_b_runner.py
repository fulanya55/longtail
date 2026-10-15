from pathlib import Path

from experiments.run_student_b import (
    build_student_b_plan,
    load_config,
    student_b_variant_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_student_b_variant_matrix_has_distinct_attributable_stages():
    variants = student_b_variant_matrix()

    assert list(variants) == ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
    assert variants["B0"]["teacher"] is False
    assert variants["B2"]["teacher"] is True
    assert variants["B3"]["dynamic_threshold"] is True
    assert variants["B4"]["prototype_gate"] is True
    assert variants["B5"]["feature_consistency"] is True
    assert variants["B6"]["parameter_heads"] is True


def test_student_b_plan_uses_nested_twenty_percent_labeled_subset():
    config = load_config(PROJECT_ROOT / "configs" / "student_b" / "daps_param.yaml")
    plan = build_student_b_plan(
        config,
        labeled_ratio=0.20,
        seed=2026,
        max_labeled_samples=0,
        max_unlabeled_samples=0,
        max_eval_samples=64,
    )

    assert plan["labeled_samples"] == 840
    assert plan["unlabeled_samples"] == 3360
    assert plan["early_stop_samples"] == 64
    assert len(plan["semi_split_sha256"]) == 64
    assert set(plan["labeled_indices"]).isdisjoint(set(plan["unlabeled_indices"]))
