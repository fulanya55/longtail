from pathlib import Path

from experiments.run_shared import build_run_plan, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_shared_runner_builds_four_class_dry_run_without_locked_test():
    config = load_config(PROJECT_ROOT / "configs" / "shared" / "four_class.yaml")
    plan = build_run_plan(
        config,
        loss_name="cb_asl",
        seed=2026,
        max_train_samples=128,
        max_eval_samples=64,
    )

    assert plan["class_mode"] == "four"
    assert plan["num_classes"] == 4
    assert plan["train_samples"] == 128
    assert plan["early_stop_samples"] == 64
    assert plan["calibration_samples"] == 64
    assert plan["threshold_samples"] == 64
    assert plan["loss"] == "cb_asl"
    assert plan["allow_locked_test"] is False
    assert len(plan["split_sha256"]) == 64


def test_shared_runner_full_plan_uses_all_four_class_development_samples():
    config = load_config(PROJECT_ROOT / "configs" / "shared" / "four_class.yaml")
    plan = build_run_plan(
        config,
        loss_name="bce",
        seed=2026,
        max_train_samples=0,
        max_eval_samples=0,
    )

    assert plan["train_samples"] == 4200
    assert plan["early_stop_samples"] == 600
    assert plan["calibration_samples"] == 600
    assert plan["threshold_samples"] == 600
