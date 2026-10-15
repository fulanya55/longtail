import json

import pytest

from cjlt.engine.run_manager import (
    RunConfigurationConflict,
    configuration_sha256,
    decide_run,
    stable_run_arguments,
)


def test_run_manager_skips_only_matching_completed_receipt(tmp_path):
    payload = {"variant": "A4", "seed": 2026}
    config_hash = configuration_sha256(payload)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run_receipt.json").write_text(
        json.dumps({"status": "completed", "config_sha256": config_hash}),
        encoding="utf-8",
    )

    decision = decide_run(output_dir, config_hash, resume=True)

    assert decision.action == "skip"


def test_run_manager_rejects_hash_mismatch_in_existing_directory(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run_receipt.json").write_text(
        json.dumps({"status": "completed", "config_sha256": "wrong"}),
        encoding="utf-8",
    )

    with pytest.raises(RunConfigurationConflict):
        decide_run(output_dir, configuration_sha256({"seed": 2026}), resume=True)


def test_operational_resume_flags_do_not_change_scientific_configuration_hash():
    first = stable_run_arguments(
        {"epochs": 60, "resume": False, "dry_run": False, "output_dir": "one"}
    )
    resumed = stable_run_arguments(
        {"epochs": 60, "resume": True, "dry_run": False, "output_dir": "two"}
    )

    assert first == resumed == {"epochs": 60}
