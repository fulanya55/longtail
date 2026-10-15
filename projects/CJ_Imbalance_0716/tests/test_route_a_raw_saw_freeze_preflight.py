import hashlib
from pathlib import Path

import pytest

from experiments.freeze_route_a_raw_saw_label_routing import (
    _assert_completed_unlocked,
    _assert_hash,
    _selected_raw_threshold,
    _validate_summary,
)


def _summary():
    return {
        "status": "completed",
        "locked_test_used": False,
        "next_decision": "freeze_raw_saw_label_routing_and_prepare_locked_test_request",
        "confirmation": {
            "promotion_gate": {"passed": True, "passed_seed_count": 4},
            "per_run": [
                {"seed": 2041, "variant": "I1_RawPlatt", "selected_threshold": 0.35}
            ],
        },
        "label_routing": {
            "promotion_gate": {"passed": True, "passed_seed_count": 5}
        },
    }


def test_summary_validation_requires_clean_frozen_gates():
    _validate_summary(_summary())
    invalid = _summary()
    invalid["label_routing"]["promotion_gate"]["passed_seed_count"] = 4
    with pytest.raises(ValueError):
        _validate_summary(invalid)


def test_selected_raw_threshold_is_seed_and_variant_specific():
    assert _selected_raw_threshold(_summary(), 2041) == 0.35
    with pytest.raises(ValueError):
        _selected_raw_threshold(_summary(), 2042)


def test_assert_completed_unlocked_rejects_locked_test_use():
    with pytest.raises(ValueError):
        _assert_completed_unlocked(
            {"status": "completed", "locked_test_used": True}, "payload"
        )


def test_assert_hash_detects_file_drift(tmp_path):
    path = Path(tmp_path) / "value.txt"
    path.write_text("frozen", encoding="utf-8")
    expected = hashlib.sha256(b"frozen").hexdigest()
    actual = _assert_hash(path, expected, "value")
    assert actual == expected
    path.write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError):
        _assert_hash(path, actual, "value")
