import hashlib

import pytest

from cjlt.engine.locked_test import (
    LockedTestAccessError,
    validate_locked_test_access,
)


def test_locked_test_is_rejected_by_default(tmp_path):
    freeze = tmp_path / "configuration_freeze.json"
    freeze.write_text('{"frozen": true}', encoding="utf-8")
    digest = hashlib.sha256(freeze.read_bytes()).hexdigest()

    with pytest.raises(LockedTestAccessError, match="allow_locked_test"):
        validate_locked_test_access(False, freeze, digest)


def test_locked_test_requires_existing_freeze_file(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(LockedTestAccessError, match="does not exist"):
        validate_locked_test_access(True, missing, "0" * 64)


def test_locked_test_rejects_wrong_freeze_hash(tmp_path):
    freeze = tmp_path / "configuration_freeze.json"
    freeze.write_text('{"frozen": true}', encoding="utf-8")
    with pytest.raises(LockedTestAccessError, match="SHA-256"):
        validate_locked_test_access(True, freeze, "0" * 64)


def test_locked_test_accepts_explicit_permission_and_matching_hash(tmp_path):
    freeze = tmp_path / "configuration_freeze.json"
    freeze.write_text('{"frozen": true}', encoding="utf-8")
    digest = hashlib.sha256(freeze.read_bytes()).hexdigest()

    validated = validate_locked_test_access(True, freeze, digest)

    assert validated == freeze.resolve()

