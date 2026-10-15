"""Explicit authorization gate for locked test data."""

import hashlib
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


class LockedTestAccessError(PermissionError):
    pass


def sha256_file(path: PathLike) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_locked_test_access(
    allow_locked_test: bool,
    configuration_freeze_path: PathLike,
    expected_freeze_sha256: str,
) -> Path:
    if not bool(allow_locked_test):
        raise LockedTestAccessError(
            "Locked test access requires allow_locked_test=True."
        )
    freeze_path = Path(configuration_freeze_path)
    if not freeze_path.is_file():
        raise LockedTestAccessError(
            "Configuration freeze file does not exist: {}".format(freeze_path)
        )
    expected = str(expected_freeze_sha256).strip().lower()
    actual = sha256_file(freeze_path)
    if expected != actual:
        raise LockedTestAccessError(
            "Configuration freeze SHA-256 mismatch: expected {}, got {}.".format(
                expected, actual
            )
        )
    return freeze_path.resolve()

