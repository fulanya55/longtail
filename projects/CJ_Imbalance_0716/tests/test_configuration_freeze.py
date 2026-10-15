import json

from experiments.freeze_configuration import (
    build_freeze_payloads,
    write_configuration_freeze,
)


def test_configuration_freeze_writes_hashed_files_without_test_authorization(tmp_path):
    payloads = build_freeze_payloads()
    manifest = write_configuration_freeze(tmp_path, payloads)

    assert len(manifest["files"]) == 8
    assert manifest["locked_test_authorized"] is False
    for name, metadata in manifest["files"].items():
        path = tmp_path / name
        assert path.is_file()
        assert len(metadata["sha256"]) == 64
        content = json.loads(path.read_text(encoding="utf-8"))
        assert content.get("allow_locked_test", False) is False
    assert (tmp_path / "configuration_freeze.json").is_file()

