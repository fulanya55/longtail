import json

from cj0724 import p2o_preflight


def test_preflight_fails_when_freeze_hash_differs(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("x", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "artifacts": {"a": {"path": str(artifact), "sha256": p2o_preflight.sha256_file(artifact)}},
        "locked_test_used": False,
    }), encoding="utf-8")
    monkeypatch.setattr(p2o_preflight, "EXPECTED_FREEZE_SHA256", "0" * 64)
    result = p2o_preflight.verify_freeze(tmp_path, freeze)
    assert not result["passed"]
    assert "P2n method-freeze hash changed" in result["errors"]
