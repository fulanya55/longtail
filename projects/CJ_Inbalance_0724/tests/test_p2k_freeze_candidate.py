import json

import pytest

from cj0724.p2k_freeze_candidate import build_receipt


def test_freeze_fails_closed_without_confirmation(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"confirmation_passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not pass"):
        build_receipt(tmp_path, summary)
