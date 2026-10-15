import json

import pytest

from cj0724.p2n_freeze_candidate import build_receipt


def test_freeze_requires_passed_p2n(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"promotion_passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="P2n did not pass"):
        build_receipt(tmp_path, summary)
