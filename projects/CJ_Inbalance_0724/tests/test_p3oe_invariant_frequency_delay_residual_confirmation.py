from pathlib import Path
import json

from cj0724.p3oe_invariant_frequency_delay_residual_confirmation import _load_development


def test_confirmation_rejects_unfrozen_or_failed_development_summary():
    project = Path(__file__).resolve().parents[1]
    preregistration = json.loads((project / "configs/p3oe_invariant_frequency_delay_residual_confirmation_preregister_20260802.json").read_text(encoding="utf-8"))
    summary = _load_development(project, preregistration)
    assert summary["development_passed"]
    assert all(summary["promotion_gates"].values())
    assert [int(row["seed"]) for row in summary["results"]] == [2041, 2042, 2043]
