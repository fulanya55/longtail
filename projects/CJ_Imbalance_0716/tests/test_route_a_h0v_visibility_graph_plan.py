from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_h0v_is_a_fixed_no_fit_no_test_observability_gate() -> None:
    plan = (ROOT / "docs/plans/route_a_h0v_visibility_graph_observability_20260803.md").read_text(encoding="utf-8")
    source = (ROOT / "experiments/generate_route_a_h0v_visibility_graph_pairs.m").read_text(encoding="utf-8")
    assert "128 evenly spaced" in plan
    assert "AUROC >= 0.58" in plan
    assert "trained" in plan.lower()
    assert "test-v3" in plan
    assert "4100000000" in source
    assert "test_v3_used',false" in source
    assert "NoOverwrite" in source
