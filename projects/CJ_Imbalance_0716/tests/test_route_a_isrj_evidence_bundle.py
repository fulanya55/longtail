from __future__ import annotations

import json
from pathlib import Path

from experiments.build_route_a_isrj_evidence_bundle import build


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _summary() -> dict:
    metrics = {"mean": 0.1, "bootstrap_95_ci": [0.01, 0.2]}
    return {
        "status": "completed", "external_refit": False, "locked_test_used": False, "test_v3_used": False,
        "external_gate": {"passed": True}, "experiment": "x", "evidence_role": "y", "external_data_sha256": "data", "seeds": [1, 2, 3, 4, 5],
        "b2_vs_b0": {"aggregate": {**{key: metrics for key in ("isrj_aupr", "isrj_weak_auroc", "isrj_weak_recall", "isrj_fpr", "macro_aupr", "macro_f1")}, "positive_weak_auroc_seeds": 5, "positive_weak_recall_seeds": 5}},
    }


def test_builds_read_only_bundle(tmp_path: Path) -> None:
    h0g, h0h, h0s, colabel, seal, output = (tmp_path / name for name in ("h0g.json", "h0h.json", "h0s.json", "colabel.json", "seal.json", "out.json"))
    _write_json(h0g, _summary()); _write_json(h0h, _summary()); _write_json(h0s, _summary())
    _write_json(colabel, {"status": "completed_nonselection_frozen_prediction_audit", "selection_or_refit_performed": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
    _write_json(seal, {"test_v3": {"access_authorized": False, "analysis_allowed": False, "used": False}, "test_v3_used": False, "historic_locked_test_used": False})
    args = type("Args", (), {"h0g_summary": h0g, "h0h_summary": h0h, "h0s_summary": h0s, "colabel_audit": colabel, "test_v3_manifest": seal, "output": output})()
    value = build(args)
    assert output.exists()
    assert value["selection_or_refit_performed"] is False
    assert value["test_v3_used"] is False
    assert len(value["external_replications"]) == 3


def test_rejects_open_test_manifest(tmp_path: Path) -> None:
    h0g, h0h, h0s, colabel, seal, output = (tmp_path / name for name in ("h0g.json", "h0h.json", "h0s.json", "colabel.json", "seal.json", "out.json"))
    _write_json(h0g, _summary()); _write_json(h0h, _summary()); _write_json(h0s, _summary())
    _write_json(colabel, {"status": "completed_nonselection_frozen_prediction_audit", "selection_or_refit_performed": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
    _write_json(seal, {"test_v3": {"access_authorized": True, "analysis_allowed": False, "used": False}, "test_v3_used": False, "historic_locked_test_used": False})
    args = type("Args", (), {"h0g_summary": h0g, "h0h_summary": h0h, "h0s_summary": h0s, "colabel_audit": colabel, "test_v3_manifest": seal, "output": output})()
    try:
        build(args)
    except ValueError as exc:
        assert "sealed" in str(exc)
    else:
        raise AssertionError("open test manifest should fail")
