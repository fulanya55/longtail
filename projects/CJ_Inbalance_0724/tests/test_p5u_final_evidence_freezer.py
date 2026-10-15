import json
from pathlib import Path

from cj0724.p5u_final_evidence_freezer import freeze


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _terminal(root: Path, pid: int):
    _write(root / "supervisor_child_started.json", {"child_pid": pid})
    _write(root / "supervisor_terminal.json", {"stage": "child_terminal", "returncode": 0, "child_pid": pid})


def _pool(ratio: float):
    return {"gates": {"all": True}, "rows": [{"timing": {"median_ratio_shared_dense_over_reference": ratio}}]}


def test_p5u_final_freezer_requires_all_receipts_and_writes_narrow_claim(tmp_path):
    project = Path(__file__).resolve().parents[1]
    parent, confirmation = project / "configs" / "p5u_direct_supervision_preregister_20260806.json", project / "configs" / "p5u_two_pool_confirmation_preregister_20260806.json"
    dev, conf, output = tmp_path / "dev", tmp_path / "confirmation", tmp_path / "final.json"
    _terminal(dev, 11); _terminal(conf, 22)
    _write(dev / "result.json", {"development_passed": True, "development": {"a": _pool(0.7), "b": _pool(0.8), "c": _pool(0.6)}})
    _write(conf / "result.json", {"confirmation_passed": True, "confirmation": {"d": _pool(0.75), "e": _pool(0.74)}})
    assert freeze(project, parent, confirmation, dev, conf, output) is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "p5_final_confirmed"
    assert payload["development_latency_ratio_summary"]["count"] == 3
    assert payload["confirmation_latency_ratio_summary"]["count"] == 2
