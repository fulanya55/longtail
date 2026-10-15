import json
from pathlib import Path

from cj0724.p5u_two_pool_confirmation_execution import validate_confirmation_context
from cj0724.p5b1_jnr_physical_oracle import sha256_file


def test_p5u_confirmation_requires_three_passing_dev_pools_and_direct_terminal(tmp_path):
    project = Path(__file__).resolve().parents[1]
    protocol = project / "configs" / "p5u_direct_supervision_preregister_20260806.json"
    confirmation = project / "configs" / "p5u_two_pool_confirmation_preregister_20260806.json"
    result = tmp_path / "development.json"
    result.write_text(json.dumps({
        "development_passed": True,
        "preregister_sha256": sha256_file(protocol),
        "development": {"a": {"gates": {"all": True}}, "b": {"gates": {"all": True}}, "c": {"gates": {"all": True}}},
    }), encoding="utf-8")
    started, terminal = tmp_path / "started.json", tmp_path / "terminal.json"
    started.write_text(json.dumps({"child_pid": 123}), encoding="utf-8")
    terminal.write_text(json.dumps({"stage": "child_terminal", "returncode": 0, "child_pid": 123}), encoding="utf-8")
    context = validate_confirmation_context(project, protocol, confirmation, result, started, terminal)
    assert context["direct_executor_pid"] == 123
    assert context["reserved_confirmation_seeds"] == [3340000000, 3350000000]
