"""Validate M1-M4 closures and write the Route-A backbone stop manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tapc/route_a_backbone_scientific_stop_20260806"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    model_roots = {
        "b0": ROOT / "outputs/tapc/route_a_backbone_b0_development_v1",
        "m1": ROOT / "outputs/tapc/route_a_backbone_m1_development_v1",
        "m2": ROOT / "outputs/tapc/route_a_backbone_m2_development_v1",
        "m3": ROOT / "outputs/tapc/route_a_backbone_m3_development_v1",
        "m4": ROOT / "outputs/tapc/route_a_backbone_m4_k16_development_v1",
    }
    summaries = {
        "b0": model_roots["b0"] / "b0_summary.json",
        "m1": model_roots["m1"] / "m1_summary.json",
        "m2": model_roots["m2"] / "m2_summary.json",
        "m3": model_roots["m3"] / "m3_summary.json",
        "m4": model_roots["m4"] / "m4_k16_summary.json",
    }
    comparisons = {
        name: model_roots[name] / "comparison_to_b0.json"
        for name in ("m1", "m2", "m3", "m4")
    }
    payloads = {name: _read(path) for name, path in summaries.items()}
    for name, payload in payloads.items():
        if payload.get("status") != "completed" or payload.get("formal_protocol") is not True:
            raise ValueError(f"{name} formal summary is incomplete")
        for flag in ("test_v2_used", "test_v3_used", "locked_test_used", "test_files_opened"):
            if payload.get(flag) is not False:
                raise ValueError(f"{name} evidence boundary failed: {flag}")
        terminal = _read(model_roots[name] / "terminal_receipt.json")
        if terminal.get("status") != "completed" or terminal.get("test_v3_used") is not False:
            raise ValueError(f"{name} terminal receipt is incomplete")
    comparison_payloads = {name: _read(path) for name, path in comparisons.items()}
    if any(
        payload.get("decision") != "fail_stage2_close_candidate"
        or payload.get("test_v3_used") is not False
        for payload in comparison_payloads.values()
    ):
        raise ValueError("not every Stage-2 candidate has a validated failure closure")

    identity_audit = ROOT / "outputs/tapc/route_a_backbone_development_v1/identity_role_audit.json"
    identity = _read(identity_audit)
    if identity.get("identity_overlap_count") != 0 or identity.get("test_v3_used") is not False:
        raise ValueError("fresh development identity audit is not intact")

    plans = [
        ROOT / "docs/plans/route_a_backbone_strengthening_goal_prompt_20260806.md",
        ROOT / "docs/plans/route_a_backbone_stage0_preregistration_20260806.md",
        ROOT / "docs/plans/route_a_backbone_stage1_b0_preregistration_20260806.md",
        ROOT / "docs/plans/route_a_backbone_m1_preregistration_20260806.md",
        ROOT / "docs/plans/route_a_backbone_m2_preregistration_20260806.md",
        ROOT / "docs/plans/route_a_backbone_m3_preregistration_20260806.md",
        ROOT / "docs/plans/route_a_backbone_m4_preregistration_20260806.md",
    ]
    reports = [
        ROOT / "docs/results/route_a_backbone_stage0_results_20260806.md",
        ROOT / "docs/results/route_a_backbone_stage1_b0_results_20260806.md",
        ROOT / "docs/results/route_a_backbone_m1_results_20260806.md",
        ROOT / "docs/results/route_a_backbone_m2_results_20260806.md",
        ROOT / "docs/results/route_a_backbone_m3_results_20260806.md",
        ROOT / "docs/results/route_a_backbone_m4_results_20260806.md",
        ROOT / "docs/results/route_a_backbone_scientific_stop_20260806.md",
        ROOT / "docs/reports/route_a_backbone_strengthening_technical_design_and_sci_evidence_20260807.md",
        ROOT / "docs/reports/route_a_backbone_goal_completion_audit_20260807.md",
    ]
    selection = ROOT / "outputs/tapc/route_a_backbone_m4_selection/selection_receipt.json"
    source_files = [
        ROOT / "cjlt/models/phase_multiscale.py",
        ROOT / "cjlt/models/backbone_adapter.py",
        ROOT / "experiments/run_route_a_backbone_b0_development_v1.py",
        ROOT / "experiments/compare_route_a_backbone_candidate.py",
        ROOT / "experiments/select_route_a_backbone_m4_topk.py",
        ROOT / "experiments/finalize_route_a_backbone_scientific_stop.py",
        ROOT / "tests/test_backbone_adapter.py",
        ROOT / "tests/test_route_a_backbone_development_v1.py",
    ]
    artifacts = [
        identity_audit,
        selection,
        *plans,
        *reports,
        *source_files,
        *summaries.values(),
        *comparisons.values(),
        *(model_roots[name] / "terminal_receipt.json" for name in model_roots),
    ]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing stop artifacts: " + ", ".join(missing))

    manifest = {
        "status": "completed_scientific_stop",
        "decision": "stop_same_source_backbone_strengthening",
        "reason": "M1, M2, M3, and M4 all failed one or more preregistered Stage-2 promotion gates.",
        "data_sha256": payloads["b0"]["data_sha256"],
        "seeds": payloads["b0"]["seeds"],
        "identity_overlap_count": 0,
        "candidate_decisions": {
            name: comparison_payloads[name]["decision"]
            for name in ("m1", "m2", "m3", "m4")
        },
        "stage3_controls_executed": False,
        "stage3_controls_omission_reason": "No Stage-2 candidate passed the B0 promotion gates.",
        "combination_executed": False,
        "confirmation_executed": False,
        "same_source_actions_forbidden": [
            "widen_backbone",
            "add_attention",
            "retune_thresholds",
            "retune_failed_module",
            "combine_failed_modules",
            "reuse_development_v1_for_candidate_selection",
        ],
        "allowed_next_research_scope": [
            "new_observation_protocol",
            "real_or_hardware_in_the_loop_data",
            "paper_scope_reduction",
        ],
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "artifact_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in artifacts
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "scientific_stop_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terminal = {
        "status": "completed_scientific_stop",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "test_v3_used": False,
    }
    (OUTPUT / "terminal_receipt.json").write_text(
        json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
