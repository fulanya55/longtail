"""Write explicit Route-B stop receipts from a frozen R1 aggregate."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    aggregate_path = project / "docs" / "receipts" / "route_b_r1_aggregate_receipt.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    failed = [variant for variant, pair in aggregate["pairwise"].items() if not pair["mean_parameter_relative_improvement_pass"]]
    receipt = {
        "stage": "R1_stop",
        "status": "stopped_parameter_gate_failure",
        "run_dir": aggregate["run_dir"],
        "failed_branches": failed,
        "gate_basis": aggregate["gates"],
        "thresholds": aggregate["thresholds"],
        "reason": "Each conditional branch failed the preregistered mean parameter Macro-NMAE relative-improvement gate versus B1.",
        "classification_safety_preserved": aggregate["gates"]["classification_safety"],
        "locked_test_used": aggregate["locked_test_used"],
        "test_mat_opened": False,
        "prohibited_rescue_actions": ["threshold_retuning", "resampling", "model_enlargement", "test_feedback"],
        "next": "retain failure evidence; do not promote B2/B3/B4; evaluate only separately preregistered branches with fresh receipts",
    }
    out = project / "docs" / "receipts" / "route_b_abort_receipt.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    report = project / "docs" / "reports" / "route_b_r1_stop.md"
    lines = ["# Route B R1 stop receipt", "", "- Status: **stopped_parameter_gate_failure**", f"- Failed branches: `{', '.join(failed)}`", "- Classification safety: `passed`", "- Parameter improvement: `failed`", "- Locked test used: `false`", "", "The preregistered parameter gate failed for every conditional branch. No threshold retuning, resampling, model enlargement or test feedback was used. The failure is retained as evidence."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
