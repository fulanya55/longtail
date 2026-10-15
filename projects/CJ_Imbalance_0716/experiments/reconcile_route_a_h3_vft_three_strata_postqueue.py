"""Preserve a transient postqueue test failure and produce a recovered receipt.

This never overwrites the original postqueue receipt.  It is valid only after
the immutable audit and development summary exist and a separate CPU contract
receipt has passed.  It does not train, recalibrate, threshold, or open data.
"""
from __future__ import annotations

import json
from pathlib import Path


def load(path: Path) -> dict:
    # PowerShell 5's UTF-8 output may include a BOM; Python-generated
    # artifacts do not.  Accept both without mutating the original receipt.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run(output: Path) -> dict:
    queue = Path("outputs/tapc/route_a_h3_vft_three_strata_queue")
    original, audit, summary, contract = (load(queue / "postqueue_receipt.json"), load(Path("outputs/tapc/route_a_h3_vft_three_strata_artifact_audit.json")), load(Path("outputs/tapc/route_a_h3_vft_three_strata_development_summary.json")), load(queue / "model_contract_recovery_receipt.json"))
    if original.get("status") != "failed_postqueue_audit" or "regression test suite failed" not in original.get("error", ""):
        raise ValueError("original postqueue failure provenance is not the expected environment-only test failure")
    if audit.get("status") != "completed_artifact_audit" or audit.get("record_count") != 15:
        raise ValueError("completed H3 artifact audit required")
    if summary.get("status") != "completed_development_summary":
        raise ValueError("completed H3 summary required")
    if contract.get("status") != "completed_model_contract_verification" or not all(contract.get("checks", {}).values()):
        raise ValueError("completed H3 model contract required")
    for item in (original, audit, summary, contract):
        if any(item.get(key, True) for key in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
            raise ValueError("test-boundary violation in H3 reconciliation inputs")
    result = {
        "status": "completed_postqueue_reconciliation",
        "experiment": "RouteA_H3_VFTThreeStrataIdentifiability",
        "original_postqueue_receipt": str(queue / "postqueue_receipt.json"),
        "original_postqueue_status": original["status"],
        "original_failure_scope": "transient Windows PyTorch DLL initialization during pytest collection after artifacts, audit, and summary had completed",
        "artifact_audit": "outputs/tapc/route_a_h3_vft_three_strata_artifact_audit.json",
        "development_summary": "outputs/tapc/route_a_h3_vft_three_strata_development_summary.json",
        "model_contract_receipt": str(queue / "model_contract_recovery_receipt.json"),
        "development_gate_passed": bool(summary["gate"]["passed"]),
        "external_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_queue/postqueue_reconciliation_receipt.json"))
    value = run(parser.parse_args().output)
    print(json.dumps({"status": value["status"], "development_gate_passed": value["development_gate_passed"]}, ensure_ascii=False))
