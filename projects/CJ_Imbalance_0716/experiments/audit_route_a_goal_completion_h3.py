"""Audit Route-A goal completion after the H3 frozen external confirmation.

The audit is read-only with respect to data and models.  It reads only existing
summaries, manifests, and the public test-v3 seal metadata.  In particular, it
never opens test-v3 I/Q or labels and cannot authorize a test run merely from a
positive H3 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path), "sha256": None}
    payload = _read(path)
    return {
        "exists": True,
        "path": str(path),
        "sha256": _sha256(path),
        "status": payload.get("status"),
        "locked_test_used": payload.get("locked_test_used"),
        "test_v2_used": payload.get("test_v2_used"),
        "test_v3_used": payload.get("test_v3_used"),
    }


def _completed(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "completed" and payload.get("smoke") is not True


def build_audit(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paths = {
        "c3_c4_unified": root / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3/route_a_unified_rawsaw_vtse_vbf_v3_summary.json",
        "h3_frozen_confirmation": root / "outputs/tapc/route_a_h3_frozen_unified_confirmation/summary.json",
        "h5o_frozen_ood_confirmation": root / "outputs/tapc/route_a_h5o_frozen_unified_ood/summary.json",
        "core14": root / "outputs/tapc/route_a_core_14_ablation_coverage/core_14_ablation_coverage.json",
        "classic_losses": root / "outputs/tapc/route_a_development_v3_mainstream_loss_baselines/route_a_development_v3_mainstream_loss_baselines_summary.json",
        "dr_loss": root / "outputs/tapc/route_a_development_v3_dr_loss_baseline/route_a_development_v3_dr_loss_baseline_summary.json",
        # The default root was later used for a one-seed smoke and therefore
        # cannot stand in for the preserved five-seed formal R1 evidence.
        "ral_loss": root / "outputs/tapc/route_a_development_v3_ral_loss_baseline_formal_r1/route_a_development_v3_ral_loss_baseline_summary.json",
        "mllt_reproducibility": root / "outputs/tapc/route_a_latest_mllt_reproducibility_audit/latest_mllt_reproducibility_audit.json",
        "external_ood": root / "outputs/tapc/route_a_final_shared_policy_isrj_ood_frozen/route_a_final_shared_policy_isrj_ood_frozen_summary.json",
        "statistics": root / "outputs/tapc/route_a_final_shared_policy_evidence/statistical_audit.json",
        "engineering": root / "outputs/tapc/route_a_final_shared_policy_engineering/engineering_audit.json",
    }
    source_receipts = {name: _source(path) for name, path in paths.items()}
    payloads = {name: _read(path) for name, path in paths.items()}
    seal_path = root / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
    seal = _read(seal_path)["test_v3"]

    c3 = payloads["c3_c4_unified"]
    h3 = payloads["h3_frozen_confirmation"]
    h5o = payloads["h5o_frozen_ood_confirmation"]
    core14 = payloads["core14"]
    external_ood = payloads["external_ood"]

    # H3 deliberately compares the pre-frozen physical-expert candidate with
    # its pre-registered controls.  It cannot retrospectively make the older
    # C3-versus-equal-capacity-C4 causality gate pass, so both checks remain.
    h3_clean = bool(
        h3.get("confirmation_passed") is True
        and h3.get("external_refit") is False
        and h3.get("hyperparameter_search") is False
        and h3.get("calibration_refit") is False
        and h3.get("threshold_refit") is False
        and h3.get("locked_test_used") is False
        and h3.get("test_v2_used") is False
        and h3.get("test_v3_used") is False
    )
    classic_baselines = all(
        _completed(payloads[name])
        and payloads[name].get("locked_test_used") is False
        and payloads[name].get("test_v3_used") is False
        for name in ("classic_losses", "dr_loss", "ral_loss")
    )
    latest_exact = bool(payloads["mllt_reproducibility"].get("equal_protocol_numerical_baseline_completed") is True)
    seal_intact = bool(
        seal.get("access_authorized") is False
        and seal.get("analysis_allowed") is False
        and seal.get("used") is False
    )

    requirements = {
        "c3_beats_same_capacity_c4_on_visibility_causal_gate": c3.get("promoted") is True,
        "core_14_direct_current_protocol_ablations": core14.get("all_14_direct_current_protocol") is True,
        "classic_and_recent_loss_baselines_same_protocol": classic_baselines,
        "latest_external_mllt_exact_equal_protocol_baseline": latest_exact,
        "full_frozen_candidate_fresh_identity_confirmation": h3_clean,
        "full_candidate_parameter_or_noise_ood_robustness": h5o.get("confirmation_passed") is True,
        "paired_statistics_for_promoted_final_candidate": payloads["statistics"].get("passed") is True and c3.get("promoted") is True,
        "engineering_cost_for_promoted_final_candidate": payloads["engineering"].get("passed") is True and c3.get("promoted") is True,
        "direct_three_stratum_mechanism_for_mft_vft_isrj": False,
        "all_development_promotion_gates_for_one_final_candidate": c3.get("promoted") is True,
        "final_bundle_complete": False,
        "test_v3_one_shot_completed": seal.get("used") is True,
    }
    candidate_ready = bool(all(requirements.values()))
    test_v3_authorization = bool(candidate_ready and seal_intact)
    return {
        "status": "completed",
        "experiment": "RouteA_GoalCompletionAudit_H3",
        "evidence_role": "read_only_goal_completion_audit",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "sources": source_receipts,
        "test_v3_seal": {
            "path": str(seal_path),
            "sha256": _sha256(seal_path),
            "access_authorized": seal.get("access_authorized"),
            "analysis_allowed": seal.get("analysis_allowed"),
            "used": seal.get("used"),
            "intact": seal_intact,
        },
        "requirement_checks": requirements,
        "completed_requirement_count": sum(requirements.values()),
        "total_requirement_count": len(requirements),
        "blocking_gaps": [name for name, passed in requirements.items() if not passed],
        "test_v3_authorization": test_v3_authorization,
        "interpretation": (
            "H3 is a clean, fresh-identity confirmation of a pre-frozen physical-expert candidate. "
            "It does not retroactively satisfy the failed C3-versus-C4 visibility-causality gate, "
            "does not provide complete-candidate OOD robustness: the preregistered H5o combined OOD confirmation failed, "
            "and it does not authorize test-v3."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/tapc/route_a_goal_completion_h3/audit.json",
    )
    args = parser.parse_args()
    payload = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "completed": payload["completed_requirement_count"],
        "total": payload["total_requirement_count"],
        "test_v3_authorization": payload["test_v3_authorization"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
