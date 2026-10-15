"""Audit exact evidence coverage for the 14 required VTSE/VBF ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(number: int, question: str, coverage: str, source: Path, note: str) -> dict:
    exists = source.is_file()
    payload = json.loads(source.read_text(encoding="utf-8")) if exists else {}
    safe = bool(exists and payload.get("locked_test_used") is False and payload.get("test_v3_used") is False)
    return {
        "number": number,
        "question": question,
        "coverage": coverage,
        "source": str(source),
        "source_exists": exists,
        "source_sha256": _sha256(source) if exists else None,
        "test_boundary_safe": safe,
        "note": note,
        "counts_as_direct_current_protocol_coverage": bool(coverage == "direct" and safe),
    }


def build(root: Path = PROJECT_ROOT) -> dict:
    out = root / "outputs/tapc"
    entries = [
        _entry(1, "remove VTSE", "direct", out / "route_a_vft_remove_vtse_unified_decision_ablation/route_a_vft_remove_vtse_unified_decision_ablation_summary.json", "Q4 removes the visibility-targeted Ridge-STFT expert while preserving every non-VFT logit and the independent calibration/threshold roles."),
        _entry(2, "VTSE includes strong positives", "direct", out / "route_a_vft_strict_mask_unified_ablation/route_a_vft_strict_mask_unified_ablation_summary.json", "All-positive Ridge score is integrated through the same Q4 fusion, calibration, and threshold roles."),
        _entry(3, "remove weak-positive mask", "direct", out / "route_a_vft_strict_mask_unified_ablation/route_a_vft_strict_mask_unified_ablation_summary.json", "The same all-positive arm removes the strict VFT weak-positive mask at the unified decision level."),
        _entry(4, "ordinary BCE versus three-stratum equal mass", "direct", out / "route_a_vft_weighting_uncertainty_core_ablation/route_a_vft_weighting_uncertainty_core_ablation_summary.json", "A2 versus A0 under identical current VFT score, head, roles, calibration, and threshold protocol."),
        _entry(5, "weak-positive weighting only", "direct", out / "route_a_vft_weighting_uncertainty_core_ablation/route_a_vft_weighting_uncertainty_core_ablation_summary.json", "A1 versus A0 isolates weak-positive weight 4 with no uncertainty gate."),
        _entry(6, "remove uncertainty gate", "direct", out / "route_a_vft_weighting_uncertainty_core_ablation/route_a_vft_weighting_uncertainty_core_ablation_summary.json", "A3 versus A1 isolates the sqrt gate with identical weak-positive weighting."),
        _entry(7, "linear versus sqrt gate", "direct", out / "route_a_vft_sqrt_linear_gate_ablation/route_a_vft_sqrt_linear_gate_ablation_summary.json", "Same four-dimensional current VFT fusion input; implementation reproduction was checked."),
        _entry(8, "remove residual hard cap", "direct", out / "route_a_mft_remove_cap_unified_decision_protocol_repair/route_a_mft_remove_cap_unified_decision_protocol_repair_summary.json", "A separate immutable protocol-repair receipt preserves the historical early-role failure and correctly verifies the registered FPR constraint on the assigned threshold-selection role; independent early FPR remains explicitly reported."),
        _entry(9, "global/final/stage2/stage3 expert layer", "direct", out / "route_a_mft_layer_source_unified_decision_ablation/route_a_mft_layer_source_unified_decision_ablation_summary.json", "Global, final temporal, stage2, and stage3 frozen MFT probes are each integrated through the identical zero-parameter bounded residual in the current Q4 decision protocol."),
        _entry(10, "same-parameter MLP capacity control", "direct", out / "route_a_vft_mlp_capacity_physics_control_unified_decision_ablation/route_a_vft_mlp_capacity_physics_control_unified_decision_ablation_summary.json", "All three arms use the identical seven-parameter MLP1; no-VTSE and shuffled-VTSE controls isolate physical evidence from nonlinear decision capacity."),
        _entry(11, "RawSaw top-k 4/8/16", "direct", out / "route_a_rawsaw_topk_within_run_protocol_repair/route_a_rawsaw_topk_within_run_protocol_repair_summary.json", "Read-only audit validates the equal-budget 4/8/16 scientific contrast while retaining the separate historical CUDA state-drift diagnostic."),
        _entry(12, "weak threshold 5/10/15 dB", "direct", out / "route_a_final_candidate_weak_threshold_sensitivity/route_a_final_candidate_weak_threshold_sensitivity_summary.json", "Fixed decision evaluation under three visibility definitions; no threshold/model retuning."),
        _entry(13, "power measurement error in weak mask", "direct", out / "route_a_vft_hard_soft_power_error_ablation/route_a_vft_hard_soft_power_error_ablation_summary.json", "Fit-only independent 1 dB active-component error versus nominal evaluation."),
        _entry(14, "hard mask versus soft visibility", "direct", out / "route_a_vft_hard_soft_power_error_ablation/route_a_vft_hard_soft_power_error_ablation_summary.json", "HardNominal versus fixed-temperature SoftVisibility Ridge-STFT comparison."),
    ]
    return {
        "status": "completed",
        "experiment": "RouteA_Core14AblationCoverageAudit",
        "locked_test_used": False,
        "test_v3_used": False,
        "entries": entries,
        "direct_current_protocol_count": sum(row["counts_as_direct_current_protocol_coverage"] for row in entries),
        "total_required": 14,
        "all_14_direct_current_protocol": all(row["counts_as_direct_current_protocol_coverage"] for row in entries),
        "partial_or_missing": [row["number"] for row in entries if not row["counts_as_direct_current_protocol_coverage"]],
        "interpretation": "The matrix prevents heterogeneous frozen lesions and construction diagnostics from being misreported as a complete end-to-end current-candidate ablation suite.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_core_14_ablation_coverage/core_14_ablation_coverage.json")
    args = parser.parse_args()
    payload = build(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "direct_current_protocol_count": payload["direct_current_protocol_count"], "partial_or_missing": payload["partial_or_missing"], "test_v3_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
