"""Freeze A/B development candidates and preregister the one-shot locked test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cj0724.route_b_mechanism_router import SEEDS, baseline_root, fusion_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, default=Path("outputs/route_b_mechanism_router_20260725"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ab_frozen_candidates_20260725"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    e5_manifest = Path("outputs/vft_physics_e5_frozen_candidate_20260725/bundle_manifest.json")
    e6_result = Path("outputs/e6_fft_audit_rfft_20260725/result.json")
    route_result = args.route_root / "result.json"
    route_audit = Path("outputs/route_b_candidate_audit_20260725/result.json")
    b1_result = Path("outputs/b1_physics_sufficiency_five_seed_20260725/result.json")
    a1_result = Path("outputs/a1_vft_mechanism_controls_rerun_20260725/result.json")
    b0_result = Path("outputs/b0_mft_isrj_error_atlas_rerun_20260725/result.json")
    manifest = {
        "status": "development_candidates_frozen_waiting_for_explicit_locked_test_authorization",
        "frozen_at_date": "2026-07-25",
        "candidates": {
            "E0": {
                "definition": "EfficientNet-B1 1D plus BCE, original six-label thresholds",
                "seed_roots": [str(baseline_root(seed).resolve()) for seed in SEEDS],
            },
            "E5": {
                "definition": "E0 plus frozen VFT matched-filter specialist",
                "manifest": member(e5_manifest), "permanent_read_only": True,
            },
            "A_final": {
                "definition": "alias of frozen E5; E6 rejected before five-seed confirmation",
                "reason": "FFT metrics were noninferior but throughput gate >=2x failed",
                "e6_rejection_evidence": member(e6_result),
                "same_statistical_candidate_as_E5": True,
            },
            "B_final": {
                "definition": "E5 plus fixed class-index routes to MFT combined-feature and ISRJ periodicity experts",
                "result": member(route_result), "audit": member(route_audit),
                "members": [],
            },
        },
        "necessary_ablations": {
            "E5_plus_MFT": "replace MFT output only",
            "E5_plus_ISRJ": "replace ISRJ output only",
            "B_final": "replace both MFT and ISRJ outputs",
            "source": member(route_audit),
        },
        "evidence": {
            "A1_mechanism_controls": member(a1_result),
            "B0_error_atlas": member(b0_result),
            "B1_five_seed_sufficiency": member(b1_result),
        },
        "locked_test_protocol": {
            "authorized": False, "used": False,
            "rule": "after explicit user authorization, read locked test exactly once for all preregistered candidates in one batch",
            "no_post_test_changes": ["models", "features", "routing", "thresholds", "calibration", "candidate_selection"],
            "primary_comparison": "B_final versus E5",
            "report": [
                "macro_AUPR", "macro_F1", "Hamming_loss", "subset_accuracy",
                "per_label_AUPR", "per_label_FPR", "weak_component_recall",
                "paired identity bootstrap confidence intervals", "inference throughput and artifact size",
            ],
            "multiplicity_note": "A_final is an alias of E5 and is not counted as an independent hypothesis",
        },
        "locked_test_used": False,
    }
    for seed in SEEDS:
        route_dir = args.route_root / f"seed_{seed}"
        manifest["candidates"]["B_final"]["members"].append({
            "seed": seed,
            "e5_protocol": member(fusion_root(seed) / "protocol.json"),
            "mft_classifier": member(route_dir / "mft_physics_classifier.joblib"),
            "isrj_classifier": member(route_dir / "isrj_physics_classifier.joblib"),
            "route_result": member(route_dir / "result.json"),
            "development_predictions": member(route_dir / "development_predictions.npz"),
        })
    manifest_path = args.output_root / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_root / "bundle_manifest.sha256").write_text(sha256(manifest_path) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "sha256": sha256(manifest_path),
                      "locked_test_used": False}, indent=2))


if __name__ == "__main__":
    main()
