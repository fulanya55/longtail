"""Create a hash manifest for the frozen H1+B2+MFT+VFT unified candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2039, 2040, 2041, 2042, 2043)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path)}


def run():
    seed_artifacts = {}
    for seed in SEEDS:
        unified = ROOT / f"outputs/tapc/route_a_h1_b2_mft_vft_unified_v3/seed_{seed}"
        h1 = ROOT / f"outputs/tapc/route_a_development_v3_h1_periodicity_increment/seed_{seed}"
        seed_artifacts[str(seed)] = {
            "focal_checkpoint": record(ROOT / f"outputs/tapc/route_a_development_v3_loss_convergence_audit/seed_{seed}/E2_Focal_36/best_model.pth"),
            "h1_head": record(ROOT / f"outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines/seed_{seed}/H1_Linear_BCE/best_model.pth"),
            "rawsaw": record(ROOT / f"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3/seed_{seed}/checkpoints/rawsaw.pth"),
            "periodicity_model": record(ROOT / f"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3/seed_{seed}/H_AlignedHGB/model.joblib"),
            "periodicity_normalization": record(ROOT / f"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3/seed_{seed}/H_AlignedHGB/normalization.npz"),
            "isrj_fusion_model": record(h1 / "head_aligned/model.joblib"),
            "isrj_fusion_normalization": record(h1 / "head_aligned/normalization.npz"),
            "mft_guard_model": record(ROOT / f"outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3/seed_{seed}/experts/aligned_model.joblib"),
            "mft_guard_normalization": record(ROOT / f"outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3/seed_{seed}/experts/normalization.npz"),
            "vft_ridge": record(ROOT / f"outputs/tapc/route_a_vft_ridge_stft_separability_v3/seed_{seed}/ridge_model.npz"),
            "vft_aligned_head": record(unified / "head_aligned/model.joblib"),
            "vft_aligned_normalization": record(unified / "head_aligned/normalization.npz"),
            "vft_base_head": record(unified / "head_base/model.joblib"),
            "vft_base_normalization": record(unified / "head_base/normalization.npz"),
            "vft_shuffled_head": record(unified / "head_shuffled/model.joblib"),
            "vft_shuffled_normalization": record(unified / "head_shuffled/normalization.npz"),
            "q0_calibration": record(unified / "Q0_H1/calibration.json"),
            "q0_thresholds": record(unified / "Q0_H1/thresholds.json"),
            "q1_calibration": record(unified / "Q1_H1B2/calibration.json"),
            "q1_thresholds": record(unified / "Q1_H1B2/thresholds.json"),
            "q2_calibration": record(unified / "Q2_H1B2_MFTGuard/calibration.json"),
            "q2_thresholds": record(unified / "Q2_H1B2_MFTGuard/thresholds.json"),
            "q3_calibration": record(unified / "Q3_VFTBaseOnlyCapacity/calibration.json"),
            "q3_thresholds": record(unified / "Q3_VFTBaseOnlyCapacity/thresholds.json"),
            "q4_calibration": record(unified / "Q4_VFTAlignedRidgeFusion/calibration.json"),
            "q4_thresholds": record(unified / "Q4_VFTAlignedRidgeFusion/thresholds.json"),
            "q5_calibration": record(unified / "Q5_VFTShuffledRidgeFusion/calibration.json"),
            "q5_thresholds": record(unified / "Q5_VFTShuffledRidgeFusion/thresholds.json"),
            "seed_summary": record(unified / "seed_summary.json"),
        }
    audit = json.loads((ROOT / "outputs/tapc/route_a_development_v11_generation/identity_audit.json").read_text(encoding="utf-8"))
    payload = {
        "status": "frozen",
        "candidate_name": "H1_B2_MFTPeakGuard_VFTRidgeFusion_v1",
        "frozen_at": "2026-08-01",
        "construction_summary": record(ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3/route_a_h1_b2_mft_vft_unified_v3_summary.json"),
        "source_code": {
            "construction": record(ROOT / "experiments/run_route_a_h1_b2_mft_vft_unified_v3.py"),
            "mft_feature": record(ROOT / "experiments/audit_route_a_mft_pulse_compression_morphology_v3.py"),
            "mft_guard": record(ROOT / "experiments/run_route_a_u2_mft_peak_count_negative_guard_v3.py"),
            "vft_feature": record(ROOT / "experiments/run_route_a_raw_iq_physics_separability_v3.py"),
        },
        "fixed_hyperparameters": {
            "weak_threshold_db": 10.0,
            "mft_residual_cap": 0.75,
            "vft_ridge_alpha": 10.0,
            "vft_fusion_C": 0.1,
            "vft_weak_positive_weight": 4.0,
            "threshold_max_negative_fpr": 0.20,
        },
        "development_v3_sha256": "b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47",
        "development_v11": audit,
        "seed_artifacts": seed_artifacts,
        "test_v3_manifest": record(ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"),
        "test_v3": {"access_authorized": False, "analysis_allowed": False, "used": False},
    }
    out = ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_unified_v1/frozen_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out), "sha256": sha(out), "seed_count": len(seed_artifacts)}, indent=2))
    return payload


if __name__ == "__main__":
    run()
