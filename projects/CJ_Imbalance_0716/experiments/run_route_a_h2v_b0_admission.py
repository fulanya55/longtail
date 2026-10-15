"""H2v admission only: evaluate whether a fresh source has usable B0 error mass.

No candidate correction or threshold is trained here.  A failure closes H2v
before a conditional-fusion model can be considered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run_route_a_h1u_mft_delay_aperture_learning import (
    MODEL_SEEDS, SPLIT_SEEDS, H1uModel, _loader, _sha256, _train_arm,
    _verify_boundary, load_h1u_source, make_roles,
)

SOURCE = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h2v_mft_visibility_ladder_r1\closedset_6class_train.mat")


@torch.no_grad()
def fixed_lfm_score(iq: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """The untrained correct-LFM mean log-peak statistic used only for admission."""
    model, chunks = H1uModel("M1").to(device).eval(), []
    for start in range(0, len(iq), batch_size):
        feature = model.residual_features(torch.from_numpy(iq[start:start + batch_size]).to(device))
        chunks.append(feature.mean(dim=1).cpu().numpy())
    return np.concatenate(chunks)


def inversion_diagnostic(logits: np.ndarray, labels: np.ndarray, weak: np.ndarray, physics: np.ndarray) -> dict[str, object]:
    absent, weak_positive = labels[:, 3] == 0, weak.astype(bool)
    b0 = logits[:, 3]
    inversion = b0[weak_positive, None] <= b0[None, absent]
    correct = physics[weak_positive, None] > physics[None, absent]
    return {"weak_vs_absent_pair_count": int(inversion.size), "b0_inverted_pair_count": int(inversion.sum()), "b0_pair_inversion_rate": float(inversion.mean()), "physics_correct_order_within_b0_inversions": float(correct[inversion].mean()) if inversion.any() else None}


def smoke(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    score = fixed_lfm_score(np.random.default_rng(7).normal(size=(4, 2, 10_000)).astype(np.float32), device, 2)
    result = {"status": "completed_synthetic_smoke", "formal_h2v_source_opened": False, "fixed_lfm_score_shape": list(score.shape), "score_finite": bool(np.isfinite(score).all()), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.smoke:
        return smoke(args)
    audit = _verify_boundary(args.source_audit, args.test_v3_manifest)
    data, device, rows = load_h1u_source(args.source), torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")), []
    for split_seed, model_seed in zip(SPLIT_SEEDS, MODEL_SEEDS):
        roles = make_roles(data["combination"], data["labels"], data["weak"], split_seed)
        split_root = args.output_dir / f"split{split_seed}_model{model_seed}"; split_root.mkdir(parents=True, exist_ok=True)
        role_path = split_root / "roles.npz"; np.savez_compressed(role_path, fit=roles["fit"], validation=roles["validation"], early=roles["early"], receiver_noise_seed=data["receiver"])
        row = _train_arm(data, roles, "B0", split_seed, model_seed, args, args.output_dir, device)
        stored = np.load(split_root / "B0" / "early_predictions.npz")
        score = fixed_lfm_score(data["iq"][roles["early"]], device, args.batch_size)
        row["roles_sha256"] = _sha256(role_path); row["role_counts"] = {key: int(len(roles[key])) for key in ("fit", "validation", "early")}
        row["fixed_correct_lfm_error_set_diagnostic"] = inversion_diagnostic(stored["logits"], stored["labels"], stored["mft_weak_mask"], score)
        rows.append(row)
    weak_auroc = np.asarray([row["early"]["weak_mft"]["auroc"] for row in rows], dtype=float)
    inversion = np.asarray([row["fixed_correct_lfm_error_set_diagnostic"]["b0_pair_inversion_rate"] for row in rows], dtype=float)
    total_inverted = int(sum(row["fixed_correct_lfm_error_set_diagnostic"]["b0_inverted_pair_count"] for row in rows))
    weighted_correct = float(sum(row["fixed_correct_lfm_error_set_diagnostic"]["b0_inverted_pair_count"] * row["fixed_correct_lfm_error_set_diagnostic"]["physics_correct_order_within_b0_inversions"] for row in rows) / total_inverted) if total_inverted else None
    checks = {"mean_weak_mft_auroc_in_0p65_to_0p93": .65 <= float(weak_auroc.mean()) <= .93, "every_seed_b0_pair_inversion_rate_ge_0p05": bool(np.all(inversion >= .05)), "fixed_lfm_correct_order_within_union_b0_inversions_ge_0p60": weighted_correct is not None and weighted_correct >= .60, "source_audit_passed": True, "test_v3_sealed": True}
    receipt = {"source": str(args.source), "source_sha256": audit["source_sha256"], "source_audit_sha256": _sha256(args.source_audit), "split_seeds": list(SPLIT_SEEDS), "model_seeds": list(MODEL_SEEDS), "arm": "B0_only", "candidate_correction_trained": False, "selection": "validation_mft_weak_vs_negative_auroc", "calibration_used": False, "threshold_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    result = {"status": "completed_fresh_source_b0_admission", "experiment": "RouteA_H2v_ConditionalComplementarityAdmission", "evidence_role": "fresh_external_admission_not_candidate_effectiveness", "rows": rows, "aggregate": {"mean_early_weak_mft_auroc": float(weak_auroc.mean()), "per_seed_early_weak_mft_auroc": weak_auroc.tolist(), "per_seed_b0_pair_inversion_rate": inversion.tolist(), "union_b0_inverted_pair_count": total_inverted, "fixed_lfm_correct_order_within_union_b0_inversions": weighted_correct}, "admission_checks": checks, "admitted_to_separate_correction_preregistration": bool(all(checks.values())), "candidate_correction_trained": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, default=SOURCE); parser.add_argument("--source-audit", type=Path, default=ROOT / "outputs/tapc/route_a_h2v_visibility_ladder_generation/source_audit.json"); parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/tapc/route_a_h2v_b0_admission"); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--max-epochs", type=int, default=24); parser.add_argument("--patience", type=int, default=6); parser.add_argument("--device", default=None); parser.add_argument("--smoke", action="store_true"); print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
