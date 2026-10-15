"""Post-hoc, non-selective diagnosis for the preregistered failed H1u study."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run_route_a_h1u_mft_delay_aperture_learning import (
    ARMS, MODEL_SEEDS, SPLIT_SEEDS, H1uModel, _loader, _sha256, load_h1u_source,
)


def _summary(value: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(value)), "median": float(np.median(value)), "p05": float(np.quantile(value, .05)), "p95": float(np.quantile(value, .95)), "saturation_abs_ge_0p49_rate": float(np.mean(np.abs(value) >= .49))}


def _discrimination(residual: np.ndarray, target: np.ndarray, weak: np.ndarray) -> dict[str, dict[str, float]]:
    absent, strong = ~target, target & ~weak
    def binary(positive: np.ndarray) -> dict[str, float]:
        selected, y = absent | positive, positive[absent | positive].astype(np.uint8)
        return {"auroc": float(roc_auc_score(y, residual[selected])), "aupr": float(average_precision_score(y, residual[selected]))}
    return {"weak_vs_absent": binary(weak), "strong_vs_absent": binary(strong)}


@torch.no_grad()
def _residual_by_state(model: H1uModel, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); values, targets, weak = [], [], []
    for iq, labels, weak_mask in loader:
        iq = iq.to(device); feature = model.residual_features(iq)
        values.append((.5 * torch.tanh(model.mft_residual(feature).squeeze(1))).cpu().numpy())
        targets.append(labels[:, 3].numpy()); weak.append(weak_mask.numpy())
    return np.concatenate(values), np.concatenate(targets).astype(bool), np.concatenate(weak).astype(bool)


def run(args: argparse.Namespace) -> dict[str, object]:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("passed"):
        raise ValueError("this diagnostic is only for the failed preregistered H1u study")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    data, device, rows, complementarity = load_h1u_source(args.source), torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")), [], []
    for split_seed, model_seed in zip(SPLIT_SEEDS, MODEL_SEEDS):
        role_path = args.output_dir / f"split{split_seed}_model{model_seed}" / "roles.npz"
        early = np.load(role_path)["early"]
        loader = _loader(data, early, args.batch_size)
        m1_residual = m1_target = m1_weak = None
        for arm in ARMS:
            path = args.output_dir / f"split{split_seed}_model{model_seed}" / arm / "best.pt"
            model = H1uModel(arm).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
            residual, target, weak = _residual_by_state(model, loader, device)
            states = {"absent": ~target, "weak": weak, "strong": target & ~weak}
            rows.append({"split_seed": split_seed, "model_seed": model_seed, "arm": arm, "checkpoint_sha256": _sha256(path), "mft_residual": {name: _summary(residual[mask]) for name, mask in states.items()}, "residual_discrimination": _discrimination(residual, target, weak), "residual_head_weight": model.mft_residual.weight.detach().cpu().numpy().reshape(-1).astype(float).tolist(), "residual_head_bias": float(model.mft_residual.bias.detach().cpu().item())})
            if arm == "M1":
                m1_residual, m1_target, m1_weak = residual, target, weak
        b0_logits = np.load(args.output_dir / f"split{split_seed}_model{model_seed}" / "B0" / "early_predictions.npz")["logits"][:, 3]
        absent, weak_positive = ~m1_target, m1_weak
        b0_inversion = b0_logits[weak_positive, None] <= b0_logits[None, absent]
        m1_corrects = m1_residual[weak_positive, None] > m1_residual[None, absent]
        complementarity.append({"split_seed": split_seed, "b0_weak_vs_absent_pair_inversion_rate": float(b0_inversion.mean()), "b0_inverted_pair_count": int(b0_inversion.sum()), "all_pair_count": int(b0_inversion.size), "m1_residual_correct_order_rate_within_b0_inversions": float(m1_corrects[b0_inversion].mean()) if b0_inversion.any() else None})
    key_rows = [row for row in rows if row["arm"] == "M1"]
    unstable = [row["split_seed"] for row in key_rows if row["mft_residual"]["weak"]["saturation_abs_ge_0p49_rate"] > .25]
    result = {"status": "completed_posthoc_failure_diagnostic", "experiment": "RouteA_H1u_MFTDelayApertureLearning", "evidence_role": "posthoc_development_failure_diagnostic_not_model_selection", "pre_registered_study_passed": False, "rows": rows, "b0_error_set_complementarity": complementarity, "interpretation_limits": ["The H0u zero-fit physical statistic proved observable signal information but did not imply that a three-number residual should improve a learned six-label classifier.", "This diagnostic does not alter the H1u conclusion, select an arm, or authorize any H1u retry.", "No threshold, calibration, test-v2, locked historical test, or test-v3 was used."], "m1_weak_residual_saturated_seed_count": len(unstable), "m1_weak_residual_saturated_seeds": unstable, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h1u_mft_delay_aperture_r1\closedset_6class_train.mat"))
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/tapc/route_a_h1u_mft_delay_aperture_learning/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/tapc/route_a_h1u_mft_delay_aperture_learning")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h1u_mft_delay_aperture_learning/failure_diagnostic.json")
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device", default=None)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
