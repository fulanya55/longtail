"""Copy-consistency augmentation for the 310D ISRJ periodicity expert."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import (
    _block_permutation,
    _block_shuffle,
    _extract,
    _fit_model,
    _joint_periodicity_features,
    _scores,
)
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import (
    _load_roles,
    _matched_view,
    _rank_metrics,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config
from experiments.run_shared import _data_path, load_config

ISRJ = 5
ROLES = ("fit", "validation", "early")


def _normalized_correlation(x, y, mask):
    m = mask.to(x.real.dtype)
    numerator = (x.conj() * y * m).sum(dim=1)
    energy_x = (x.abs().square() * m).sum(dim=1)
    energy_y = (y.abs().square() * m).sum(dim=1)
    return numerator / torch.sqrt(energy_x * energy_y).clamp_min(1e-8)


def _copy_consistency_features(signal, window_count=8):
    if signal.ndim != 3 or signal.shape[1:] != (2, 3000):
        raise ValueError(f"signal must have shape [B,2,3000], got {tuple(signal.shape)}")
    z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
    z = z / torch.sqrt(z.abs().square().mean(dim=1, keepdim=True)).clamp_min(1e-8)
    batch, width = z.shape
    envelope = z.abs() - z.abs().mean(dim=1, keepdim=True)
    fft = torch.fft.fft(envelope, n=8192, dim=1)
    acf = torch.fft.ifft(fft.abs().square(), dim=1).real[:, 40:1201]
    lag = acf.argmax(dim=1).long() + 40
    time = torch.arange(width, device=z.device).view(1, -1).expand(batch, -1)
    shifted_index = time + lag[:, None]
    valid = shifted_index < width
    shifted = z.gather(1, shifted_index.clamp_max(width - 1))
    global_corr = _normalized_correlation(z, shifted, valid)
    overlap = (width - lag).clamp_min(1)
    bin_index = torch.div(time * int(window_count), overlap[:, None], rounding_mode="floor")
    complex_windows = []
    envelope_windows = []
    residual_windows = []
    global_phase = global_corr / global_corr.abs().clamp_min(1e-8)
    residual = (shifted - z * global_phase[:, None]).abs().square()
    shifted_envelope = shifted.abs()
    base_envelope = z.abs()
    for index in range(int(window_count)):
        mask = valid & (bin_index == index)
        complex_windows.append(_normalized_correlation(z, shifted, mask))
        m = mask.to(z.real.dtype)
        a = base_envelope * m
        b = shifted_envelope * m
        a = a - a.sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        b = b - b.sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        env_num = (a * b * m).sum(dim=1)
        env_den = torch.sqrt((a.square() * m).sum(dim=1) * (b.square() * m).sum(dim=1)).clamp_min(1e-8)
        envelope_windows.append(env_num / env_den)
        residual_windows.append((residual * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0))
    cw = torch.stack(complex_windows, dim=1)
    cm = cw.abs()
    unit = cw / cm.clamp_min(1e-8)
    phase_coherence = unit.mean(dim=1).abs()
    adjacent_coherence = (unit[:, 1:] * unit[:, :-1].conj()).mean(dim=1).abs()
    ew = torch.stack(envelope_windows, dim=1)
    rw = torch.stack(residual_windows, dim=1)

    harmonic = []
    for multiplier in (2, 3):
        harmonic_lag = lag * multiplier
        harmonic_index = time + harmonic_lag[:, None]
        harmonic_valid = harmonic_index < width
        harmonic_shifted = z.gather(1, harmonic_index.clamp_max(width - 1))
        harmonic.append(_normalized_correlation(z, harmonic_shifted, harmonic_valid).abs())
    h2, h3 = harmonic
    features = torch.stack(
        [
            lag.float() / float(width),
            global_corr.abs(),
            cm.mean(dim=1), cm.std(dim=1), cm.amin(dim=1), cm.amax(dim=1),
            phase_coherence, adjacent_coherence,
            ew.mean(dim=1), ew.std(dim=1), ew.amin(dim=1), ew.amax(dim=1),
            rw.mean(dim=1), rw.std(dim=1), rw.amin(dim=1), rw.amax(dim=1),
            h2, h3,
            h2 / global_corr.abs().clamp_min(1e-8),
            h3 / global_corr.abs().clamp_min(1e-8),
        ],
        dim=1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0).float()


def _extract_augmented(signals, device, batch_size):
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            rows.append(torch.cat((_joint_periodicity_features(batch), _copy_consistency_features(batch)), dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def _run_seed(args, config, weak, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    permutation, permutation_sha = _block_permutation(120, 20301000 + seed)
    aligned = {role: _extract_augmented(roles[role]["signals"], device, args.feature_batch_size) for role in ROLES}
    shuffled = {role: _extract_augmented(_block_shuffle(roles[role]["signals"], permutation, args.block_size), device, args.feature_batch_size) for role in ROLES}
    views = {}
    receipts = {}
    for role in ROLES:
        selected, target, receipt = _matched_view(roles[role]["targets6"], weak[roles[role]["sample_ids"]], ISRJ)
        views[role] = {"selected": selected, "target": target.astype(np.uint8)}
        receipts[role] = receipt
    results = {}
    for offset, (name, features) in enumerate((("A1_AlignedCopyConsistency", aligned), ("S1_ShuffledCopyConsistency", shuffled))):
        mean = features["fit"].mean(axis=0).astype(np.float32)
        std = np.maximum(features["fit"].std(axis=0), 1e-5).astype(np.float32)
        normalized = {role: ((features[role] - mean[None, :]) / std[None, :]).astype(np.float32) for role in ROLES}
        fit_x = normalized["fit"][views["fit"]["selected"]]
        fit_y = views["fit"]["target"]
        model = _fit_model("H_AlignedHGB", fit_x, fit_y, 20344000 + seed + offset, False)
        metrics = {}
        prediction = {}
        for role in ("validation", "early"):
            score = _scores(model, normalized[role][views[role]["selected"]])
            metrics[role] = _rank_metrics(views[role]["target"], score)
            prediction[f"{role}_scores"] = score
            prediction[f"{role}_labels"] = views[role]["target"]
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
        np.savez_compressed(out / "predictions.npz", **prediction)
        results[name] = {"validation": metrics["validation"], "early": metrics["early"], "validation_early_auroc_gap": metrics["validation"]["auroc"] - metrics["early"]["auroc"], "feature_dim": int(fit_x.shape[1]), "finite": bool(np.isfinite(fit_x).all()), "model_sha256": _sha256(out / "model.joblib")}
    old = json.loads((args.periodicity_root / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))["models"]["H_AlignedHGB"]
    row = {"seed": seed, "status": "completed", "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path), "permutation_sha256": permutation_sha, "matched_counts": receipts, "A0_Aligned310D": old, "variants": results, "test_v3_used": False}
    _write_json(root / "seed_summary.json", row)
    return row


def run(args):
    config = load_config(args.config); _validate_config(config); data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]: raise ValueError("test-v3 is not sealed")
    labels = read_labels(data_path); values, _ = read_parameters(data_path); weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in args.seeds]
    a0_auc = np.asarray([r["A0_Aligned310D"]["early"]["auroc"] for r in rows]); a0_aupr = np.asarray([r["A0_Aligned310D"]["early"]["aupr"] for r in rows])
    a1_auc = np.asarray([r["variants"]["A1_AlignedCopyConsistency"]["early"]["auroc"] for r in rows]); a1_aupr = np.asarray([r["variants"]["A1_AlignedCopyConsistency"]["early"]["aupr"] for r in rows]); s1_auc = np.asarray([r["variants"]["S1_ShuffledCopyConsistency"]["early"]["auroc"] for r in rows]); gap = np.asarray([r["variants"]["A1_AlignedCopyConsistency"]["validation_early_auroc_gap"] for r in rows])
    comparisons = {"a1_vs_a0_early_auroc": _mean_ci((a1_auc-a0_auc).tolist(),20345000), "a1_vs_a0_early_aupr": _mean_ci((a1_aupr-a0_aupr).tolist(),20345001), "a1_vs_s1_early_auroc": _mean_ci((a1_auc-s1_auc).tolist(),20345002), "a1_validation_early_gap": _mean_ci(gap.tolist(),20345003)}
    checks = {"a1_auroc_gain_ge_0p010": comparisons["a1_vs_a0_early_auroc"]["mean"] >= .010, "a1_aupr_gain_nonnegative": comparisons["a1_vs_a0_early_aupr"]["mean"] >= 0, "a1_vs_s1_ge_0p020": comparisons["a1_vs_s1_early_auroc"]["mean"] >= .020, "four_of_five_positive_vs_a0": int(((a1_auc-a0_auc)>0).sum()) >= 4, "four_of_five_above_s1": int(((a1_auc-s1_auc)>0).sum()) >= 4, "validation_early_gap_lt_0p15": comparisons["a1_validation_early_gap"]["mean"] < .15, "finite_fixed_dimension": all(r["variants"][v]["finite"] and r["variants"][v]["feature_dim"] == 330 for r in rows for v in ("A1_AlignedCopyConsistency","S1_ShuffledCopyConsistency")), "same_hgb_configuration": True, "test_boundary_safe": all(not r["test_v3_used"] for r in rows)}; checks={k:bool(v) for k,v in checks.items()}
    result={"status":"completed","experiment":"RouteA_ISRJ_CopyConsistencyExpert","seeds":args.seeds,"data_sha256":data_sha,"rows":rows,"comparisons":comparisons,"component_gate":{"passed":bool(all(checks.values())),"checks":checks},"locked_test_used":False,"test_v3_used":False}; args.output_root.mkdir(parents=True,exist_ok=True); summary=args.output_root/"route_a_isrj_copy_consistency_expert_summary.json"; _write_json(summary,result); print(json.dumps({"status":"completed","gate":result["component_gate"],"comparisons":comparisons,"summary":str(summary)},ensure_ascii=False,indent=2)); return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml"); p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection"); p.add_argument("--periodicity-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3"); p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_copy_consistency_expert"); p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]); p.add_argument("--device",default=None); p.add_argument("--io-batch-size",type=int,default=256); p.add_argument("--feature-batch-size",type=int,default=64); p.add_argument("--block-size",type=int,default=25); p.add_argument("--weak-threshold-db",type=float,default=10.0); run(p.parse_args())

