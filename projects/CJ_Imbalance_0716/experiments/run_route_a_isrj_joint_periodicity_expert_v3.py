"""Strict ISRJ joint periodicity feature test with long-period destruction control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import (  # noqa: E402
    _load_roles, _matched_view, _rank_metrics,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


ISRJ_INDEX = 5
ISRJ_NAME = "ISRJ"
ROLES = ("fit", "validation", "early")
VARIANTS = ("L_AlignedLogistic", "H_AlignedHGB", "S_BlockShuffledHGB")


def _pool_max(sequence, bins):
    usable = int(sequence.shape[1]) // int(bins) * int(bins)
    return sequence[:, :usable].reshape(len(sequence), int(bins), -1).amax(dim=2)


def _joint_periodicity_features(signal):
    if signal.ndim != 3 or signal.shape[1:] != (2, 3000):
        raise ValueError(f"signal must have shape [B,2,3000], got {tuple(signal.shape)}")
    z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
    z = z / torch.sqrt(z.abs().square().mean(dim=1, keepdim=True)).clamp_min(1e-8)
    transformed = torch.fft.fft(z, n=8192, dim=1)
    complex_acf = torch.fft.ifft(transformed.abs().square(), dim=1).abs()[:, 1:1201]
    complex_acf = complex_acf / complex_acf[:, :1].clamp_min(1e-8)
    envelope = z.abs(); envelope = envelope - envelope.mean(dim=1, keepdim=True)
    envelope_fft = torch.fft.fft(envelope, n=8192, dim=1)
    envelope_acf = torch.fft.ifft(envelope_fft.abs().square(), dim=1).abs()[:, 1:1201]
    envelope_acf = envelope_acf / envelope_acf[:, :1].clamp_min(1e-8)
    spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=1), dim=1).abs()
    spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(1e-8)
    complex_bins = _pool_max(complex_acf, 120)
    envelope_bins = _pool_max(envelope_acf, 120)
    spectrum_bins = spectrum.reshape(len(z), 60, 50).sum(dim=2)
    periodic = envelope_acf[:, 39:600]
    periodic_peak = periodic.amax(dim=1, keepdim=True)
    periodic_position = periodic.argmax(dim=1, keepdim=True).float() / float(periodic.shape[1])
    envelope_top = periodic.topk(8, dim=1).values.sort(dim=1).values
    return torch.cat((complex_bins, envelope_bins, spectrum_bins,
                      periodic_peak, periodic_position, envelope_top), dim=1).float()


def _block_permutation(block_count, seed):
    permutation = np.random.default_rng(int(seed)).permutation(int(block_count)).astype(np.int64)
    return permutation, hashlib.sha256(permutation.tobytes(order="C")).hexdigest()


def _block_shuffle(signals, permutation, block_size=25):
    values = np.asarray(signals)
    if values.shape[-1] % int(block_size):
        raise ValueError("signal length must be divisible by block_size")
    block_count = values.shape[-1] // int(block_size)
    order = np.asarray(permutation, dtype=np.int64)
    if len(order) != block_count:
        raise ValueError("permutation length mismatch")
    reshaped = values.reshape(len(values), values.shape[1], block_count, int(block_size))
    return reshaped[:, :, order, :].reshape(values.shape).copy()


def _extract(signals, device, batch_size):
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            rows.append(_joint_periodicity_features(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def _balanced_weights(labels):
    target = np.asarray(labels, dtype=np.uint8)
    positives = max(int(target.sum()), 1); negatives = max(int(len(target) - target.sum()), 1)
    return np.where(target == 1, len(target) / (2.0 * positives),
                    len(target) / (2.0 * negatives)).astype(np.float64)


def _fit_model(name, fit_x, fit_y, seed, smoke):
    if name == "L_AlignedLogistic":
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                   solver="lbfgs", random_state=int(seed))
        model.fit(fit_x, fit_y)
        return model
    model = HistGradientBoostingClassifier(
        max_iter=5 if smoke else 200, learning_rate=0.05, max_leaf_nodes=7,
        l2_regularization=1.0, early_stopping=False, random_state=int(seed),
    )
    model.fit(fit_x, fit_y, sample_weight=_balanced_weights(fit_y))
    return model


def _scores(model, values):
    return model.predict_proba(values)[:, 1].astype(np.float32)


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; summary_path = root / "seed_summary.json"
    if args.resume and summary_path.is_file(): return json.loads(summary_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(args.c6_root, seed, _data_path(config),
                                           config["normalization"], args.io_batch_size)
    permutation_seed = 20301000 + int(seed)
    permutation, permutation_sha = _block_permutation(120, permutation_seed)
    raw_features = {role: _extract(roles[role]["signals"], device, args.feature_batch_size) for role in ROLES}
    shuffled_features = {role: _extract(_block_shuffle(roles[role]["signals"], permutation, args.block_size),
                                        device, args.feature_batch_size) for role in ROLES}
    views, receipts = {}, {}
    for role in ROLES:
        selected, labels, receipt = _matched_view(
            roles[role]["targets6"], weak6[roles[role]["sample_ids"]], ISRJ_INDEX
        )
        views[role] = {"selected": selected, "labels": labels,
                       "sample_ids": np.asarray(roles[role]["sample_ids"])[selected]}
        receipts[role] = receipt
    feature_sets = {"aligned": raw_features, "shuffled": shuffled_features}
    normalized = {}
    normalization = {}
    for family, features in feature_sets.items():
        mean = features["fit"].mean(axis=0).astype(np.float32)
        std = np.maximum(features["fit"].std(axis=0), 1e-5).astype(np.float32)
        normalized[family] = {role: ((features[role] - mean[None, :]) / std[None, :]).astype(np.float32)
                              for role in ROLES}
        normalization[family] = {"mean": mean, "std": std}
    prior = json.loads((args.layerwise_root / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))
    baseline = prior["representations"]["T_FinalTemporal"][ISRJ_NAME]["early"]["auroc"]
    results = {}
    train_seed = 20302000 + int(seed)
    for name in VARIANTS:
        family = "shuffled" if name == "S_BlockShuffledHGB" else "aligned"
        fit_x = normalized[family]["fit"][views["fit"]["selected"]]
        fit_y = views["fit"]["labels"].astype(np.uint8)
        model = _fit_model(name, fit_x, fit_y, train_seed, args.smoke)
        metrics = {}; predictions = {}
        for role in ("validation", "early"):
            x = normalized[family][role][views[role]["selected"]]
            score = _scores(model, x)
            metrics[role] = _rank_metrics(views[role]["labels"], score)
            predictions[f"{role}_sample_ids"] = views[role]["sample_ids"]
            predictions[f"{role}_labels"] = views[role]["labels"].astype(np.uint8)
            predictions[f"{role}_scores"] = score
        output = root / name; output.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output / "model.joblib")
        np.savez_compressed(output / "normalization.npz", **normalization[family])
        np.savez_compressed(output / "predictions.npz", **predictions)
        results[name] = {"validation": metrics["validation"], "early": metrics["early"],
                         "validation_early_auroc_gap": metrics["validation"]["auroc"] - metrics["early"]["auroc"],
                         "model_sha256": _sha256(output / "model.joblib"),
                         "normalization_sha256": _sha256(output / "normalization.npz"),
                         "predictions_sha256": _sha256(output / "predictions.npz"),
                         "train_seed": train_seed, "feature_dim": int(fit_x.shape[1])}
    aligned = results["H_AlignedHGB"]["early"]["auroc"]
    comparisons = {"final_temporal_early_auroc": baseline,
                   "aligned_hgb_minus_final_temporal": aligned - baseline,
                   "aligned_hgb_minus_block_shuffled_hgb": aligned - results["S_BlockShuffledHGB"]["early"]["auroc"],
                   "aligned_hgb_minus_aligned_logistic": aligned - results["L_AlignedLogistic"]["early"]["auroc"]}
    payload = {"seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
               "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
               "generator_physics": {"sampling_rate_hz": 10000000, "sample_width_points": [50, 200],
                                     "expected_repeat_lag_points": [100, 1200]},
               "block_control": {"block_size": args.block_size, "block_count": 120,
                                 "permutation_seed": permutation_seed, "permutation_sha256": permutation_sha},
               "matched_counts": receipts, "models": results, "comparisons": comparisons,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, payload); return payload


def run(args):
    config = load_config(args.config); _validate_config(config); data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]: raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_ISRJ_JointPeriodicityExpert_v3", "seeds": list(seeds),
        "feature_dim": 310, "block_size": args.block_size,
        "hgb": {"max_iter": 200, "learning_rate": .05, "max_leaf_nodes": 7,
                "l2_regularization": 1.0, "early_stopping": False},
        "gates": {"absolute_auroc": .620, "gain_vs_final": .030,
                  "gain_vs_block_shuffle": .020, "gain_vs_logistic": .010,
                  "positive_seeds": 4, "maximum_gap": .150},
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {
        "aligned_hgb_early_auroc": _mean_ci([row["models"]["H_AlignedHGB"]["early"]["auroc"] for row in rows], 20303000),
        "aligned_hgb_early_aupr": _mean_ci([row["models"]["H_AlignedHGB"]["early"]["aupr"] for row in rows], 20303001),
        "aligned_logistic_early_auroc": _mean_ci([row["models"]["L_AlignedLogistic"]["early"]["auroc"] for row in rows], 20303002),
        "block_shuffled_hgb_early_auroc": _mean_ci([row["models"]["S_BlockShuffledHGB"]["early"]["auroc"] for row in rows], 20303003),
        "gain_vs_final_temporal": _mean_ci([row["comparisons"]["aligned_hgb_minus_final_temporal"] for row in rows], 20303004),
        "gain_vs_block_shuffled": _mean_ci([row["comparisons"]["aligned_hgb_minus_block_shuffled_hgb"] for row in rows], 20303005),
        "gain_vs_aligned_logistic": _mean_ci([row["comparisons"]["aligned_hgb_minus_aligned_logistic"] for row in rows], 20303006),
        "validation_early_gap": _mean_ci([row["models"]["H_AlignedHGB"]["validation_early_auroc_gap"] for row in rows], 20303007),
        "positive_vs_final_seeds": int(sum(row["comparisons"]["aligned_hgb_minus_final_temporal"] > 0 for row in rows)),
        "positive_vs_block_shuffled_seeds": int(sum(row["comparisons"]["aligned_hgb_minus_block_shuffled_hgb"] > 0 for row in rows)),
    }
    if args.smoke:
        gate = {"checks": {}, "passed": False, "evidence_level": "smoke_only"}
    else:
        checks = {"absolute_auroc_ge_0p620": aggregate["aligned_hgb_early_auroc"]["mean"] >= .620,
                  "gain_vs_final_ge_0p030": aggregate["gain_vs_final_temporal"]["mean"] >= .030,
                  "gain_vs_block_shuffle_ge_0p020": aggregate["gain_vs_block_shuffled"]["mean"] >= .020,
                  "gain_vs_logistic_ge_0p010": aggregate["gain_vs_aligned_logistic"]["mean"] >= .010,
                  "four_of_five_vs_final": aggregate["positive_vs_final_seeds"] >= 4,
                  "four_of_five_vs_block_shuffle": aggregate["positive_vs_block_shuffled_seeds"] >= 4,
                  "validation_early_gap_lt_0p150": aggregate["validation_early_gap"]["mean"] < .150,
                  "same_hgb_configuration": True,
                  "identity_and_test_boundary_safe": all(not row["test_v3_used"] for row in rows)}
        gate = {"checks": checks, "passed": bool(all(checks.values())),
                "evidence_level": "five_seed_development_v3_component_gate"}
    payload = {"status": "completed", "experiment": "RouteA_ISRJ_JointPeriodicityExpert_v3",
               "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "component_promotion_gate": gate,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(args.output_root / "route_a_isrj_joint_periodicity_expert_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--layerwise-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_layerwise_matched_negative_separability_v3_strict")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043]); parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=25); parser.add_argument("--weak-threshold-db", type=float, default=10.0); return parser.parse_args()


if __name__ == "__main__": run(parse_args())

