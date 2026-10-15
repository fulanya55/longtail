"""Fixed raw-I/Q VFT-versus-ISRJ source-separability diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _load_roles, _rank_metrics  # noqa: E402
from experiments.run_route_a_raw_iq_physics_separability_v3 import _vft_features  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VFT_INDEX = 4
ISRJ_INDEX = 5
ROLES = ("fit", "validation", "early")
ARMS = ("RidgeSTFT", "RawSource", "PhaseScrambledControl")
EXTERNAL_SHA256 = "a7e318ba3eeaedb4fea67c894238e4674b133059bc405342ea40c9dfb767e081"


def _phase_scramble(signal: torch.Tensor) -> torch.Tensor:
    """Preserve amplitude but destroy temporal phase organisation deterministically."""
    z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
    index = (torch.arange(z.shape[-1], device=z.device) * 7919 + 17) % z.shape[-1]
    scrambled = z.abs() * torch.exp(1j * torch.angle(z[:, index]))
    return torch.stack((scrambled.real, scrambled.imag), dim=1)


def _phase_copy_stats(z: torch.Tensor) -> torch.Tensor:
    values = []
    for lag in (32, 64, 128, 256):
        product = z[:, lag:] * z[:, :-lag].conj()
        unit = product / product.abs().clamp_min(1e-8)
        values.extend((unit.mean(dim=-1).abs(), torch.angle(product).std(dim=-1, unbiased=False)))
    return torch.stack(values, dim=1)


def _segment_copy_stats(z: torch.Tensor) -> torch.Tensor:
    length = (z.shape[-1] // 10) * 10
    parts = z[:, :length].reshape(z.shape[0], 10, length // 10)
    product = (parts[:, 1:] * parts[:, :-1].conj()).mean(dim=-1)
    denom = (parts[:, 1:].abs().square().mean(dim=-1) * parts[:, :-1].abs().square().mean(dim=-1)).sqrt().clamp_min(1e-8)
    correlation = product / denom
    amplitude = correlation.abs()
    angle = torch.angle(correlation)
    return torch.stack((amplitude.mean(dim=-1), amplitude.std(dim=-1, unbiased=False), amplitude.amax(dim=-1),
                        angle.std(dim=-1, unbiased=False)), dim=1)


def _ridge_reset_stats(z: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(128, device=z.device)
    spectrum = torch.stft(z, n_fft=128, hop_length=32, window=window, center=True, return_complex=True, onesided=False)
    power = torch.fft.fftshift(spectrum.abs().square(), dim=1)
    distribution = power / power.sum(dim=1, keepdim=True).clamp_min(1e-12)
    coordinate = torch.linspace(-1.0, 1.0, power.shape[1], device=z.device, dtype=power.dtype)
    centroid = (distribution * coordinate[None, :, None]).sum(dim=1)
    peak = coordinate[distribution.argmax(dim=1)]
    ridge = torch.stack((centroid, peak), dim=1)
    first = ridge[:, :, 1:] - ridge[:, :, :-1]
    second = first[:, :, 1:] - first[:, :, :-1]
    ridge_stats = torch.cat((
        first.abs().mean(dim=-1), first.std(dim=-1, unbiased=False), first.abs().amax(dim=-1),
        second.abs().mean(dim=-1), second.std(dim=-1, unbiased=False), second.abs().amax(dim=-1),
    ), dim=1)
    magnitude = z.abs()
    phase_step = torch.angle(z[:, 1:] * z[:, :-1].conj())
    amplitude_jump = (magnitude[:, 1:] - magnitude[:, :-1]).abs()
    phase_jump = (phase_step[:, 1:] - phase_step[:, :-1]).abs()
    reset_stats = []
    for values in (amplitude_jump, phase_jump):
        reset_stats.extend((values.mean(dim=-1), values.std(dim=-1, unbiased=False),
                            values.amax(dim=-1), torch.topk(values, k=16, dim=-1).values.mean(dim=-1)))
    return torch.cat((ridge_stats, torch.stack(reset_stats, dim=1)), dim=1)


def raw_source_features(signal: torch.Tensor) -> torch.Tensor:
    z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
    return torch.cat((_phase_copy_stats(z), _segment_copy_stats(z), _ridge_reset_stats(z)), dim=1)


def _extract_all(signals: np.ndarray, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    rows = {name: [] for name in ARMS}
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            rows["RidgeSTFT"].append(_vft_features(batch, grid_size=16).cpu().numpy().astype(np.float32))
            rows["RawSource"].append(raw_source_features(batch).cpu().numpy().astype(np.float32))
            rows["PhaseScrambledControl"].append(raw_source_features(_phase_scramble(batch)).cpu().numpy().astype(np.float32))
    return {name: np.concatenate(value, axis=0) for name, value in rows.items()}


def _hard_source_view(targets6: np.ndarray, weak6: np.ndarray):
    """Weak VFT positives versus VFT-absent, ISRJ-present negatives, context-matched."""
    targets = np.asarray(targets6, dtype=np.uint8)
    weak = np.asarray(weak6, dtype=bool)
    context_columns = [0, 1, 2, 3]
    positive = (targets[:, VFT_INDEX] == 1) & weak[:, VFT_INDEX]
    negative = (targets[:, VFT_INDEX] == 0) & (targets[:, ISRJ_INDEX] == 1)
    pos_context = {tuple(row) for row in targets[positive][:, context_columns].tolist()}
    neg_context = {tuple(row) for row in targets[negative][:, context_columns].tolist()}
    matched = pos_context & neg_context
    positive_keep = np.asarray([bool(positive[index]) and tuple(targets[index, context_columns].tolist()) in matched for index in range(len(targets))])
    negative_keep = np.asarray([bool(negative[index]) and tuple(targets[index, context_columns].tolist()) in matched for index in range(len(targets))])
    selected = positive_keep | negative_keep
    return selected, positive_keep[selected].astype(np.uint8), {
        "weak_vft_positive_count": int(positive_keep.sum()),
        "vft_absent_isrj_present_negative_count": int(negative_keep.sum()),
        "matched_context_count": int(len(matched)),
    }


def _fit_and_score(train_x, train_y, evaluate_x, alpha: float):
    positives = max(int(train_y.sum()), 1)
    negatives = max(int((train_y == 0).sum()), 1)
    weight = np.where(train_y == 1, 0.5 / positives, 0.5 / negatives).astype(np.float64)
    model = RidgeClassifier(alpha=float(alpha), fit_intercept=True)
    model.fit(train_x, train_y, sample_weight=weight)
    return model, model.decision_function(evaluate_x).astype(np.float32)


def _preflight_views(args, config, weak):
    """Verify source labels are estimable before touching expensive raw features."""
    rows = []
    for seed in args.seeds[:1] if args.smoke else args.seeds:
        roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
        role_counts = {}
        for role in ROLES:
            _selected, _labels, receipt = _hard_source_view(roles[role]["targets6"], weak[roles[role]["sample_ids"]])
            role_counts[role] = receipt
        rows.append({"seed": int(seed), "role_indices_sha256": _sha256(role_path), "roles": role_counts})
    checks = {
        "fit_has_both_matched_classes": all(row["roles"]["fit"]["weak_vft_positive_count"] > 0 and row["roles"]["fit"]["vft_absent_isrj_present_negative_count"] > 0 for row in rows),
        "validation_has_both_matched_classes": all(row["roles"]["validation"]["weak_vft_positive_count"] > 0 and row["roles"]["validation"]["vft_absent_isrj_present_negative_count"] > 0 for row in rows),
        "early_has_both_matched_classes": all(row["roles"]["early"]["weak_vft_positive_count"] > 0 and row["roles"]["early"]["vft_absent_isrj_present_negative_count"] > 0 for row in rows),
    }
    return rows, {"checks": checks, "passed": all(checks.values())}


def _normalise(features: dict[str, np.ndarray], arm: str):
    mean = features["fit"][arm].mean(axis=0).astype(np.float32)
    std = np.maximum(features["fit"][arm].std(axis=0), 1e-5).astype(np.float32)
    return {role: ((features[role][arm] - mean[None, :]) / std[None, :]).astype(np.float32) for role in features}, mean, std


def _run_seed(args, config, weak, seed, device, data_sha, external):
    root = args.output_root / f"seed_{seed}"
    summary_path = root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    features = {role: _extract_all(roles[role]["signals"], device, args.feature_batch_size) for role in ROLES}
    views, receipts = {}, {}
    for role in ROLES:
        selected, labels, receipt = _hard_source_view(roles[role]["targets6"], weak[roles[role]["sample_ids"]])
        views[role] = (selected, labels)
        receipts[role] = receipt
    external_selected, external_labels, external_receipt = _hard_source_view(external["labels"], external["weak"])
    result, models = {}, {}
    for arm in ARMS:
        normalized, mean, std = _normalise(features, arm)
        train_selected, train_labels = views["fit"]
        model, _ = _fit_and_score(normalized["fit"][train_selected], train_labels, normalized["fit"][train_selected], args.ridge_alpha)
        arm_metrics = {}
        for role in ("validation", "early"):
            selected, labels = views[role]
            score = model.decision_function(normalized[role][selected])
            arm_metrics[role] = _rank_metrics(labels, score)
        external_x = ((external["features"][arm] - mean[None, :]) / std[None, :]).astype(np.float32)
        arm_metrics["external_ood"] = _rank_metrics(external_labels, model.decision_function(external_x[external_selected]))
        result[arm] = {"feature_dim": int(mean.size), **arm_metrics}
        models[arm] = {"coef": model.coef_.astype(np.float32), "intercept": model.intercept_.astype(np.float32), "mean": mean, "std": std}
    deltas = {
        "raw_minus_ridge": {role: {metric: float(result["RawSource"][role][metric] - result["RidgeSTFT"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early", "external_ood")},
        "raw_minus_scrambled": {role: {metric: float(result["RawSource"][role][metric] - result["PhaseScrambledControl"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early", "external_ood")},
        "validation_early_raw_auroc_gap": float(result["RawSource"]["validation"]["auroc"] - result["RawSource"]["early"]["auroc"]),
    }
    np.savez_compressed(root / "models.npz", **{f"{arm}_{key}": value for arm, item in models.items() for key, value in item.items()})
    payload = {
        "status": "completed", "seed": int(seed), "data_sha256": data_sha, "external_data_sha256": EXTERNAL_SHA256,
        "role_indices_sha256": _sha256(role_path), "ridge_alpha": float(args.ridge_alpha), "weak_threshold_db": float(args.weak_threshold_db),
        "results": result, "deltas": deltas, "matched_counts": receipts, "external_matched_counts": external_receipt,
        "validation_used_for_selection": False, "external_refit": False, "threshold_selected": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 boundary is not sealed")
    labels = read_labels(data_path); values, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    preflight_rows, feasibility_gate = _preflight_views(args, config, weak)
    if not feasibility_gate["passed"]:
        payload = {
            "status": "completed", "experiment": "RouteA_VFTISRJRawSourceDiagnostic_v1", "smoke": bool(args.smoke),
            "seeds": list(args.seeds[:1] if args.smoke else args.seeds), "data_sha256": data_sha,
            "feasibility_rows": preflight_rows, "feasibility_gate": feasibility_gate,
            "promotion_gate": {"checks": {"source_view_estimable": False}, "passed": False},
            "stopped_before_feature_extraction": True,
            "stop_reason": "No composition-matched weak-VFT versus VFT-absent/ISRJ-present source view exists in fit/validation roles; relaxing it would introduce co-label confounding.",
            "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_root / "route_a_vft_isrj_raw_source_diagnostic_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    external_path = Path(args.external_data_path)
    if _sha256(external_path).lower() != EXTERNAL_SHA256:
        raise ValueError("external OOD SHA mismatch")
    external_labels = read_labels(external_path); external_values, _ = read_parameters(external_path)
    external_weak = compute_weak_component_mask(external_labels, _component_power(external_values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    external_signals = _load_signals(external_path, np.arange(len(external_labels)), args.io_batch_size)
    external = {"labels": external_labels, "weak": external_weak, "features": _extract_all(external_signals, device, args.feature_batch_size)}
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak, seed, device, data_sha, external) for seed in seeds]
    aggregate = {}
    for delta_name in ("raw_minus_ridge", "raw_minus_scrambled"):
        aggregate[delta_name] = {}
        for role in ("validation", "early", "external_ood"):
            aggregate[delta_name][role] = {metric: _mean_ci([row["deltas"][delta_name][role][metric] for row in rows], 2026080100 + 100 * (delta_name == "raw_minus_scrambled") + 10 * (role == "early") + (role == "external_ood") + (metric == "aupr")) for metric in ("auroc", "aupr")}
    aggregate["validation_early_raw_auroc_gap"] = _mean_ci([row["deltas"]["validation_early_raw_auroc_gap"] for row in rows], 2026080999)
    positive_ridge = int(sum(row["deltas"]["raw_minus_ridge"]["early"]["auroc"] > 0 for row in rows))
    nonnegative_ood = int(sum(row["deltas"]["raw_minus_ridge"]["external_ood"]["auroc"] >= 0 for row in rows))
    checks = {} if args.smoke else {
        "early_raw_beats_ridge_ge_0p010": aggregate["raw_minus_ridge"]["early"]["auroc"]["mean"] >= 0.010,
        "early_raw_beats_ridge_four_of_five": positive_ridge >= 4,
        "early_raw_beats_phase_scrambled_ge_0p020": aggregate["raw_minus_scrambled"]["early"]["auroc"]["mean"] >= 0.020,
        "raw_validation_early_gap_lt_0p150": aggregate["validation_early_raw_auroc_gap"]["mean"] < 0.150,
        "external_raw_not_worse_than_ridge": aggregate["raw_minus_ridge"]["external_ood"]["auroc"]["mean"] >= 0.0,
        "external_nonnegative_four_of_five": nonnegative_ood >= 4,
    }
    payload = {
        "status": "completed", "experiment": "RouteA_VFTISRJRawSourceDiagnostic_v1", "smoke": bool(args.smoke),
        "seeds": list(seeds), "data_sha256": data_sha, "external_data_sha256": EXTERNAL_SHA256,
        "ridge_alpha": float(args.ridge_alpha), "weak_threshold_db": float(args.weak_threshold_db), "aggregate": aggregate,
        "promotion_gate": {"checks": checks, "passed": bool(checks and all(checks.values())), "positive_early_vs_ridge_seeds": positive_ridge, "nonnegative_external_vs_ridge_seeds": nonnegative_ood},
        "scope": "fixed_feature_source_diagnostic_only_no_fusion_calibration_or_threshold", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_vft_isrj_raw_source_diagnostic_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--external-data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_isrj_ood_v1\closedset_6class_train.mat"))
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_raw_source_diagnostic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=10.0); parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
