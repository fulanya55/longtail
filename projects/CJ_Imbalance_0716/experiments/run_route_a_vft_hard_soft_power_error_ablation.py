"""Current VFT Ridge hard-mask, power-error, and soft-visibility ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _load_roles, _matched_view, _rank_metrics  # noqa: E402
from experiments.run_route_a_vft_ridge_stft_separability_v3 import _extract  # noqa: E402
from experiments.run_route_a_vft_weak_mixed_positive_ablation import TARGET_INDEX, _all_positive_matched_view  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


ARMS = ("HardNominal", "HardPowerError", "SoftVisibility")
ROLES = ("fit", "validation", "early")


def relative_component_power(labels: np.ndarray, power: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    power = np.asarray(power, dtype=np.float64)
    output = np.full(power.shape, np.nan, dtype=np.float64)
    for row in range(len(labels)):
        active = np.flatnonzero(labels[row])
        if len(active) < 2:
            output[row, active] = 0.0
            continue
        for index in active:
            output[row, index] = power[row, index] - np.max(power[row, active[active != index]])
    return output


def soft_visibility_weight(relative_db: np.ndarray, threshold_db: float = 10.0, temperature_db: float = 2.5) -> np.ndarray:
    z = (-float(threshold_db) - np.asarray(relative_db, dtype=np.float64)) / float(temperature_db)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))).astype(np.float64)


def _fit_weighted(x: np.ndarray, labels: np.ndarray, positive_strength: np.ndarray, alpha: float):
    labels = np.asarray(labels, dtype=np.uint8)
    strength = np.asarray(positive_strength, dtype=np.float64)
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("both classes are required")
    if np.any(strength[positive] <= 0) or not np.isfinite(strength).all():
        raise ValueError("positive strengths must be finite and positive")
    weight = np.zeros(len(labels), dtype=np.float64)
    weight[positive] = 0.5 * strength[positive] / strength[positive].sum()
    weight[negative] = 0.5 / negative.sum()
    model = RidgeClassifier(alpha=float(alpha), fit_intercept=True)
    model.fit(x, labels, sample_weight=weight)
    return model, weight


def _feature_roles(roles, device, batch_size):
    features = {role: _extract(roles[role]["signals"], device, batch_size) for role in ROLES}
    mean = features["fit"].mean(axis=0).astype(np.float32)
    std = np.maximum(features["fit"].std(axis=0), 1e-5).astype(np.float32)
    return {role: ((value - mean[None, :]) / std[None, :]).astype(np.float32) for role, value in features.items()}, mean, std


def _run_seed(args, config, labels6, power6, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; summary = root / "seed_summary.json"
    if args.resume and summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    normalized, mean, std = _feature_roles(roles, device, args.feature_batch_size)
    nominal_views, all_views, receipts = {}, {}, {}
    for role in ROLES:
        ids = roles[role]["sample_ids"]
        selected, y, receipt = _matched_view(roles[role]["targets6"], weak6[ids], TARGET_INDEX)
        nominal_views[role] = (selected, y.astype(np.uint8)); receipts[role] = {"nominal": receipt}
        selected, y, receipt = _all_positive_matched_view(roles[role]["targets6"], TARGET_INDEX)
        all_views[role] = (selected, y.astype(np.uint8)); receipts[role]["all_positive"] = receipt
    fit_ids = roles["fit"]["sample_ids"]
    error_rng = np.random.default_rng(2026081300 + int(seed))
    fit_error = np.zeros_like(power6[fit_ids], dtype=np.float64)
    active = labels6[fit_ids].astype(bool)
    fit_error[active] = error_rng.normal(0.0, float(args.power_error_std_db), size=int(active.sum()))
    perturbed_weak = compute_weak_component_mask(labels6[fit_ids], power6[fit_ids] + fit_error, args.weak_threshold_db)
    error_selected, error_y, error_receipt = _matched_view(roles["fit"]["targets6"], perturbed_weak, TARGET_INDEX)
    receipts["fit"]["power_error"] = {**error_receipt, "active_error_mean": float(fit_error[active].mean()), "active_error_std": float(fit_error[active].std(ddof=0))}
    results, models = {}, {}
    definitions = {
        # _fit_weighted receives a per-sample vector.  Hard-mask positives all
        # have unit visibility; negatives retain their fixed balanced mass.
        "HardNominal": (nominal_views["fit"][0], nominal_views["fit"][1], np.ones(len(nominal_views["fit"][1]), dtype=np.float64)),
        "HardPowerError": (error_selected, error_y.astype(np.uint8), np.ones(len(error_y), dtype=np.float64)),
        "SoftVisibility": (all_views["fit"][0], all_views["fit"][1], None),
    }
    for arm, (fit_selected, fit_y, strength) in definitions.items():
        if arm == "SoftVisibility":
            selected_ids = fit_ids[fit_selected]
            all_relative = relative_component_power(labels6[selected_ids], power6[selected_ids])[:, TARGET_INDEX]
            strength = np.ones(len(fit_y), dtype=np.float64)
            strength[fit_y == 1] = soft_visibility_weight(all_relative[fit_y == 1], args.weak_threshold_db, args.soft_temperature_db)
        model, weights = _fit_weighted(normalized["fit"][fit_selected], fit_y, strength, args.ridge_alpha)
        arm_result = {"feature_dim": int(mean.size), "fit_positive_mass_min": float(strength[fit_y == 1].min()), "fit_positive_mass_max": float(strength[fit_y == 1].max())}
        for role in ("validation", "early"):
            selected, target = nominal_views[role]
            arm_result[role] = _rank_metrics(target, model.decision_function(normalized[role][selected]))
        results[arm] = arm_result
        models[arm] = {"coef": model.coef_.astype(np.float32), "intercept": model.intercept_.astype(np.float32), "sample_weight": weights.astype(np.float64)}
    deltas = {f"{arm}_minus_HardNominal": {role: {metric: float(results[arm][role][metric] - results["HardNominal"][role][metric]) for metric in ("auroc", "aupr")} for role in ("validation", "early")} for arm in ("HardPowerError", "SoftVisibility")}
    np.savez_compressed(root / "ridge_models.npz", mean=mean, std=std, **{f"{arm}_{key}": value for arm, item in models.items() for key, value in item.items()})
    payload = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path),
               "ridge_alpha": float(args.ridge_alpha), "weak_threshold_db": float(args.weak_threshold_db), "power_error_std_db": float(args.power_error_std_db), "soft_temperature_db": float(args.soft_temperature_db),
               "results": results, "deltas": deltas, "receipts": receipts, "validation_used_for_selection": False, "threshold_selected": False,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary, payload); return payload


def run(args):
    config = load_config(args.config); _validate_config(config)
    path = _data_path(config); data_sha = _sha256(path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")): raise ValueError("test-v3 boundary is not sealed")
    labels = read_labels(path); values, _ = read_parameters(path); power = _component_power(values)
    weak = compute_weak_component_mask(labels, power, args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, labels, power, weak, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {role: {metric: _mean_ci([row["results"][arm][role][metric] for row in rows], 2026081300 + 100 * ARMS.index(arm) + 10 * (role == "early") + (metric == "aupr")) for metric in ("auroc", "aupr")} for role in ("validation", "early")}
    deltas = {name: {role: {metric: _mean_ci([row["deltas"][name][role][metric] for row in rows], 2026081800 + 100 * index + 10 * (role == "early") + (metric == "aupr")) for metric in ("auroc", "aupr")} for role in ("validation", "early")} for index, name in enumerate(("HardPowerError_minus_HardNominal", "SoftVisibility_minus_HardNominal"))}
    checks = {} if args.smoke else {"hard_mask_power_error_early_auroc_drop_le_0p010": deltas["HardPowerError_minus_HardNominal"]["early"]["auroc"]["mean"] >= -.010,
                                    "soft_visibility_improves_early_weak_auroc": deltas["SoftVisibility_minus_HardNominal"]["early"]["auroc"]["mean"] > 0,
                                    "soft_visibility_improves_early_weak_aupr": deltas["SoftVisibility_minus_HardNominal"]["early"]["aupr"]["mean"] > 0}
    payload = {"status": "completed", "experiment": "RouteA_VFTHardSoftPowerErrorAblation_v1", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "deltas_vs_hard_nominal": deltas, "mechanism_gate": {"checks": checks, "passed": bool(checks and all(checks.values()))},
               "scope": "current_vft_ridge_mechanism_ablation_no_calibration_threshold_or_candidate_selection", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "route_a_vft_hard_soft_power_error_ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_hard_soft_power_error_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=10.0); parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--power-error-std-db", type=float, default=1.0); parser.add_argument("--soft-temperature-db", type=float, default=2.5)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
