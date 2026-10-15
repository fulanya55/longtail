"""P2d sample-level bounded router between raw and residual SawFM estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_unet
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles, transition_files
from cj0724.p1k_joint_level3_system import SAW_PARAMETER_COLUMN, _hash_int64
from cj0724.p2b_sawfm_power_sufficiency import PowerViewSawDataset
from cj0724.p2c_sawfm_residual_isolation import rms_normalize_batch
from cj0724.physics_parameter_specialists import build_physics_specialist


POWERS = (35, 40, 45, 50)
SEEDS = (2041, 2042, 2043)
ALPHA_BY_SEED = {2041: 0.5, 2042: 1.0, 2043: 1.0}
SAW_CLASS_INDEX = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    selected = np.asarray(mask, dtype=bool)
    error = np.abs(prediction[selected] - target[selected])
    correlation = 0.0
    if np.std(prediction[selected]) > 0.0 and np.std(target[selected]) > 0.0:
        correlation = float(np.corrcoef(prediction[selected], target[selected])[0, 1])
    return {
        "mae": float(error.mean()),
        "mae_us": float(error.mean() * 1.0e6),
        "pearson": correlation,
        "row_count": int(selected.sum()),
    }


def relative_reduction(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    if baseline["mae"] <= 0.0:
        return 0.0 if candidate["mae"] <= 0.0 else -float("inf")
    return float(1.0 - candidate["mae"] / baseline["mae"])


def select_threshold(
    predicted_gain: np.ndarray,
    raw_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    target: np.ndarray,
    no_variable_mask: np.ndarray,
    coverage_max: float = 0.80,
    no_variable_floor: float = -0.01,
) -> dict[str, float | bool]:
    predicted_gain = np.asarray(predicted_gain, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, 41)
    candidates = np.unique(np.concatenate(([np.inf], np.quantile(predicted_gain, quantiles))))
    raw_all = row_metrics(raw_prediction, target)
    raw_no_variable = row_metrics(raw_prediction, target, no_variable_mask)
    rows = []
    for threshold in candidates:
        accept = predicted_gain >= threshold
        coverage = float(accept.mean())
        if coverage > coverage_max + 1.0e-12:
            continue
        routed = np.where(accept, residual_prediction, raw_prediction)
        overall_reduction = relative_reduction(row_metrics(routed, target), raw_all)
        no_variable_reduction = relative_reduction(row_metrics(routed, target, no_variable_mask), raw_no_variable)
        eligible = bool(no_variable_reduction >= no_variable_floor - 1.0e-12)
        rows.append({
            "threshold": float(threshold),
            "coverage": coverage,
            "overall_reduction": overall_reduction,
            "no_variable_reduction": no_variable_reduction,
            "eligible": eligible,
        })
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        return {"threshold": float("inf"), "coverage": 0.0, "overall_reduction": 0.0, "no_variable_reduction": 0.0, "eligible": False}
    return max(eligible_rows, key=lambda row: (row["overall_reduction"], -row["coverage"], row["threshold"]))


@torch.inference_mode()
def collect_payload(
    unet: torch.nn.Module,
    saw: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    scale: float,
    alpha: float,
) -> dict[str, np.ndarray]:
    features, raw_predictions, residual_predictions = [], [], []
    targets, sample_indices, power_values = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat = iq.reshape(batch_size * power_count, *iq.shape[2:]).float()
        components = unet.forward_components(flat, None)
        estimated = components["estimated_iq"].float()
        residual_unscaled = flat - float(alpha) * estimated
        residual = rms_normalize_batch(residual_unscaled)
        raw_normalized = saw(flat)["prediction"]
        residual_normalized = saw(residual)["prediction"]
        raw_prediction = raw_normalized * float(scale) + float(mean)
        residual_prediction = residual_normalized * float(scale) + float(mean)
        raw_energy = flat.square().sum(1).mean(-1).clamp_min(1.0e-12)
        estimated_energy = estimated.square().sum(1).mean(-1)
        residual_energy = residual_unscaled.square().sum(1).mean(-1)
        cosine = F.cosine_similarity(flat.flatten(1), estimated.flatten(1), dim=1)
        probability = torch.sigmoid(components["logits"].float())
        feature = torch.column_stack((
            probability,
            estimated_energy / raw_energy,
            residual_energy / raw_energy,
            cosine,
            (raw_prediction - residual_prediction).abs() / float(scale),
            raw_normalized,
            residual_normalized,
        ))
        features.append(feature.cpu().numpy())
        raw_predictions.append(raw_prediction.cpu().numpy())
        residual_predictions.append(residual_prediction.cpu().numpy())
        targets.append(np.repeat(batch["target"].numpy(), power_count))
        sample_indices.append(np.repeat(batch["sample_index"].numpy(), power_count))
        power_values.append(np.tile(np.asarray(POWERS, dtype=np.int16), batch_size))
    return {
        "features": np.concatenate(features).astype(np.float32),
        "raw_prediction": np.concatenate(raw_predictions).astype(np.float32),
        "residual_prediction": np.concatenate(residual_predictions).astype(np.float32),
        "target": np.concatenate(targets).astype(np.float32),
        "sample_index": np.concatenate(sample_indices).astype(np.int64),
        "power": np.concatenate(power_values).astype(np.int16),
    }


def identity_audit(files: dict[int, Path], excluded_files: list[Path]) -> dict[str, object]:
    alignment = audit_power_alignment(files)
    identities, _ = read_identity_metadata(files[POWERS[0]])
    exclusions = []
    for path in excluded_files:
        prior, _ = read_identity_metadata(path)
        exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
    if any(row["overlap_count"] != 0 for row in exclusions):
        raise RuntimeError("P2d identities overlap a prior split")
    return {
        "alignment": alignment,
        "identity_count": int(len(identities)),
        "identity_sha256": _hash_int64(identities),
        "excluded_prior_splits": exclusions,
        "test_generated": False,
        "test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", default="generated_data/p2d_sawfm_bounded_router_seed1500000000_rechunked")
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--preregister", default="configs/p2d_sawfm_bounded_residual_router_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2d_sawfm_bounded_residual_router_development_v4_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / args.development_root
    reference_root = project / args.reference_freeze_root
    separator_root = project / args.separator_root
    specialist_root = project / args.specialist_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    files = transition_files(root, "train", POWERS)
    audit = identity_audit(files, [Path(path) for path in args.excluded_identity_files])
    receiver_identities, labels = read_identity_metadata(files[POWERS[0]])
    roles_all = stratified_train_roles(labels, 20260825)
    with h5py.File(files[POWERS[0]], "r") as handle:
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    roles = {
        name: indices[(labels[indices, SAW_CLASS_INDEX] > 0) & mask[SAW_PARAMETER_COLUMN, indices]]
        for name, indices in roles_all.items()
    }
    role_overlap = {
        "fit_selection": int(np.intersect1d(roles["fit"], roles["selection"]).size),
        "fit_calibration": int(np.intersect1d(roles["fit"], roles["calibration"]).size),
        "selection_calibration": int(np.intersect1d(roles["selection"], roles["calibration"]).size),
    }
    if any(role_overlap.values()):
        raise RuntimeError("P2d roles overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in SEEDS:
        started = time.perf_counter()
        freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
        mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
        separator_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        specialist_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(separator_checkpoint, device)
        saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        saw.load_state_dict(torch.load(specialist_checkpoint, map_location=device, weights_only=True))
        saw.eval()
        payloads = {}
        for role in ("fit", "selection", "calibration"):
            dataset = PowerViewSawDataset(files, roles[role], POWERS)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
            payloads[role] = collect_payload(unet, saw, loader, device, mean, scale, ALPHA_BY_SEED[seed])
            dataset.close()

        fit = payloads["fit"]
        fit_gain = np.abs(fit["raw_prediction"] - fit["target"]) - np.abs(fit["residual_prediction"] - fit["target"])
        router = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=seed,
        )
        router.fit(fit["features"], fit_gain)
        selection = payloads["selection"]
        selection_score = router.predict(selection["features"])
        selection_labels = labels[selection["sample_index"]]
        selection_no_variable = selection_labels[:, 3:6].sum(axis=1) == 0
        threshold = select_threshold(
            selection_score, selection["raw_prediction"], selection["residual_prediction"],
            selection["target"], selection_no_variable,
        )

        calibration = payloads["calibration"]
        score = router.predict(calibration["features"])
        accept = score >= float(threshold["threshold"])
        routed_prediction = np.where(accept, calibration["residual_prediction"], calibration["raw_prediction"])
        calibration_labels = labels[calibration["sample_index"]]
        variable = calibration_labels[:, 3:6].sum(axis=1) >= 1
        no_variable = ~variable
        metrics = {}
        for name, selected in {
            "overall": np.ones(len(variable), dtype=bool),
            "variable_competitor": variable,
            "no_variable_competitor": no_variable,
        }.items():
            raw_metric = row_metrics(calibration["raw_prediction"], calibration["target"], selected)
            residual_metric = row_metrics(calibration["residual_prediction"], calibration["target"], selected)
            routed_metric = row_metrics(routed_prediction, calibration["target"], selected)
            metrics[name] = {
                "raw": raw_metric,
                "unconditional_residual": residual_metric,
                "routed": routed_metric,
                "unconditional_reduction_vs_raw": relative_reduction(residual_metric, raw_metric),
                "routed_reduction_vs_raw": relative_reduction(routed_metric, raw_metric),
            }
        actual_gain = np.abs(calibration["raw_prediction"] - calibration["target"]) - np.abs(calibration["residual_prediction"] - calibration["target"])
        separation = spearmanr(score, actual_gain).statistic
        separation = 0.0 if not np.isfinite(separation) else float(separation)
        run = {
            "seed": seed,
            "alpha": ALPHA_BY_SEED[seed],
            "selection": threshold,
            "calibration_coverage": float(accept.mean()),
            "predicted_actual_gain_spearman": separation,
            "metrics": metrics,
            "fit_identity_sha256": _hash_int64(receiver_identities[roles["fit"]]),
            "selection_identity_sha256": _hash_int64(receiver_identities[roles["selection"]]),
            "calibration_identity_sha256": _hash_int64(receiver_identities[roles["calibration"]]),
            "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
            "specialist_checkpoint_sha256": sha256_file(specialist_checkpoint),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        joblib.dump(router, seed_dir / "benefit_router.joblib")
        np.savez_compressed(
            seed_dir / "calibration_predictions.npz",
            sample_index=calibration["sample_index"], power=calibration["power"],
            target=calibration["target"], raw_prediction=calibration["raw_prediction"],
            residual_prediction=calibration["residual_prediction"], routed_prediction=routed_prediction,
            predicted_gain=score, accept=accept,
        )
        (seed_dir / "receipt.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
        runs.append(run)
        del unet, saw, router
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({
            "seed": seed, "coverage": run["calibration_coverage"], "spearman": separation,
            "overall_reduction": metrics["overall"]["routed_reduction_vs_raw"],
            "variable_reduction": metrics["variable_competitor"]["routed_reduction_vs_raw"],
            "no_variable_reduction": metrics["no_variable_competitor"]["routed_reduction_vs_raw"],
        }), flush=True)

    gates = {
        "each_seed_overall_reduction_at_least_0p05": bool(all(run["metrics"]["overall"]["routed_reduction_vs_raw"] >= 0.05 for run in runs)),
        "each_seed_variable_reduction_at_least_0p05": bool(all(run["metrics"]["variable_competitor"]["routed_reduction_vs_raw"] >= 0.05 for run in runs)),
        "each_seed_no_variable_degradation_no_worse_than_minus_0p01": bool(all(run["metrics"]["no_variable_competitor"]["routed_reduction_vs_raw"] >= -0.01 for run in runs)),
        "each_seed_coverage_between_0p05_and_0p80": bool(all(0.05 <= run["calibration_coverage"] <= 0.80 for run in runs)),
        "each_seed_gain_spearman_at_least_0p10": bool(all(run["predicted_actual_gain_spearman"] >= 0.10 for run in runs)),
    }
    passed = bool(all(gates.values()))
    if passed:
        interpretation = "Bounded residual routing passed; create a fresh two-seed confirmation and end-to-end parameter audit."
    elif not gates["each_seed_gain_spearman_at_least_0p10"]:
        interpretation = "Separator confidence features do not reliably expose residual benefit; stop learned routing."
    elif not gates["each_seed_no_variable_degradation_no_worse_than_minus_0p01"]:
        interpretation = "Benefit is partly predictable but the route is not safe on no-variable compounds; a new explicit risk target would be required."
    else:
        interpretation = "Bounded routing gain is insufficient; stop this separator route and pursue dedicated SawFM component supervision or reset/ridge extraction."
    summary = {
        "experiment": "P2d bounded residual benefit router development-v4",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "identity_audit": audit,
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "role_hashes": {name: _hash_int64(receiver_identities[indices]) for name, indices in roles.items()},
        "role_overlap": role_overlap,
        "promotion_gates": gates,
        "promotion_passed": passed,
        "mean_routed_reduction_vs_raw": {
            subset: float(np.mean([run["metrics"][subset]["routed_reduction_vs_raw"] for run in runs]))
            for subset in ("overall", "variable_competitor", "no_variable_competitor")
        },
        "mean_unconditional_reduction_vs_raw": float(np.mean([run["metrics"]["overall"]["unconditional_reduction_vs_raw"] for run in runs])),
        "interpretation": interpretation,
        "runs": runs,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "identity_audit.json").write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
