"""P2c fresh-development screen for frozen separator residual isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_unet
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles, transition_files
from cj0724.p1k_joint_level3_system import SAW_PARAMETER_COLUMN, _hash_int64
from cj0724.p2b_sawfm_power_sufficiency import PowerViewSawDataset
from cj0724.physics_parameter_specialists import build_physics_specialist


POWERS = (30, 35, 40, 45, 50)
HIGH_POWERS = (35, 40, 45, 50)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (2041, 2042, 2043)
SAW_CLASS_INDEX = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms_normalize_batch(iq: torch.Tensor) -> torch.Tensor:
    energy = iq.float().square().sum(1).mean(-1, keepdim=True).clamp_min(1.0e-12)
    return iq.float() / torch.sqrt(energy)[:, None, :]


def select_alpha(mae_by_alpha: dict[float, float]) -> float:
    return float(min(sorted(mae_by_alpha), key=lambda alpha: (float(mae_by_alpha[alpha]), float(alpha))))


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if np.std(prediction) <= 0.0 or np.std(target) <= 0.0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def subset_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    row_mask: np.ndarray | None = None,
) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if row_mask is None:
        row_mask = np.ones(len(target), dtype=bool)
    rows = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    if len(rows) == 0:
        raise ValueError("metric subset is empty")
    high_columns = [POWERS.index(power) for power in HIGH_POWERS]

    def metrics(columns: list[int]) -> dict[str, float]:
        selected_prediction = prediction[rows][:, columns]
        selected_target = np.repeat(target[rows, None], len(columns), axis=1)
        return {
            "mae": float(np.abs(selected_prediction - selected_target).mean()),
            "mae_us": float(np.abs(selected_prediction - selected_target).mean() * 1.0e6),
            "pearson": _correlation(selected_prediction, selected_target),
        }

    return {
        "all_30_50": metrics(list(range(len(POWERS)))),
        "high_35_50": metrics(high_columns),
        "per_power": {
            str(power): metrics([index]) for index, power in enumerate(POWERS)
        },
        "identity_count": int(len(rows)),
    }


def reduction(candidate: dict[str, object], baseline: dict[str, object], block: str = "high_35_50") -> float:
    return float(1.0 - float(candidate[block]["mae"]) / float(baseline[block]["mae"]))


@torch.inference_mode()
def predict_variants(
    unet: torch.nn.Module,
    saw: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    scale: float,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    unet.eval()
    saw.eval()
    output: dict[str, list[np.ndarray]] = {"R0_raw": [], "R1_estimated_component": []}
    output.update({f"R2_residual_alpha_{alpha:g}": [] for alpha in alphas})
    targets, indices = [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat = iq.reshape(batch_size * power_count, *iq.shape[2:]).float()
        components = unet.forward_components(flat, None)
        estimated = components["estimated_iq"].float()
        inputs = {
            "R0_raw": flat,
            "R1_estimated_component": rms_normalize_batch(estimated),
        }
        for alpha in alphas:
            inputs[f"R2_residual_alpha_{alpha:g}"] = rms_normalize_batch(flat - float(alpha) * estimated)
        for name, values in inputs.items():
            prediction = saw(values)["prediction"].reshape(batch_size, power_count)
            output[name].append((prediction * float(scale) + float(mean)).cpu().numpy())
        targets.append(batch["target"].numpy())
        indices.append(batch["sample_index"].numpy())
    return {name: np.concatenate(parts) for name, parts in output.items()}, np.concatenate(targets), np.concatenate(indices)


def identity_audit(files: dict[int, Path], excluded_files: list[Path]) -> dict[str, object]:
    alignment = audit_power_alignment(files)
    identities, _ = read_identity_metadata(files[POWERS[0]])
    exclusions = []
    for path in excluded_files:
        prior, _ = read_identity_metadata(path)
        exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
    if any(row["overlap_count"] != 0 for row in exclusions):
        raise RuntimeError("P2c identities overlap a prior split")
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
    parser.add_argument("--development-root", default="generated_data/p2c_sawfm_residual_isolation_seed1400000000_rechunked")
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--preregister", default="configs/p2c_sawfm_residual_isolation_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2c_sawfm_residual_isolation_development_v3_20260801")
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
    roles_all = stratified_train_roles(labels, 20260815)
    with h5py.File(files[POWERS[0]], "r") as handle:
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
        values = np.asarray(handle["parameter_values"][SAW_PARAMETER_COLUMN], dtype=np.float64)
    roles = {
        name: indices[(labels[indices, SAW_CLASS_INDEX] > 0) & mask[SAW_PARAMETER_COLUMN, indices]]
        for name, indices in roles_all.items()
    }
    fit_median = float(np.median(values[roles["fit"]]))
    role_overlap = {
        "fit_selection": int(np.intersect1d(roles["fit"], roles["selection"]).size),
        "fit_calibration": int(np.intersect1d(roles["fit"], roles["calibration"]).size),
        "selection_calibration": int(np.intersect1d(roles["selection"], roles["calibration"]).size),
    }
    if any(role_overlap.values()):
        raise RuntimeError("P2c roles overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in SEEDS:
        freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
        mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
        separator_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        specialist_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(separator_checkpoint, device)
        saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        saw.load_state_dict(torch.load(specialist_checkpoint, map_location=device, weights_only=True))
        saw.eval()

        selection_dataset = PowerViewSawDataset(files, roles["selection"], POWERS)
        selection_loader = DataLoader(selection_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        started = time.perf_counter()
        selection_prediction, selection_target, selection_indices = predict_variants(
            unet, saw, selection_loader, device, mean, scale,
        )
        selection_dataset.close()
        selection_alpha_mae = {
            alpha: subset_metrics(selection_prediction[f"R2_residual_alpha_{alpha:g}"], selection_target)["high_35_50"]["mae"]
            for alpha in ALPHAS
        }
        selected_alpha = select_alpha(selection_alpha_mae)

        calibration_dataset = PowerViewSawDataset(files, roles["calibration"], POWERS)
        calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        calibration_prediction, target, sample_indices = predict_variants(
            unet, saw, calibration_loader, device, mean, scale, (selected_alpha,),
        )
        calibration_dataset.close()
        calibration_labels = labels[sample_indices]
        variable_mask = calibration_labels[:, 3:6].sum(axis=1) >= 1
        no_variable_mask = ~variable_mask
        constant = np.full_like(calibration_prediction["R0_raw"], fit_median)
        selected_name = f"R2_residual_alpha_{selected_alpha:g}"
        metrics = {}
        for subset_name, subset_mask in {
            "overall": np.ones(len(target), dtype=bool),
            "variable_competitor": variable_mask,
            "no_variable_competitor": no_variable_mask,
        }.items():
            metrics[subset_name] = {
                "constant": subset_metrics(constant, target, subset_mask),
                "R0_raw": subset_metrics(calibration_prediction["R0_raw"], target, subset_mask),
                "R1_estimated_component": subset_metrics(calibration_prediction["R1_estimated_component"], target, subset_mask),
                "R2_residual": subset_metrics(calibration_prediction[selected_name], target, subset_mask),
            }
        comparisons = {
            "overall_reduction_vs_R0": reduction(metrics["overall"]["R2_residual"], metrics["overall"]["R0_raw"]),
            "overall_reduction_vs_constant": reduction(metrics["overall"]["R2_residual"], metrics["overall"]["constant"]),
            "variable_reduction_vs_R0": reduction(metrics["variable_competitor"]["R2_residual"], metrics["variable_competitor"]["R0_raw"]),
            "no_variable_reduction_vs_R0": reduction(metrics["no_variable_competitor"]["R2_residual"], metrics["no_variable_competitor"]["R0_raw"]),
            "R1_overall_reduction_vs_R0": reduction(metrics["overall"]["R1_estimated_component"], metrics["overall"]["R0_raw"]),
        }
        run = {
            "seed": seed,
            "selected_alpha": selected_alpha,
            "selection_alpha_high_mae_us": {str(alpha): float(mae * 1.0e6) for alpha, mae in selection_alpha_mae.items()},
            "selection_identity_sha256": _hash_int64(receiver_identities[selection_indices]),
            "calibration_identity_sha256": _hash_int64(receiver_identities[sample_indices]),
            "metrics": metrics,
            "comparisons": comparisons,
            "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
            "specialist_checkpoint_sha256": sha256_file(specialist_checkpoint),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        np.savez_compressed(
            seed_dir / "calibration_predictions.npz",
            sample_index=sample_indices,
            target=target,
            powers=np.asarray(POWERS, dtype=np.int16),
            constant=constant,
            R0_raw=calibration_prediction["R0_raw"],
            R1_estimated_component=calibration_prediction["R1_estimated_component"],
            R2_residual=calibration_prediction[selected_name],
        )
        (seed_dir / "receipt.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
        runs.append(run)
        del unet, saw
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"seed": seed, "selected_alpha": selected_alpha, **comparisons}), flush=True)

    r2_overall = [run["comparisons"]["overall_reduction_vs_R0"] for run in runs]
    r2_variable = [run["comparisons"]["variable_reduction_vs_R0"] for run in runs]
    r2_constant = [run["comparisons"]["overall_reduction_vs_constant"] for run in runs]
    r2_no_variable = [run["comparisons"]["no_variable_reduction_vs_R0"] for run in runs]
    r2_pearson = [run["metrics"]["overall"]["R2_residual"]["high_35_50"]["pearson"] for run in runs]
    r1_mean = float(np.mean([run["comparisons"]["R1_overall_reduction_vs_R0"] for run in runs]))
    r2_mean = float(np.mean(r2_overall))
    gates = {
        "each_seed_overall_reduction_vs_R0_at_least_0p02": bool(all(value >= 0.02 for value in r2_overall)),
        "each_seed_variable_reduction_vs_R0_at_least_0p05": bool(all(value >= 0.05 for value in r2_variable)),
        "each_seed_overall_reduction_vs_constant_at_least_0p10": bool(all(value >= 0.10 for value in r2_constant)),
        "each_seed_overall_pearson_at_least_0p30": bool(all(value >= 0.30 for value in r2_pearson)),
        "each_seed_no_variable_degradation_no_worse_than_minus_0p01": bool(all(value >= -0.01 for value in r2_no_variable)),
        "R1_negative_control_not_better_than_R2_mean": bool(r1_mean <= r2_mean),
    }
    passed = bool(all(gates.values()))
    if passed:
        interpretation = "Frozen residual isolation passed; preregister a trainable shared-front-end ablation on new identities."
    elif r1_mean > r2_mean:
        interpretation = "Estimated-component semantics do not support subtraction; stop reuse of this separator for SawFM."
    else:
        interpretation = "Residual isolation failed on variable competitors; require dedicated SawFM component supervision or a mixture-robust reset/ridge estimator."
    summary = {
        "experiment": "P2c frozen separator residual-isolation development-v3",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "identity_audit": audit,
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "role_hashes": {name: _hash_int64(receiver_identities[indices]) for name, indices in roles.items()},
        "role_overlap": role_overlap,
        "promotion_gates": gates,
        "promotion_passed": passed,
        "mean_comparisons": {
            "R2_overall_reduction_vs_R0": r2_mean,
            "R2_variable_reduction_vs_R0": float(np.mean(r2_variable)),
            "R2_overall_reduction_vs_constant": float(np.mean(r2_constant)),
            "R2_no_variable_reduction_vs_R0": float(np.mean(r2_no_variable)),
            "R1_overall_reduction_vs_R0": r1_mean,
        },
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
