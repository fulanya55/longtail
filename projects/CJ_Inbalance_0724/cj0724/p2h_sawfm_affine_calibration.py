"""P2h bounded affine calibration for frozen separated SawFM period outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import lsq_linear
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles, transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2e_sawfm_component_oracle import SAW_PERIOD_COLUMN
from cj0724.p2f_dedicated_sawfm_separator import POWERS, SawSeparatorDataset, evaluate_separator, metric
from cj0724.physics_parameter_specialists import build_physics_specialist


SEEDS = (2041, 2042, 2043)
SAW_CLASS_INDEX = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_calibrators(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    bias = float(np.clip(np.mean(target - prediction), -20.0e-6, 20.0e-6))
    design = np.column_stack((prediction, np.ones(len(prediction), dtype=np.float64)))
    result = lsq_linear(
        design, target,
        bounds=(np.asarray([0.5, -20.0e-6]), np.asarray([1.5, 20.0e-6])),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"bounded affine calibration failed: {result.message}")
    return {"bias_only_b": bias, "affine_a": float(result.x[0]), "affine_b": float(result.x[1])}


def apply_calibrators(prediction: np.ndarray, coefficients: dict[str, float]) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "C0_uncalibrated": prediction,
        "C1_bias_only": prediction + coefficients["bias_only_b"],
        "C2_bounded_affine": coefficients["affine_a"] * prediction + coefficients["affine_b"],
    }


def relative(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    return float(1.0 - candidate["mae"] / baseline["mae"])


def role_evaluation(
    separator: torch.nn.Module,
    expert: torch.nn.Module,
    files: dict[int, Path],
    indices: np.ndarray,
    device: torch.device,
    mean: float,
    scale: float,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    dataset = SawSeparatorDataset(files, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    output = evaluate_separator(separator, expert, loader, device, mean, scale)
    dataset.close()
    return output


def identity_audit(files: dict[int, Path], excluded_files: list[Path]) -> dict[str, object]:
    alignment = audit_power_alignment(files)
    identities, _ = read_identity_metadata(files[POWERS[0]])
    exclusions = []
    for path in excluded_files:
        prior, _ = read_identity_metadata(path)
        exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
    if any(row["overlap_count"] != 0 for row in exclusions):
        raise RuntimeError("P2h identities overlap a prior split")
    return {"alignment": alignment, "identity_count": int(len(identities)), "identity_sha256": _hash_int64(identities), "excluded_prior_splits": exclusions, "test_generated": False, "test_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", default="generated_data/p2h_sawfm_affine_calibration_seed2000000000_rechunked")
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--preregister", default="configs/p2h_sawfm_affine_calibration_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2h_sawfm_affine_calibration_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / args.development_root
    reference_root = project / args.reference_freeze_root
    specialist_root = project / args.specialist_root
    separator_root = project / args.separator_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    files = transition_files(root, "train", POWERS)
    audit = identity_audit(files, [Path(path) for path in args.excluded_identity_files])
    identities, labels = read_identity_metadata(files[POWERS[0]])
    roles_all = stratified_train_roles(labels, 20260915)
    with h5py.File(files[POWERS[0]], "r") as handle:
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    roles = {name: indices[(labels[indices, SAW_CLASS_INDEX] > 0) & mask[SAW_PERIOD_COLUMN, indices]] for name, indices in roles_all.items()}
    role_overlap = {
        "fit_selection": int(np.intersect1d(roles["fit"], roles["selection"]).size),
        "fit_calibration": int(np.intersect1d(roles["fit"], roles["calibration"]).size),
        "selection_calibration": int(np.intersect1d(roles["selection"], roles["calibration"]).size),
    }
    if any(role_overlap.values()):
        raise RuntimeError("P2h roles overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in SEEDS:
        freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
        mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
        expert_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
        separator_checkpoint = separator_root / f"seed_{seed}" / "S1_waveform_only" / "best.pt"
        expert = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        expert.load_state_dict(torch.load(expert_checkpoint, map_location=device, weights_only=True))
        separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
        separator.load_state_dict(torch.load(separator_checkpoint, map_location=device, weights_only=True))
        payloads = {role: role_evaluation(separator, expert, files, roles[role], device, mean, scale, args.batch_size) for role in ("fit", "selection", "calibration")}
        coefficients = fit_calibrators(payloads["fit"]["separated_prediction"], payloads["fit"]["target"])
        role_metrics = {}
        predictions = {}
        for role, payload in payloads.items():
            calibrated = apply_calibrators(payload["separated_prediction"], coefficients)
            raw_metrics = metric(payload["raw_prediction"], payload["target"])
            method_metrics = {name: metric(value, payload["target"]) for name, value in calibrated.items()}
            role_metrics[role] = {
                "raw": raw_metrics,
                **method_metrics,
                "C0_reduction_vs_raw": relative(method_metrics["C0_uncalibrated"], raw_metrics),
                "C1_reduction_vs_raw": relative(method_metrics["C1_bias_only"], raw_metrics),
                "C2_reduction_vs_raw": relative(method_metrics["C2_bounded_affine"], raw_metrics),
                "C2_improvement_over_C0": float(relative(method_metrics["C2_bounded_affine"], raw_metrics) - relative(method_metrics["C0_uncalibrated"], raw_metrics)),
                "C2_reduction_vs_C1": relative(method_metrics["C2_bounded_affine"], method_metrics["C1_bias_only"]),
            }
            predictions[role] = {"target": payload["target"], "raw": payload["raw_prediction"], **calibrated}
        run = {
            "seed": seed, "coefficients": coefficients,
            "roles": role_metrics,
            "separator_checkpoint_sha256": sha256_file(separator_checkpoint),
            "expert_checkpoint_sha256": sha256_file(expert_checkpoint),
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "receipt.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
        for role, values in predictions.items():
            np.savez_compressed(seed_dir / f"{role}_predictions.npz", **values)
        runs.append(run)
        del separator, expert
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({
            "seed": seed, "coefficients": coefficients,
            "selection_C2_reduction": role_metrics["selection"]["C2_reduction_vs_raw"],
            "calibration_C2_reduction": role_metrics["calibration"]["C2_reduction_vs_raw"],
            "calibration_C2_improvement_over_C0": role_metrics["calibration"]["C2_improvement_over_C0"],
        }), flush=True)

    gates = {
        "each_seed_calibration_C2_reduction_vs_raw_at_least_0p35": bool(all(run["roles"]["calibration"]["C2_reduction_vs_raw"] >= 0.35 for run in runs)),
        "each_seed_calibration_C2_improvement_over_C0_at_least_0p15": bool(all(run["roles"]["calibration"]["C2_improvement_over_C0"] >= 0.15 for run in runs)),
        "each_seed_calibration_C2_pearson_at_least_0p60": bool(all(run["roles"]["calibration"]["C2_bounded_affine"]["pearson"] >= 0.60 for run in runs)),
        "each_seed_selection_and_calibration_C2_improvement_positive": bool(all(run["roles"]["selection"]["C2_improvement_over_C0"] > 0.0 and run["roles"]["calibration"]["C2_improvement_over_C0"] > 0.0 for run in runs)),
        "each_seed_C2_not_worse_than_C1_by_more_than_0p01": bool(all(run["roles"]["calibration"]["C2_reduction_vs_C1"] >= -0.01 for run in runs)),
        "mean_calibration_C2_reduction_vs_raw_at_least_0p40": bool(np.mean([run["roles"]["calibration"]["C2_reduction_vs_raw"] for run in runs]) >= 0.40),
    }
    passed = bool(all(gates.values()))
    c1_pass = bool(all(run["roles"]["calibration"]["C1_reduction_vs_raw"] >= 0.35 for run in runs))
    if passed:
        interpretation = "Bounded affine calibration passed; freeze coefficients and confirm without refitting on two new identity seeds."
    elif c1_pass:
        interpretation = "Bias-only calibration passed while affine failed; retain C1 only."
    else:
        interpretation = "Calibration mismatch is not stable; do not attach or select checkpoints."
    summary = {
        "experiment": "P2h bounded affine calibration development",
        "protocol": vars(args), "preregister_sha256": sha256_file(preregister),
        "identity_audit": audit,
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "role_hashes": {name: _hash_int64(identities[indices]) for name, indices in roles.items()},
        "role_overlap": role_overlap,
        "promotion_gates": gates, "promotion_passed": passed, "C1_ablation_passed": c1_pass,
        "mean_calibration_reduction_vs_raw": {
            method: float(np.mean([run["roles"]["calibration"][f"{method}_reduction_vs_raw"] for run in runs]))
            for method in ("C0", "C1", "C2")
        },
        "interpretation": interpretation, "runs": runs,
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
