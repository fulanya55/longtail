"""Formal P2b development-v2 SawFM period physical-sufficiency screen.

Only the fresh train split is opened.  Identities are split into fit,
selection and calibration roles before any model is fitted.  The experiment
compares a frozen specialist, a fixed autocorrelation estimator, a
compute-matched 30 dB fine-tune, equal-identity nine-power fine-tuning, and
the same fine-tuning with a bounded cross-power consistency penalty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles, transition_files
from cj0724.p1k_joint_level3_system import SAW_PARAMETER_COLUMN, _hash_int64
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.train import set_seed


POWERS = tuple(range(10, 51, 5))
HIGH_POWERS = (35, 40, 45, 50)
LOW_MID_POWERS = (10, 15, 20, 25, 30)
SAW_CLASS_INDEX = 1
DEVELOPMENT_SEEDS = (3041, 3042, 3043)
REFERENCE_SEEDS = (2041, 2042, 2043)
VARIANTS = {
    "E2_single_30db_compute_matched": {"training_powers": (30,) * len(POWERS), "consistency_weight": 0.0},
    "E3_nine_power_equal_identity": {"training_powers": POWERS, "consistency_weight": 0.0},
    "E4_nine_power_consistency": {"training_powers": POWERS, "consistency_weight": 0.05},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PowerViewSawDataset(Dataset):
    """Return a frozen list of power views for one receiver-noise identity."""

    def __init__(self, files: dict[int, Path], indices: np.ndarray, view_powers: tuple[int, ...]) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.view_powers = tuple(int(power) for power in view_powers)
        self.datasets = {
            power: V73IQDataset(files[power], self.indices, [SAW_PARAMETER_COLUMN])
            for power in sorted(set(self.view_powers))
        }

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        views = [self.datasets[power][position] for power in self.view_powers]
        sample_indices = [int(view["sample_index"]) for view in views]
        if len(set(sample_indices)) != 1:
            raise RuntimeError("paired-power sample indices differ")
        targets = torch.stack([view["parameter_values"][0] for view in views])
        if not torch.allclose(targets, targets[:1].expand_as(targets), rtol=0.0, atol=1.0e-10):
            raise RuntimeError("SawFM period target differs across power views")
        return {
            "iq": torch.stack([view["iq"] for view in views]),
            "target": targets[0],
            "sample_index": torch.tensor(sample_indices[0], dtype=torch.int64),
        }

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()


def paired_saw_loss(
    normalized_prediction: torch.Tensor,
    target_raw: torch.Tensor,
    mean: float,
    scale: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = ((target_raw.float() - float(mean)) / float(scale))[:, None]
    supervised = F.smooth_l1_loss(normalized_prediction.float(), target.expand_as(normalized_prediction))
    centered = normalized_prediction.float() - normalized_prediction.float().mean(dim=1, keepdim=True)
    consistency = centered.square().mean()
    total = supervised + float(consistency_weight) * consistency
    return total, supervised, consistency


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if np.std(prediction) <= 0.0 or np.std(target) <= 0.0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def segment_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    powers: tuple[int, ...] = POWERS,
) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != (len(target), len(powers)):
        raise ValueError(f"prediction shape {prediction.shape} does not match targets and powers")
    error = np.abs(prediction - target[:, None])

    def block(selected: tuple[int, ...]) -> dict[str, float]:
        columns = [powers.index(power) for power in selected]
        block_prediction = prediction[:, columns]
        block_target = np.repeat(target[:, None], len(columns), axis=1)
        return {
            "mae": float(np.abs(block_prediction - block_target).mean()),
            "pearson_correlation": _correlation(block_prediction, block_target),
        }

    return {
        "all": {
            "mae": float(error.mean()),
            "pearson_correlation": _correlation(prediction, np.repeat(target[:, None], len(powers), axis=1)),
        },
        "high_35_50": block(HIGH_POWERS),
        "low_mid_10_30": block(LOW_MID_POWERS),
        "per_power": {
            str(power): {
                "mae": float(error[:, index].mean()),
                "pearson_correlation": _correlation(prediction[:, index], target),
            }
            for index, power in enumerate(powers)
        },
        "mean_within_identity_prediction_std": float(np.std(prediction, axis=1).mean()),
        "identity_count": int(len(target)),
        "view_count": int(error.size),
    }


def relative_mae_reduction(candidate: dict[str, object], baseline: dict[str, object], block: str) -> float:
    candidate_mae = float(candidate[block]["mae"])
    baseline_mae = float(baseline[block]["mae"])
    return float(1.0 - candidate_mae / baseline_mae)


@torch.inference_mode()
def predict_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets, identities = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        output = model(iq.reshape(batch_size * power_count, *iq.shape[2:]))["prediction"]
        predictions.append((output.reshape(batch_size, power_count) * float(scale) + float(mean)).cpu().numpy())
        targets.append(batch["target"].numpy())
        identities.append(batch["sample_index"].numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(identities)


@torch.inference_mode()
def predict_classical_autocorrelation(
    loader: DataLoader,
    device: torch.device,
    sample_rate_hz: float = 10.0e6,
    minimum_period_s: float = 20.0e-6,
    maximum_period_s: float = 80.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum_lag = int(round(minimum_period_s * sample_rate_hz))
    maximum_lag = int(round(maximum_period_s * sample_rate_hz))
    predictions, targets, identities = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        iq = iq.reshape(batch_size * power_count, *iq.shape[2:]).float()
        real, imag = iq[:, 0], iq[:, 1]
        cross = real[:, :-1] * imag[:, 1:] - imag[:, :-1] * real[:, 1:]
        dot = real[:, :-1] * real[:, 1:] + imag[:, :-1] * imag[:, 1:]
        frequency = torch.atan2(cross, dot) / torch.pi
        frequency = frequency - frequency.mean(dim=-1, keepdim=True)
        length = int(frequency.shape[-1])
        fft_length = 1 << (2 * length - 1).bit_length()
        spectrum = torch.fft.rfft(frequency, n=fft_length, dim=-1)
        correlation = torch.fft.irfft(spectrum * spectrum.conj(), n=fft_length, dim=-1)
        search = correlation[:, minimum_lag : maximum_lag + 1]
        lag = search.argmax(dim=-1) + minimum_lag
        prediction = lag.float().reshape(batch_size, power_count) / float(sample_rate_hz)
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["target"].numpy())
        identities.append(batch["sample_index"].numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(identities)


def _mean_epoch(rows: list[tuple[float, float, float]]) -> dict[str, float]:
    values = np.asarray(rows, dtype=np.float64)
    return {"total": float(values[:, 0].mean()), "supervised": float(values[:, 1].mean()), "consistency": float(values[:, 2].mean())}


def train_variant(
    name: str,
    spec: dict[str, object],
    files: dict[int, Path],
    roles: dict[str, np.ndarray],
    baseline_checkpoint: Path,
    output_dir: Path,
    development_seed: int,
    mean: float,
    scale: float,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, list[dict[str, object]], float]:
    set_seed(development_seed)
    training_dataset = PowerViewSawDataset(files, roles["fit"], tuple(spec["training_powers"]))
    selection_dataset = PowerViewSawDataset(files, roles["selection"], POWERS)
    generator = torch.Generator().manual_seed(development_seed)
    training_loader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        pin_memory=device.type == "cuda",
    )
    selection_loader = DataLoader(
        selection_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda",
    )
    model = build_physics_specialist("sawfm_period_sequence", args.width).to(device)
    model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir.mkdir(parents=True)
    best_loss, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            training_rows = []
            for batch in training_loader:
                optimizer.zero_grad(set_to_none=True)
                iq = batch["iq"].to(device)
                batch_size, view_count = iq.shape[:2]
                prediction = model(iq.reshape(batch_size * view_count, *iq.shape[2:]))["prediction"]
                prediction = prediction.reshape(batch_size, view_count)
                losses = paired_saw_loss(
                    prediction, batch["target"].to(device), mean, scale,
                    float(spec["consistency_weight"]),
                )
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
                training_rows.append(tuple(float(value.detach()) for value in losses))

            model.eval()
            selection_rows = []
            with torch.inference_mode():
                for batch in selection_loader:
                    iq = batch["iq"].to(device)
                    batch_size, view_count = iq.shape[:2]
                    prediction = model(iq.reshape(batch_size * view_count, *iq.shape[2:]))["prediction"]
                    prediction = prediction.reshape(batch_size, view_count)
                    selection_rows.append(tuple(float(value.detach()) for value in paired_saw_loss(
                        prediction, batch["target"].to(device), mean, scale,
                        float(spec["consistency_weight"]),
                    )))
            training_epoch = _mean_epoch(training_rows)
            selection_epoch = _mean_epoch(selection_rows)
            history.append({"epoch": epoch, "train": training_epoch, "selection": selection_epoch})
            if selection_epoch["total"] < best_loss:
                best_loss, stale = selection_epoch["total"], 0
                torch.save(model.state_dict(), output_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device, weights_only=True))
    finally:
        training_dataset.close()
        selection_dataset.close()
    return model, history, float(time.perf_counter() - started)


def identity_audit(files: dict[int, Path], excluded_files: list[Path]) -> dict[str, object]:
    alignment = audit_power_alignment(files)
    identities, labels = read_identity_metadata(files[POWERS[0]])
    exclusions = []
    for path in excluded_files:
        prior_identities, _ = read_identity_metadata(path)
        exclusions.append({
            "path": str(path.resolve()),
            "identity_count": int(len(prior_identities)),
            "overlap_count": int(np.intersect1d(identities, prior_identities).size),
        })
    if any(row["overlap_count"] != 0 for row in exclusions):
        raise RuntimeError("fresh P2b train identities overlap an excluded prior split")
    return {
        "alignment": alignment,
        "identity_count": int(len(identities)),
        "identity_sha256": _hash_int64(identities),
        "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "excluded_prior_splits": exclusions,
        "p2b_test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", default="generated_data/p2b_nine_power_confirmation_seed1300000000_rechunked_train_only")
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--baseline-specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--preregister", default="configs/p2b_sawfm_power_sufficiency_preregister_20260801.json")
    parser.add_argument("--amendment", default="configs/p2b_registration_amendment_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2b_sawfm_power_sufficiency_development_v2_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    development_root = project / args.development_root
    reference_root = project / args.reference_freeze_root
    baseline_root = project / args.baseline_specialist_root
    preregister = project / args.preregister
    amendment = project / args.amendment
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)

    files = transition_files(development_root, "train", POWERS)
    audit = identity_audit(files, [Path(path) for path in args.excluded_identity_files])
    identities, labels = read_identity_metadata(files[POWERS[0]])
    roles_all = stratified_train_roles(labels, 20260805)
    with h5py.File(files[POWERS[0]], "r") as handle:
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
        target_values = np.asarray(handle["parameter_values"][SAW_PARAMETER_COLUMN], dtype=np.float64)
    roles = {
        name: indices[(labels[indices, SAW_CLASS_INDEX] > 0) & mask[SAW_PARAMETER_COLUMN, indices]]
        for name, indices in roles_all.items()
    }
    if any(len(indices) == 0 for indices in roles.values()):
        raise RuntimeError("a SawFM-positive role is empty")
    role_overlap = {
        "fit_selection": int(np.intersect1d(roles["fit"], roles["selection"]).size),
        "fit_calibration": int(np.intersect1d(roles["fit"], roles["calibration"]).size),
        "selection_calibration": int(np.intersect1d(roles["selection"], roles["calibration"]).size),
    }
    if any(role_overlap.values()):
        raise RuntimeError("development roles overlap")
    fit_median = float(np.median(target_values[roles["fit"]]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_rows = []

    for development_seed, reference_seed in zip(DEVELOPMENT_SEEDS, REFERENCE_SEEDS):
        seed_dir = output_root / f"seed_{development_seed}"
        seed_dir.mkdir(parents=True)
        freeze = json.loads((reference_root / f"seed_{reference_seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
        mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
        checkpoint = baseline_root / "sawfm_period_sequence" / f"seed_{reference_seed}" / "best.pt"
        calibration_dataset = PowerViewSawDataset(files, roles["calibration"], POWERS)
        calibration_loader = DataLoader(
            calibration_dataset, batch_size=args.batch_size, shuffle=False,
            pin_memory=device.type == "cuda",
        )
        baseline_model = build_physics_specialist("sawfm_period_sequence", args.width).to(device)
        baseline_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        baseline_prediction, target, calibration_identities = predict_model(
            baseline_model, calibration_loader, device, mean, scale,
        )
        classical_prediction, classical_target, classical_identities = predict_classical_autocorrelation(
            calibration_loader, device,
        )
        if not np.array_equal(target, classical_target) or not np.array_equal(calibration_identities, classical_identities):
            raise RuntimeError("classical and frozen evaluator order differs")
        constant_prediction = np.full_like(baseline_prediction, fit_median)
        metrics = {
            "E_const_fit_median": segment_metrics(constant_prediction, target),
            "E0_frozen_specialist": segment_metrics(baseline_prediction, target),
            "E1_classical_autocorrelation": segment_metrics(classical_prediction, target),
        }
        predictions = {
            "E_const_fit_median": constant_prediction,
            "E0_frozen_specialist": baseline_prediction,
            "E1_classical_autocorrelation": classical_prediction,
        }
        histories, costs, checkpoint_hashes = {}, {}, {}
        for variant_name, spec in VARIANTS.items():
            model, history, elapsed = train_variant(
                variant_name, spec, files, roles, checkpoint, seed_dir / variant_name,
                development_seed, mean, scale, device, args,
            )
            prediction, candidate_target, candidate_identities = predict_model(
                model, calibration_loader, device, mean, scale,
            )
            if not np.array_equal(target, candidate_target) or not np.array_equal(calibration_identities, candidate_identities):
                raise RuntimeError(f"{variant_name} calibration order differs")
            metrics[variant_name] = segment_metrics(prediction, target)
            predictions[variant_name] = prediction
            histories[variant_name] = history
            costs[variant_name] = {"train_seconds": elapsed, "epochs_ran": len(history)}
            checkpoint_hashes[variant_name] = sha256_file(seed_dir / variant_name / "best.pt")
            (seed_dir / variant_name / "history.json").write_text(
                json.dumps(_jsonable(history), indent=2), encoding="utf-8",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        comparisons = {}
        for name, value in metrics.items():
            if name == "E_const_fit_median":
                continue
            comparisons[name] = {
                "high_reduction_vs_constant": relative_mae_reduction(value, metrics["E_const_fit_median"], "high_35_50"),
                "high_reduction_vs_E0": relative_mae_reduction(value, metrics["E0_frozen_specialist"], "high_35_50"),
                "low_mid_reduction_vs_E0": relative_mae_reduction(value, metrics["E0_frozen_specialist"], "low_mid_10_30"),
                "all_reduction_vs_E0": relative_mae_reduction(value, metrics["E0_frozen_specialist"], "all"),
            }
        row = {
            "development_seed": development_seed,
            "reference_checkpoint_seed": reference_seed,
            "reference_checkpoint": str(checkpoint.resolve()),
            "reference_checkpoint_sha256": sha256_file(checkpoint),
            "normalization": {"mean": mean, "scale": scale},
            "metrics": metrics,
            "comparisons": comparisons,
            "training_cost": costs,
            "candidate_checkpoint_sha256": checkpoint_hashes,
            "calibration_identity_count": int(len(calibration_identities)),
            "calibration_identity_sha256": _hash_int64(identities[roles["calibration"]]),
        }
        np.savez_compressed(
            seed_dir / "calibration_predictions.npz",
            identity=calibration_identities,
            target=target,
            powers=np.asarray(POWERS, dtype=np.int16),
            **predictions,
        )
        (seed_dir / "receipt.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
        seed_rows.append(row)
        calibration_dataset.close()
        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({
            "development_seed": development_seed,
            "E4_high_reduction_vs_constant": comparisons["E4_nine_power_consistency"]["high_reduction_vs_constant"],
            "E4_high_reduction_vs_E0": comparisons["E4_nine_power_consistency"]["high_reduction_vs_E0"],
        }), flush=True)

    e4 = [row["comparisons"]["E4_nine_power_consistency"] for row in seed_rows]
    e3 = [row["comparisons"]["E3_nine_power_equal_identity"] for row in seed_rows]
    e4_correlation = [
        float(row["metrics"]["E4_nine_power_consistency"]["high_35_50"]["pearson_correlation"])
        for row in seed_rows
    ]
    gates = {
        "E4_each_seed_high_reduction_vs_constant_at_least_0p10": bool(all(row["high_reduction_vs_constant"] >= 0.10 for row in e4)),
        "E4_each_seed_high_pearson_at_least_0p30": bool(all(value >= 0.30 for value in e4_correlation)),
        "E4_each_seed_high_reduction_vs_E0_positive": bool(all(row["high_reduction_vs_E0"] > 0.0 for row in e4)),
        "E4_mean_high_reduction_vs_E0_at_least_0p02": bool(np.mean([row["high_reduction_vs_E0"] for row in e4]) >= 0.02),
        "E4_each_seed_low_mid_degradation_no_worse_than_minus_0p01": bool(all(row["low_mid_reduction_vs_E0"] >= -0.01 for row in e4)),
        "E4_not_worse_than_E3_mean_high_by_more_than_0p005": bool(
            np.mean([row["high_reduction_vs_E0"] for row in e4])
            >= np.mean([row["high_reduction_vs_E0"] for row in e3]) - 0.005
        ),
    }
    e1_physical_gate = {
        "high_reduction_vs_constant_at_least_0p10": bool(all(
            row["comparisons"]["E1_classical_autocorrelation"]["high_reduction_vs_constant"] >= 0.10
            for row in seed_rows
        )),
        "high_pearson_at_least_0p30": bool(all(
            row["metrics"]["E1_classical_autocorrelation"]["high_35_50"]["pearson_correlation"] >= 0.30
            for row in seed_rows
        )),
    }
    e3_pass = bool(
        all(row["high_reduction_vs_constant"] >= 0.10 and row["high_reduction_vs_E0"] > 0.0 and row["low_mid_reduction_vs_E0"] >= -0.01 for row in e3)
        and all(row["metrics"]["E3_nine_power_equal_identity"]["high_35_50"]["pearson_correlation"] >= 0.30 for row in seed_rows)
        and np.mean([row["high_reduction_vs_E0"] for row in e3]) >= 0.02
    )
    promotion_passed = bool(all(gates.values()))
    if promotion_passed:
        interpretation = "E4 passed development; create new identity confirmation data before any test use."
    elif e3_pass:
        interpretation = "Power-balanced supervision passed but consistency regularization failed; retain E3 and stop E4."
    elif all(e1_physical_gate.values()):
        interpretation = "Fixed autocorrelation is physically sufficient while learned variants failed; prefer explicit estimation and stop neural-capacity expansion."
    else:
        interpretation = "Oracle-positive physical sufficiency failed; inspect mixture isolation and do not deepen the SawFM expert."

    summary = {
        "experiment": "P2b formal SawFM period power sufficiency development-v2",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "registration_amendment_sha256": sha256_file(amendment),
        "development_root": str(development_root.resolve()),
        "identity_audit": audit,
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "role_hashes": {name: _hash_int64(identities[indices]) for name, indices in roles.items()},
        "role_overlap": role_overlap,
        "fit_target_median": fit_median,
        "promotion_gates": gates,
        "promotion_passed": promotion_passed,
        "E1_physical_gate": e1_physical_gate,
        "E3_ablation_passed": e3_pass,
        "interpretation": interpretation,
        "mean_comparisons": {
            name: {
                key: float(np.mean([row["comparisons"][name][key] for row in seed_rows]))
                for key in seed_rows[0]["comparisons"][name]
            }
            for name in seed_rows[0]["comparisons"]
        },
        "runs": seed_rows,
        "p2b_test_generated_but_opened": False,
        "p2a_test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "identity_audit.json").write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": promotion_passed, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
