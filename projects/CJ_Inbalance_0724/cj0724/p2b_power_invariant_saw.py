"""P2b: identity-paired nine-power consistency fine-tuning for SawFM period.

The frozen root-trained SawFM specialist is fine-tuned only on P2a training
identities.  All nine power views of one receiver-noise identity remain in the
same role and are presented together.  The supervised period loss is combined
with a small within-identity prediction-variance penalty.  Promotion is judged
on the untouched P2a calibration identities; no P2a test prediction is opened.
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
from cj0724.dual_axis import read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.oracle_swept_isolation_experiment import set_seed
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles, transition_files
from cj0724.p1k_joint_level3_system import (
    MODEL_SEEDS,
    SAW_PARAMETER_COLUMN,
    _hash_int64,
)
from cj0724.physics_parameter_specialists import build_physics_specialist


POWERS = tuple(range(10, 51, 5))
SAW_CLASS_INDEX = 1


class PairedPowerSawDataset(Dataset):
    """Return all power views for one identity without breaking role grouping."""

    def __init__(self, files: dict[int, Path], indices: np.ndarray) -> None:
        self.powers = tuple(sorted(files))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.datasets = {
            power: V73IQDataset(files[power], self.indices, [SAW_PARAMETER_COLUMN])
            for power in self.powers
        }

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        views = [self.datasets[power][position] for power in self.powers]
        sample_indices = [int(view["sample_index"]) for view in views]
        if len(set(sample_indices)) != 1:
            raise RuntimeError("paired-power sample indices differ")
        target = torch.stack([view["parameter_values"][0] for view in views])
        if not torch.allclose(target, target[:1].expand_as(target), rtol=0.0, atol=1.0e-10):
            raise RuntimeError("SawFM period target differs across paired powers")
        return {
            "iq": torch.stack([view["iq"] for view in views]),
            "target": target[0],
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
    """Supervise every view and penalize power-dependent predictions."""
    target = ((target_raw.float() - float(mean)) / float(scale))[:, None]
    supervised = F.smooth_l1_loss(normalized_prediction.float(), target.expand_as(normalized_prediction))
    centered = normalized_prediction.float() - normalized_prediction.float().mean(dim=1, keepdim=True)
    consistency = centered.square().mean()
    total = supervised + float(consistency_weight) * consistency
    return total, supervised, consistency


@torch.inference_mode()
def predict_paired(
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
        output = output.reshape(batch_size, power_count) * float(scale) + float(mean)
        predictions.append(output.cpu().numpy())
        targets.append(batch["target"].numpy())
        identities.append(batch["sample_index"].numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(identities)


def paired_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    error = np.abs(np.asarray(prediction) - np.asarray(target)[:, None])
    flat_prediction = np.asarray(prediction).reshape(-1)
    flat_target = np.repeat(np.asarray(target), prediction.shape[1])
    correlation = float(np.corrcoef(flat_prediction, flat_target)[0, 1])
    return {
        "mae": float(error.mean()),
        "per_power_mae": {str(power): float(error[:, index].mean()) for index, power in enumerate(POWERS)},
        "pearson_correlation": correlation,
        "mean_within_identity_std": float(np.std(prediction, axis=1).mean()),
        "identity_count": int(len(target)),
        "view_count": int(error.size),
    }


def relative_reduction(candidate: dict[str, object], baseline: dict[str, object]) -> float:
    return float(1.0 - float(candidate["mae"]) / float(baseline["mae"]))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", default="generated_data/p2a_nine_power_generalization_seed1200000000_rechunked")
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--baseline-specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--output-root", default="outputs/p2b_power_invariant_saw_specialists_20260801")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    development_root = project / args.development_root
    reference_root = project / args.reference_freeze_root
    baseline_root = project / args.baseline_specialist_root
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    files = transition_files(development_root, "train", POWERS)
    identities, labels = read_identity_metadata(files[POWERS[0]])
    roles = stratified_train_roles(labels, 20260805)
    with h5py.File(files[POWERS[0]], "r") as handle:
        parameter_mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    positive_roles = {
        name: indices[(labels[indices, SAW_CLASS_INDEX] > 0) & parameter_mask[SAW_PARAMETER_COLUMN, indices]]
        for name, indices in roles.items()
    }
    if any(len(indices) == 0 for indices in positive_roles.values()):
        raise RuntimeError("a SawFM-positive development role is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_runs = []
    for seed in MODEL_SEEDS:
        set_seed(seed)
        freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
        mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
        baseline_checkpoint = baseline_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
        run_dir = output_root / "sawfm_period_sequence" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        datasets = {name: PairedPowerSawDataset(files, indices) for name, indices in positive_roles.items()}
        loaders = {
            "fit": DataLoader(
                datasets["fit"], batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
            ),
            "selection": DataLoader(datasets["selection"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
            "calibration": DataLoader(datasets["calibration"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
        }
        model = build_physics_specialist("sawfm_period_sequence", args.width).to(device)
        model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_loss, stale, history = float("inf"), 0, []
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_rows = []
            for batch in loaders["fit"]:
                optimizer.zero_grad(set_to_none=True)
                iq = batch["iq"].to(device)
                batch_size, power_count = iq.shape[:2]
                prediction = model(iq.reshape(batch_size * power_count, *iq.shape[2:]))["prediction"]
                prediction = prediction.reshape(batch_size, power_count)
                total, supervised, consistency = paired_saw_loss(
                    prediction, batch["target"].to(device), mean, scale, args.consistency_weight,
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_rows.append((float(total.detach()), float(supervised.detach()), float(consistency.detach())))
            model.eval()
            selection_rows = []
            with torch.inference_mode():
                for batch in loaders["selection"]:
                    iq = batch["iq"].to(device)
                    batch_size, power_count = iq.shape[:2]
                    prediction = model(iq.reshape(batch_size * power_count, *iq.shape[2:]))["prediction"]
                    prediction = prediction.reshape(batch_size, power_count)
                    selection_rows.append(tuple(float(value.detach()) for value in paired_saw_loss(
                        prediction, batch["target"].to(device), mean, scale, args.consistency_weight,
                    )))
            selection_loss = float(np.mean([row[0] for row in selection_rows]))
            history.append({
                "epoch": epoch,
                "train_total": float(np.mean([row[0] for row in train_rows])),
                "train_supervised": float(np.mean([row[1] for row in train_rows])),
                "train_consistency": float(np.mean([row[2] for row in train_rows])),
                "selection_total": selection_loss,
                "selection_supervised": float(np.mean([row[1] for row in selection_rows])),
                "selection_consistency": float(np.mean([row[2] for row in selection_rows])),
            })
            if selection_loss < best_loss:
                best_loss, stale = selection_loss, 0
                torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break

        baseline_model = build_physics_specialist("sawfm_period_sequence", args.width).to(device)
        baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
        baseline_prediction, target, calibration_identities = predict_paired(
            baseline_model, loaders["calibration"], device, mean, scale,
        )
        candidate_prediction, candidate_target, candidate_identities = predict_paired(
            model, loaders["calibration"], device, mean, scale,
        )
        if not np.array_equal(calibration_identities, candidate_identities) or not np.array_equal(target, candidate_target):
            raise RuntimeError("baseline and candidate calibration order differs")
        baseline_metrics = paired_metrics(baseline_prediction, target)
        candidate_metrics = paired_metrics(candidate_prediction, target)
        per_power_reduction = {
            str(power): float(1.0 - candidate_metrics["per_power_mae"][str(power)] / baseline_metrics["per_power_mae"][str(power)])
            for power in POWERS
        }
        receipt = {
            "seed": seed,
            "hypothesis": "paired nine-power supervision plus within-identity period consistency improves the SawFM specialist",
            "normalization": {"mean": mean, "scale": scale},
            "baseline_checkpoint": str(baseline_checkpoint.resolve()),
            "baseline_checkpoint_sha256": file_sha256(baseline_checkpoint),
            "candidate_checkpoint_sha256": file_sha256(run_dir / "best.pt"),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "relative_mae_reduction": relative_reduction(candidate_metrics, baseline_metrics),
            "per_power_relative_mae_reduction": per_power_reduction,
            "minimum_per_power_relative_mae_reduction": float(min(per_power_reduction.values())),
            "fit_identity_count": int(len(positive_roles["fit"])),
            "selection_identity_count": int(len(positive_roles["selection"])),
            "calibration_identity_count": int(len(positive_roles["calibration"])),
            "fit_identity_sha256": _hash_int64(identities[positive_roles["fit"]]),
            "selection_identity_sha256": _hash_int64(identities[positive_roles["selection"]]),
            "calibration_identity_sha256": _hash_int64(identities[positive_roles["calibration"]]),
            "epochs_ran": int(len(history)),
            "best_selection_loss": best_loss,
            "elapsed_seconds": float(time.perf_counter() - started),
            "powers_db": list(POWERS),
            "classification_loss_used": False,
            "oracle_positive_only": True,
            "p2a_test_opened": False,
            "locked_test_used": False,
        }
        (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
        (run_dir / "receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
        np.savez_compressed(
            run_dir / "calibration_predictions.npz",
            identity=calibration_identities,
            target=target,
            baseline_prediction=baseline_prediction,
            candidate_prediction=candidate_prediction,
            powers=np.asarray(POWERS, dtype=np.int16),
        )
        all_runs.append(receipt)
        print(json.dumps({
            "seed": seed,
            "relative_mae_reduction": receipt["relative_mae_reduction"],
            "minimum_per_power_relative_mae_reduction": receipt["minimum_per_power_relative_mae_reduction"],
        }), flush=True)
        for dataset in datasets.values():
            dataset.close()
        del model, baseline_model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reductions = np.asarray([row["relative_mae_reduction"] for row in all_runs], dtype=np.float64)
    minimum_power = min(float(row["minimum_per_power_relative_mae_reduction"]) for row in all_runs)
    gate = {
        "mean_relative_mae_reduction_at_least_0p02": bool(reductions.mean() >= 0.02),
        "three_of_three_positive": bool(np.all(reductions > 0)),
        "no_seed_power_degradation_below_minus_0p01": bool(minimum_power >= -0.01),
    }
    summary = {
        "experiment": "P2b identity-paired nine-power SawFM consistency fine-tuning",
        "protocol": vars(args),
        "development_data_root": str(development_root.resolve()),
        "development_identity_hash": _hash_int64(identities),
        "role_identity_overlap": {
            "fit_selection": int(np.intersect1d(positive_roles["fit"], positive_roles["selection"]).size),
            "fit_calibration": int(np.intersect1d(positive_roles["fit"], positive_roles["calibration"]).size),
            "selection_calibration": int(np.intersect1d(positive_roles["selection"], positive_roles["calibration"]).size),
        },
        "mean_relative_mae_reduction": float(reductions.mean()),
        "per_seed_relative_mae_reduction": reductions.tolist(),
        "minimum_seed_power_relative_mae_reduction": minimum_power,
        "promotion_gates": gate,
        "promotion_passed": bool(all(gate.values())),
        "runs": all_runs,
        "p2a_test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gate, "promotion_passed": summary["promotion_passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
