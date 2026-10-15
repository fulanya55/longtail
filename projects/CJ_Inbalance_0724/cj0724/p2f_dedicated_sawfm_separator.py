"""P2f train a dedicated SawFM component separator on fresh development data."""

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

from cj0724.conditional_unet_separator_experiment import (
    ConditionalComplexUNetSeparator,
    multi_resolution_spectral_loss,
)
from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles, transition_files
from cj0724.p1k_joint_level3_system import SAW_PARAMETER_COLUMN, _hash_int64
from cj0724.p2e_sawfm_component_oracle import (
    SAW_BANDWIDTH_COLUMN,
    SAW_DELAY_COLUMN,
    SAW_PERIOD_COLUMN,
    reconstruct_sawfm_component,
)
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.train import set_seed


POWERS = (35, 40, 45, 50)
SEEDS = (2041, 2042, 2043)
SAW_CLASS_INDEX = 1
VARIANTS = {"S1_waveform_only": 0.0, "S2_waveform_plus_period": 0.25}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SawSeparatorDataset(Dataset):
    """Return all power views and the exact normalized SawFM component target."""

    def __init__(self, files: dict[int, Path], indices: np.ndarray) -> None:
        self.powers = tuple(sorted(files))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.datasets = {power: V73IQDataset(files[power], self.indices, [SAW_PARAMETER_COLUMN]) for power in self.powers}
        with h5py.File(files[self.powers[0]], "r") as handle:
            values = np.asarray(handle["parameter_values"], dtype=np.float64)
        self.period = values[SAW_PERIOD_COLUMN, self.indices].astype(np.float32)
        self.target_iq = reconstruct_sawfm_component(
            values[SAW_BANDWIDTH_COLUMN, self.indices],
            values[SAW_PERIOD_COLUMN, self.indices],
            values[SAW_DELAY_COLUMN, self.indices],
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        views = [self.datasets[power][position] for power in self.powers]
        sample_indices = [int(view["sample_index"]) for view in views]
        if len(set(sample_indices)) != 1:
            raise RuntimeError("P2f power-view sample indices differ")
        return {
            "iq": torch.stack([view["iq"] for view in views]),
            "target_iq": torch.from_numpy(self.target_iq[position]),
            "period": torch.tensor(self.period[position], dtype=torch.float32),
            "sample_index": torch.tensor(sample_indices[0], dtype=torch.int64),
        }

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()


def separation_objective(
    separator: ConditionalComplexUNetSeparator,
    period_expert: torch.nn.Module,
    iq: torch.Tensor,
    target_iq: torch.Tensor,
    period_raw: torch.Tensor,
    period_mean: float,
    period_scale: float,
    period_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    batch_size, power_count = iq.shape[:2]
    flat_iq = iq.reshape(batch_size * power_count, *iq.shape[2:])
    flat_target = target_iq[:, None].expand(-1, power_count, -1, -1).reshape_as(flat_iq)
    components = separator.forward_components(flat_iq, None)
    estimate = components["normalized_estimate"].float()
    cosine = 1.0 - F.cosine_similarity(estimate.flatten(1), flat_target.flatten(1), dim=1).mean()
    time_l1 = F.l1_loss(estimate, flat_target)
    spectral = multi_resolution_spectral_loss(estimate, flat_target)
    waveform = cosine + 0.2 * time_l1 + 0.25 * spectral
    period_prediction = period_expert(estimate)["prediction"].float()
    normalized_target = ((period_raw.float() - float(period_mean)) / float(period_scale))[:, None]
    normalized_target = normalized_target.expand(-1, power_count).reshape(-1)
    period_loss = F.smooth_l1_loss(period_prediction, normalized_target)
    total = waveform + float(period_weight) * period_loss
    return total, {
        "waveform": waveform.detach(), "cosine_loss": cosine.detach(),
        "time_l1": time_l1.detach(), "spectral": spectral.detach(),
        "period_loss": period_loss.detach(),
    }, estimate


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(prediction) - np.asarray(target))
    correlation = 0.0
    if np.std(prediction) > 0.0 and np.std(target) > 0.0:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {"mae": float(error.mean()), "mae_us": float(error.mean() * 1.0e6), "pearson": correlation}


@torch.inference_mode()
def evaluate_separator(
    separator: ConditionalComplexUNetSeparator,
    period_expert: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    scale: float,
) -> dict[str, np.ndarray | float]:
    separator.eval()
    period_expert.eval()
    raw_prediction, separated_prediction, target, indices, cosines = [], [], [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat = iq.reshape(batch_size * power_count, *iq.shape[2:])
        target_iq = batch["target_iq"].to(device)[:, None].expand(-1, power_count, -1, -1).reshape_as(flat)
        estimate = separator.forward_components(flat, None)["normalized_estimate"].float()
        raw = period_expert(flat)["prediction"] * float(scale) + float(mean)
        separated = period_expert(estimate)["prediction"] * float(scale) + float(mean)
        raw_prediction.append(raw.cpu().numpy())
        separated_prediction.append(separated.cpu().numpy())
        target.append(np.repeat(batch["period"].numpy(), power_count))
        indices.append(np.repeat(batch["sample_index"].numpy(), power_count))
        cosines.append(F.cosine_similarity(estimate.flatten(1), target_iq.flatten(1), dim=1).cpu().numpy())
    return {
        "raw_prediction": np.concatenate(raw_prediction),
        "separated_prediction": np.concatenate(separated_prediction),
        "target": np.concatenate(target),
        "sample_index": np.concatenate(indices),
        "waveform_cosine": float(np.mean(np.concatenate(cosines))),
    }


def train_variant(
    name: str,
    period_weight: float,
    files: dict[int, Path],
    roles: dict[str, np.ndarray],
    period_expert: torch.nn.Module,
    mean: float,
    scale: float,
    seed: int,
    run_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[ConditionalComplexUNetSeparator, list[dict[str, object]], float]:
    set_seed(seed)
    separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
    optimizer = torch.optim.AdamW(separator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    datasets = {role: SawSeparatorDataset(files, roles[role]) for role in ("fit", "selection")}
    loaders = {
        "fit": DataLoader(datasets["fit"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda"),
        "selection": DataLoader(datasets["selection"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
    }
    run_dir.mkdir(parents=True)
    best_loss, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    try:
        for epoch in range(1, args.epochs + 1):
            separator.train()
            train_rows = []
            for batch in loaders["fit"]:
                optimizer.zero_grad(set_to_none=True)
                total, parts, _ = separation_objective(
                    separator, period_expert, batch["iq"].to(device), batch["target_iq"].to(device),
                    batch["period"].to(device), mean, scale, period_weight,
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(separator.parameters(), args.gradient_clip_norm)
                optimizer.step()
                train_rows.append({"total": float(total.detach()), **{key: float(value) for key, value in parts.items()}})
            separator.eval()
            selection_rows = []
            with torch.inference_mode():
                for batch in loaders["selection"]:
                    total, parts, _ = separation_objective(
                        separator, period_expert, batch["iq"].to(device), batch["target_iq"].to(device),
                        batch["period"].to(device), mean, scale, period_weight,
                    )
                    selection_rows.append({"total": float(total), **{key: float(value) for key, value in parts.items()}})
            train_epoch, selection_epoch = _mean_rows(train_rows), _mean_rows(selection_rows)
            history.append({"epoch": epoch, "train": train_epoch, "selection": selection_epoch})
            if selection_epoch["total"] < best_loss:
                best_loss, stale = selection_epoch["total"], 0
                torch.save(separator.state_dict(), run_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        separator.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    finally:
        for dataset in datasets.values():
            dataset.close()
    return separator, history, float(time.perf_counter() - started)


def identity_audit(files: dict[int, Path], excluded_files: list[Path]) -> dict[str, object]:
    alignment = audit_power_alignment(files)
    identities, _ = read_identity_metadata(files[POWERS[0]])
    exclusions = []
    for path in excluded_files:
        prior, _ = read_identity_metadata(path)
        exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
    if any(row["overlap_count"] != 0 for row in exclusions):
        raise RuntimeError("P2f identities overlap a prior split")
    return {"alignment": alignment, "identity_count": int(len(identities)), "identity_sha256": _hash_int64(identities), "excluded_prior_splits": exclusions, "test_generated": False, "test_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", default="generated_data/p2f_dedicated_sawfm_separator_seed1700000000_rechunked")
    parser.add_argument("--reference-freeze-root", default="outputs/p2a_nine_power_generalization_20260731")
    parser.add_argument("--specialist-root", default="outputs/physics_parameter_specialists")
    parser.add_argument("--preregister", default="configs/p2f_dedicated_sawfm_separator_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / args.development_root
    reference_root = project / args.reference_freeze_root
    specialist_root = project / args.specialist_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    files = transition_files(root, "train", POWERS)
    audit = identity_audit(files, [Path(path) for path in args.excluded_identity_files])
    receiver_identities, labels = read_identity_metadata(files[POWERS[0]])
    roles_all = stratified_train_roles(labels, 20260905)
    with h5py.File(files[POWERS[0]], "r") as handle:
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    roles = {name: indices[(labels[indices, SAW_CLASS_INDEX] > 0) & mask[SAW_PERIOD_COLUMN, indices]] for name, indices in roles_all.items()}
    role_overlap = {
        "fit_selection": int(np.intersect1d(roles["fit"], roles["selection"]).size),
        "fit_calibration": int(np.intersect1d(roles["fit"], roles["calibration"]).size),
        "selection_calibration": int(np.intersect1d(roles["selection"], roles["calibration"]).size),
    }
    if any(role_overlap.values()):
        raise RuntimeError("P2f roles overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in SEEDS:
        freeze = json.loads((reference_root / f"seed_{seed}" / "seed_freeze.json").read_text(encoding="utf-8"))
        mean, scale = float(freeze["saw_mean"]), float(freeze["saw_scale"])
        expert_checkpoint = specialist_root / "sawfm_period_sequence" / f"seed_{seed}" / "best.pt"
        period_expert = build_physics_specialist("sawfm_period_sequence", width=16).to(device)
        period_expert.load_state_dict(torch.load(expert_checkpoint, map_location=device, weights_only=True))
        period_expert.eval()
        for parameter in period_expert.parameters():
            parameter.requires_grad_(False)
        calibration_dataset = SawSeparatorDataset(files, roles["calibration"])
        calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        variant_rows = {}
        for variant, period_weight in VARIANTS.items():
            run_dir = output_root / f"seed_{seed}" / variant
            separator, history, elapsed = train_variant(
                variant, period_weight, files, roles, period_expert, mean, scale,
                seed, run_dir, device, args,
            )
            evaluation = evaluate_separator(separator, period_expert, calibration_loader, device, mean, scale)
            raw_metrics = metric(evaluation["raw_prediction"], evaluation["target"])
            separated_metrics = metric(evaluation["separated_prediction"], evaluation["target"])
            row = {
                "variant": variant, "period_weight": period_weight,
                "raw": raw_metrics, "separated": separated_metrics,
                "period_reduction_vs_raw": float(1.0 - separated_metrics["mae"] / raw_metrics["mae"]),
                "waveform_cosine": evaluation["waveform_cosine"],
                "epochs_ran": len(history), "train_seconds": elapsed,
                "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            }
            (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
            (run_dir / "receipt.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
            np.savez_compressed(
                run_dir / "calibration_predictions.npz",
                sample_index=evaluation["sample_index"], target=evaluation["target"],
                raw_prediction=evaluation["raw_prediction"], separated_prediction=evaluation["separated_prediction"],
            )
            variant_rows[variant] = row
            del separator
            if device.type == "cuda":
                torch.cuda.empty_cache()
        calibration_dataset.close()
        s1 = variant_rows["S1_waveform_only"]
        s2 = variant_rows["S2_waveform_plus_period"]
        run = {
            "seed": seed,
            "variants": variant_rows,
            "S2_reduction_vs_S1": float(1.0 - s2["separated"]["mae"] / s1["separated"]["mae"]),
            "expert_checkpoint_sha256": sha256_file(expert_checkpoint),
        }
        (output_root / f"seed_{seed}" / "receipt.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
        runs.append(run)
        del period_expert
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"seed": seed, "S1_reduction": s1["period_reduction_vs_raw"], "S2_reduction": s2["period_reduction_vs_raw"], "S2_cosine": s2["waveform_cosine"], "S2_vs_S1": run["S2_reduction_vs_S1"]}), flush=True)

    gates = {
        "each_seed_S2_reduction_vs_raw_at_least_0p20": bool(all(run["variants"]["S2_waveform_plus_period"]["period_reduction_vs_raw"] >= 0.20 for run in runs)),
        "each_seed_S2_pearson_at_least_0p60": bool(all(run["variants"]["S2_waveform_plus_period"]["separated"]["pearson"] >= 0.60 for run in runs)),
        "each_seed_S2_waveform_cosine_at_least_0p50": bool(all(run["variants"]["S2_waveform_plus_period"]["waveform_cosine"] >= 0.50 for run in runs)),
        "each_seed_S2_vs_S1_no_worse_than_minus_0p01": bool(all(run["S2_reduction_vs_S1"] >= -0.01 for run in runs)),
        "mean_S2_reduction_vs_raw_at_least_0p30": bool(np.mean([run["variants"]["S2_waveform_plus_period"]["period_reduction_vs_raw"] for run in runs]) >= 0.30),
    }
    passed = bool(all(gates.values()))
    s1_pass = bool(all(run["variants"]["S1_waveform_only"]["period_reduction_vs_raw"] >= 0.20 and run["variants"]["S1_waveform_only"]["separated"]["pearson"] >= 0.60 for run in runs))
    if passed:
        interpretation = "Dedicated separator plus frozen period loss passed; create a fresh two-seed confirmation before attachment."
    elif s1_pass:
        interpretation = "Waveform supervision passed but downstream period loss caused negative transfer; retain S1 only."
    else:
        interpretation = "Dedicated U-Net did not transfer oracle sufficiency; stop architecture scaling and inspect supervision alignment/loss."
    summary = {
        "experiment": "P2f dedicated SawFM separator development-v5",
        "protocol": vars(args), "preregister_sha256": sha256_file(preregister),
        "identity_audit": audit,
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "role_hashes": {name: _hash_int64(receiver_identities[indices]) for name, indices in roles.items()},
        "role_overlap": role_overlap,
        "promotion_gates": gates, "promotion_passed": passed,
        "S1_ablation_passed": s1_pass,
        "mean_period_reduction_vs_raw": {variant: float(np.mean([run["variants"][variant]["period_reduction_vs_raw"] for run in runs])) for variant in VARIANTS},
        "interpretation": interpretation, "runs": runs,
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "identity_audit.json").write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"promotion_gates": gates, "promotion_passed": passed, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
