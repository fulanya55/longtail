"""Train and screen autocorrelation-consistent SawFM component separators."""

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
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator, multi_resolution_spectral_loss
from cj0724.dual_axis import read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import stratified_train_roles, transition_files
from cj0724.p2f_dedicated_sawfm_separator import POWERS as TRAIN_POWERS, SAW_CLASS_INDEX, SawSeparatorDataset, metric
from cj0724.p2j_full_parameter_attachment import MODEL_SEEDS
from cj0724.p2m_analytic_saw_period import POWERS as OOD_POWERS, REGIMES, analytic_period_estimate
from cj0724.train import set_seed


VARIANTS = {"A1_acf_0p10": 0.10, "A2_acf_0p50": 0.50}
PEARSON_LIMIT = {"ID_control": 0.55, "near_low": 0.50, "near_high": 0.15, "far_low": 0.85, "far_high": 0.25}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_complex_autocorrelation(iq: torch.Tensor, maximum_lag: int = 1600, stride: int = 4) -> torch.Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [batch, 2, time]")
    signal = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    length = signal.shape[-1]
    spectrum = torch.fft.fft(signal, n=2 * length)
    acf = torch.fft.ifft(spectrum * spectrum.conj()).real[:, : min(maximum_lag + 1, length)] / float(length)
    acf = acf / acf[:, :1].clamp_min(1.0e-8)
    return acf[:, 40::stride]


def objective(
    separator: ConditionalComplexUNetSeparator,
    iq: torch.Tensor,
    target_iq: torch.Tensor,
    acf_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, power_count = iq.shape[:2]
    flat_iq = iq.reshape(batch_size * power_count, *iq.shape[2:])
    flat_target = target_iq[:, None].expand(-1, power_count, -1, -1).reshape_as(flat_iq)
    estimate = separator.forward_components(flat_iq, None)["normalized_estimate"].float()
    cosine = 1.0 - F.cosine_similarity(estimate.flatten(1), flat_target.flatten(1), dim=1).mean()
    time_l1 = F.l1_loss(estimate, flat_target)
    spectral = multi_resolution_spectral_loss(estimate, flat_target)
    waveform = cosine + 0.2 * time_l1 + 0.25 * spectral
    acf_loss = F.smooth_l1_loss(
        normalized_complex_autocorrelation(estimate),
        normalized_complex_autocorrelation(flat_target),
    )
    total = waveform + float(acf_weight) * acf_loss
    return total, {
        "waveform": waveform.detach(),
        "cosine_loss": cosine.detach(),
        "time_l1": time_l1.detach(),
        "spectral": spectral.detach(),
        "acf_loss": acf_loss.detach(),
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_variant(
    files: dict[int, Path],
    roles: dict[str, np.ndarray],
    seed: int,
    acf_weight: float,
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
                total, parts = objective(separator, batch["iq"].to(device), batch["target_iq"].to(device), acf_weight)
                total.backward()
                torch.nn.utils.clip_grad_norm_(separator.parameters(), args.gradient_clip_norm)
                optimizer.step()
                train_rows.append({"total": float(total.detach()), **{key: float(value) for key, value in parts.items()}})
            separator.eval()
            selection_rows = []
            with torch.inference_mode():
                for batch in loaders["selection"]:
                    total, parts = objective(separator, batch["iq"].to(device), batch["target_iq"].to(device), acf_weight)
                    selection_rows.append({"total": float(total), **{key: float(value) for key, value in parts.items()}})
            train_epoch, selection_epoch = mean_rows(train_rows), mean_rows(selection_rows)
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


@torch.inference_mode()
def waveform_cosine(separator: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    values = []
    separator.eval()
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat = iq.reshape(batch_size * power_count, *iq.shape[2:])
        target = batch["target_iq"].to(device)[:, None].expand(-1, power_count, -1, -1).reshape_as(flat)
        estimate = separator.forward_components(flat, None)["normalized_estimate"].float()
        values.append(F.cosine_similarity(estimate.flatten(1), target.flatten(1), dim=1).cpu().numpy())
    return float(np.mean(np.concatenate(values)))


@torch.inference_mode()
def ood_predictions(separator: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    targets, predictions = [], []
    separator.eval()
    for batch in loader:
        iq = batch["iq"].to(device)
        batch_size, power_count = iq.shape[:2]
        flat = iq.reshape(batch_size * power_count, *iq.shape[2:])
        estimate = separator.forward_components(flat, None)["normalized_estimate"].float()
        predictions.append(analytic_period_estimate(estimate.cpu().numpy())["prediction"])
        targets.append(np.repeat(batch["period"].numpy(), power_count))
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", default="generated_data/p2f_dedicated_sawfm_separator_seed1700000000_rechunked")
    parser.add_argument("--ood-root-template", default="generated_data/p2m_analytic_saw_period_seed2700000000_{regime}_rechunked")
    parser.add_argument("--baseline-separator-root", default="outputs/p2f_dedicated_sawfm_separator_development_v5_20260801")
    parser.add_argument("--baseline-ood-root", default="outputs/p2m_analytic_saw_period_physics_20260801")
    parser.add_argument("--preregister", default="configs/p2s_acf_consistent_separator_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2s_acf_consistent_separator_development_20260801")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    training_root = project / args.training_root
    baseline_root = project / args.baseline_separator_root
    baseline_ood_root = project / args.baseline_ood_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    train_files = transition_files(training_root, "train", TRAIN_POWERS)
    identities, labels = read_identity_metadata(train_files[TRAIN_POWERS[0]])
    roles_all = stratified_train_roles(labels, 20260905)
    with h5py.File(train_files[TRAIN_POWERS[0]], "r") as handle:
        mask = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
    roles = {name: indices[(labels[indices, SAW_CLASS_INDEX] > 0) & mask[14, indices]] for name, indices in roles_all.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ood_datasets, ood_loaders = {}, {}
    for regime in REGIMES:
        files = transition_files(project / args.ood_root_template.format(regime=regime), "train", OOD_POWERS)
        _, ood_labels = read_identity_metadata(files[OOD_POWERS[0]])
        selected = np.flatnonzero(ood_labels[:, SAW_CLASS_INDEX] > 0)
        ood_datasets[regime] = SawSeparatorDataset(files, selected)
        ood_loaders[regime] = DataLoader(ood_datasets[regime], batch_size=16, shuffle=False, pin_memory=device.type == "cuda")

    calibration_dataset = SawSeparatorDataset(train_files, roles["calibration"])
    calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    runs = []
    try:
        for seed in MODEL_SEEDS:
            baseline_model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
            baseline_checkpoint = baseline_root / f"seed_{seed}" / "S1_waveform_only" / "best.pt"
            baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
            baseline_cosine = waveform_cosine(baseline_model, calibration_loader, device)
            del baseline_model
            for variant, weight in VARIANTS.items():
                run_dir = output_root / f"seed_{seed}" / variant
                separator, history, elapsed = train_variant(train_files, roles, seed, weight, run_dir, device, args)
                candidate_cosine = waveform_cosine(separator, calibration_loader, device)
                regime_rows = {}
                for regime in REGIMES:
                    prediction, target = ood_predictions(separator, ood_loaders[regime], device)
                    with np.load(baseline_ood_root / regime / f"seed_{seed}" / "predictions.npz") as saved:
                        baseline_prediction = saved["analytic_separated"]
                        learned_prediction = saved["learned_separated"]
                    regime_rows[regime] = {
                        "candidate": metric(prediction, target),
                        "baseline": metric(baseline_prediction, target),
                        "learned": metric(learned_prediction, target),
                    }
                    np.savez_compressed(run_dir / f"ood_{regime}.npz", target=target, prediction=prediction)
                row = {
                    "seed": seed,
                    "variant": variant,
                    "acf_weight": weight,
                    "baseline_waveform_cosine": baseline_cosine,
                    "candidate_waveform_cosine": candidate_cosine,
                    "waveform_cosine_delta": candidate_cosine - baseline_cosine,
                    "epochs_ran": len(history),
                    "train_seconds": elapsed,
                    "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                    "regimes": regime_rows,
                }
                (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
                (run_dir / "receipt.json").write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")
                runs.append(row)
                print(json.dumps({
                    "seed": seed,
                    "variant": variant,
                    "cosine_delta": row["waveform_cosine_delta"],
                    "near_high_pearson": regime_rows["near_high"]["candidate"]["pearson"],
                    "far_high_pearson": regime_rows["far_high"]["candidate"]["pearson"],
                }), flush=True)
                del separator
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        calibration_dataset.close()
        for dataset in ood_datasets.values():
            dataset.close()

    summaries, passing = {}, []
    for variant in VARIANTS:
        subset = [row for row in runs if row["variant"] == variant]
        per_regime = {}
        for regime in REGIMES:
            per_regime[regime] = {
                "candidate_mae_us_mean": float(np.mean([row["regimes"][regime]["candidate"]["mae_us"] for row in subset])),
                "baseline_mae_us_mean": float(np.mean([row["regimes"][regime]["baseline"]["mae_us"] for row in subset])),
                "candidate_pearson_mean": float(np.mean([row["regimes"][regime]["candidate"]["pearson"] for row in subset])),
                "baseline_pearson_mean": float(np.mean([row["regimes"][regime]["baseline"]["pearson"] for row in subset])),
            }
        gates = {
            "pearson_limits": bool(all(per_regime[r]["candidate_pearson_mean"] >= PEARSON_LIMIT[r] for r in REGIMES)),
            "pearson_nonworse_each_regime": bool(all(per_regime[r]["candidate_pearson_mean"] >= per_regime[r]["baseline_pearson_mean"] for r in REGIMES)),
            "mae_nonworse_each_regime": bool(all(per_regime[r]["candidate_mae_us_mean"] <= per_regime[r]["baseline_mae_us_mean"] for r in REGIMES)),
            "all_15_improve_over_learned": bool(all(
                row["regimes"][r]["candidate"]["mae"] < row["regimes"][r]["learned"]["mae"]
                for row in subset for r in REGIMES
            )),
            "hard_regimes_three_seed_direction": bool(all(
                row["regimes"][r]["candidate"]["pearson"] > row["regimes"][r]["baseline"]["pearson"]
                for row in subset for r in ("near_high", "far_high")
            )),
            "waveform_cosine_drop_at_most_0p02": bool(all(row["waveform_cosine_delta"] >= -0.02 for row in subset)),
        }
        deltas = [per_regime[r]["candidate_pearson_mean"] - per_regime[r]["baseline_pearson_mean"] for r in REGIMES]
        summaries[variant] = {
            "per_regime": per_regime,
            "gates": gates,
            "passed": bool(all(gates.values())),
            "minimum_pearson_delta": float(min(deltas)),
            "mean_pearson_delta": float(np.mean(deltas)),
            "mean_candidate_mae_us": float(np.mean([per_regime[r]["candidate_mae_us_mean"] for r in REGIMES])),
        }
        if summaries[variant]["passed"]:
            passing.append(variant)
    selected = None
    if passing:
        selected = max(passing, key=lambda name: (
            summaries[name]["minimum_pearson_delta"],
            summaries[name]["mean_pearson_delta"],
            -summaries[name]["mean_candidate_mae_us"],
        ))
    summary_path = output_root / "summary.json"
    summary = {
        "experiment": "P2s autocorrelation-consistent SawFM separator development",
        "preregister_sha256": sha256_file(preregister),
        "role_counts": {name: int(len(value)) for name, value in roles.items()},
        "variant_summary": summaries,
        "selected_variant": selected,
        "development_passed": selected is not None,
        "runs": runs,
        "confirmation_identities_used": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if selected:
        selected_rows = [row for row in runs if row["variant"] == selected]
        (output_root / "method_freeze.json").write_text(json.dumps({
            "method": "autocorrelation-consistent SawFM separator",
            "selected_variant": selected,
            "acf_weight": VARIANTS[selected],
            "checkpoint_sha256": {str(row["seed"]): row["checkpoint_sha256"] for row in selected_rows},
            "summary_sha256": sha256_file(summary_path),
            "preregister_sha256": sha256_file(preregister),
            "forbidden_confirmation_seeds": [2800000000, 2900000000],
            "locked_test_used": False,
        }, indent=2), encoding="utf-8")
    print(json.dumps({
        "development_passed": summary["development_passed"],
        "selected_variant": selected,
        "variant_summary": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
