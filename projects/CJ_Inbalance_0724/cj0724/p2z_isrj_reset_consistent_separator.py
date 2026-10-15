"""Train and evaluate a dedicated ISRJ separator with phase-reset consistency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator, multi_resolution_spectral_loss
from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import (
    ARRIVAL_COLUMN, ISRJ_CLASS_INDEX, POWERS, REGIMES, REPLICAS_COLUMN, WIDTH_COLUMN,
    metrics, reconstruct_isrj_component, sha256_file,
)
from cj0724.p2x_isrj_phase_reset_ood import estimate_width_phase_resets
from cj0724.p2y_isrj_phase_reset_isolation import waveform_cosine
from cj0724.train import set_seed


MODEL_SEEDS = (2041, 2042, 2043)
VARIANTS = {"S0_waveform_only": 0.0, "S1_waveform_plus_reset": 0.5}


class ISRJSeparatorDataset(Dataset):
    def __init__(self, files: dict[int, Path], indices: np.ndarray) -> None:
        self.powers = tuple(sorted(files))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.datasets = {power: V73IQDataset(files[power], self.indices, [WIDTH_COLUMN]) for power in self.powers}
        with h5py.File(files[self.powers[0]], "r") as handle:
            values = np.asarray(handle["parameter_values"], dtype=np.float64)
        self.width = values[WIDTH_COLUMN, self.indices].astype(np.float32)
        self.target_iq = reconstruct_isrj_component(
            values[WIDTH_COLUMN, self.indices], values[REPLICAS_COLUMN, self.indices], values[ARRIVAL_COLUMN, self.indices]
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        views = [self.datasets[power][position] for power in self.powers]
        sample_indices = [int(view["sample_index"]) for view in views]
        if len(set(sample_indices)) != 1:
            raise RuntimeError("P2z power-view sample indices differ")
        return {
            "iq": torch.stack([view["iq"] for view in views]),
            "target_iq": torch.from_numpy(self.target_iq[position]),
            "width": torch.tensor(self.width[position], dtype=torch.float32),
            "sample_index": torch.tensor(sample_indices[0], dtype=torch.int64),
        }

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()


def phase_curvature_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    estimate_complex = torch.complex(estimate[:, 0].float(), estimate[:, 1].float())
    target_complex = torch.complex(target[:, 0].float(), target[:, 1].float())

    def curvature(signal: torch.Tensor) -> torch.Tensor:
        increment = signal[:, 1:] * signal[:, :-1].conj()
        increment = increment / increment.abs().clamp_min(1.0e-6)
        return increment[:, 1:] * increment[:, :-1].conj()

    estimate_curvature = curvature(estimate_complex)
    target_curvature = curvature(target_complex)
    target_amplitude = target_complex.abs()
    threshold = 0.001 * target_amplitude.amax(dim=1, keepdim=True).clamp_min(1.0e-6)
    active = (target_amplitude[:, :-2] >= threshold) & (target_amplitude[:, 1:-1] >= threshold) & (target_amplitude[:, 2:] >= threshold)
    reset_strength = (1.0 - target_curvature.real).clamp_min(0.0)
    weight = active.float() * (1.0 + 4.0 * reset_strength)
    squared = (estimate_curvature.real - target_curvature.real).square() + (estimate_curvature.imag - target_curvature.imag).square()
    return (squared * weight).sum() / weight.sum().clamp_min(1.0)


def separation_objective(
    separator: ConditionalComplexUNetSeparator,
    iq: torch.Tensor,
    target_iq: torch.Tensor,
    reset_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, power_count = iq.shape[:2]
    flat_iq = iq.reshape(batch_size * power_count, *iq.shape[2:])
    flat_target = target_iq[:, None].expand(-1, power_count, -1, -1).reshape_as(flat_iq)
    estimate = separator.forward_components(flat_iq, None)["normalized_estimate"].float()
    cosine = 1.0 - F.cosine_similarity(estimate.flatten(1), flat_target.flatten(1), dim=1).mean()
    time_l1 = F.l1_loss(estimate, flat_target)
    spectral = multi_resolution_spectral_loss(estimate, flat_target)
    waveform = cosine + 0.2 * time_l1 + 0.25 * spectral
    reset = phase_curvature_loss(estimate, flat_target)
    total = waveform + float(reset_weight) * reset
    return total, {"waveform": waveform.detach(), "cosine_loss": cosine.detach(), "time_l1": time_l1.detach(), "spectral": spectral.detach(), "reset": reset.detach()}


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def stratified_roles_10(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Split each of the 63 ten-identity combinations as 6/2/2."""
    labels = np.asarray(labels, dtype=np.uint8)
    keys = np.packbits(labels, axis=1, bitorder="little")[:, 0]
    rng = np.random.default_rng(seed)
    roles = {"fit": [], "selection": [], "calibration": []}
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key)
        if len(indices) != 10:
            raise ValueError(f"Expected 10 identities per combination, got {len(indices)}")
        indices = rng.permutation(indices)
        roles["fit"].extend(indices[:6].tolist())
        roles["selection"].extend(indices[6:8].tolist())
        roles["calibration"].extend(indices[8:].tolist())
    return {name: np.sort(np.asarray(indices, dtype=np.int64)) for name, indices in roles.items()}


def train_variant(
    files: dict[int, Path], roles: dict[str, np.ndarray], seed: int, reset_weight: float,
    run_dir: Path, device: torch.device, args: argparse.Namespace,
) -> tuple[ConditionalComplexUNetSeparator, list[dict[str, object]], float]:
    set_seed(seed)
    separator = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
    optimizer = torch.optim.AdamW(separator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    datasets = {role: ISRJSeparatorDataset(files, roles[role]) for role in ("fit", "selection")}
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
                total, parts = separation_objective(separator, batch["iq"].to(device), batch["target_iq"].to(device), reset_weight)
                total.backward()
                torch.nn.utils.clip_grad_norm_(separator.parameters(), args.gradient_clip_norm)
                optimizer.step()
                train_rows.append({"total": float(total.detach()), **{key: float(value) for key, value in parts.items()}})
            separator.eval()
            selection_rows = []
            with torch.inference_mode():
                for batch in loaders["selection"]:
                    total, parts = separation_objective(separator, batch["iq"].to(device), batch["target_iq"].to(device), reset_weight)
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


@torch.inference_mode()
def evaluate_separator(
    separator: ConditionalComplexUNetSeparator, files: dict[int, Path], indices: np.ndarray,
    labels: np.ndarray, device: torch.device, batch_size: int,
) -> dict[str, np.ndarray | float]:
    dataset = ISRJSeparatorDataset(files, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    arrays = {"identity_position": [], "power": [], "combination_size": [], "target": [], "raw_prediction": [], "separated_prediction": [], "exact_prediction": [], "waveform_cosine": []}
    try:
        separator.eval()
        for batch in loader:
            iq = batch["iq"].to(device)
            batch_count, power_count = iq.shape[:2]
            flat = iq.reshape(batch_count * power_count, *iq.shape[2:])
            target_iq = batch["target_iq"].to(device)
            flat_target = target_iq[:, None].expand(-1, power_count, -1, -1).reshape_as(flat)
            estimate = separator.forward_components(flat, None)["normalized_estimate"].float()
            raw_prediction = estimate_width_phase_resets(flat.cpu().numpy())["prediction"]
            separated_prediction = estimate_width_phase_resets(estimate.cpu().numpy())["prediction"]
            exact_prediction = estimate_width_phase_resets(flat_target.cpu().numpy())["prediction"]
            arrays["identity_position"].append(np.repeat(batch["sample_index"].numpy(), power_count))
            arrays["power"].append(np.tile(np.asarray(sorted(files), dtype=np.int16), batch_count))
            arrays["combination_size"].append(np.repeat(labels[batch["sample_index"].numpy()].sum(axis=1).astype(np.int8), power_count))
            arrays["target"].append(np.repeat(batch["width"].numpy(), power_count))
            arrays["raw_prediction"].append(raw_prediction)
            arrays["separated_prediction"].append(separated_prediction)
            arrays["exact_prediction"].append(exact_prediction)
            arrays["waveform_cosine"].append(waveform_cosine(estimate.cpu().numpy(), flat_target.cpu().numpy()))
    finally:
        dataset.close()
    return {name: np.concatenate(parts) for name, parts in arrays.items()}


def variant_metrics(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    compound = arrays["combination_size"] >= 2
    output = {}
    for name, key in (("exact", "exact_prediction"), ("raw", "raw_prediction"), ("separated", "separated_prediction")):
        output[name] = {
            "overall": metrics(arrays[key], arrays["target"]),
            "compound": metrics(arrays[key][compound], arrays["target"][compound]),
            "by_power": {str(power): metrics(arrays[key][arrays["power"] == power], arrays["target"][arrays["power"] == power]) for power in POWERS},
        }
    output["waveform_cosine_mean"] = float(arrays["waveform_cosine"].mean())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="generated_data/p2z_isrj_reset_separator_train_seed3700000000_ID_control_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p2z_isrj_reset_separator_dev_seed3800000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p2z_isrj_reset_consistent_separator_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2z_isrj_reset_consistent_separator_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    preregister = project / args.preregister
    train_files = transition_files(project / args.train_root, "train", POWERS)
    train_alignment = audit_power_alignment(train_files)
    train_identities, train_labels = read_identity_metadata(train_files[POWERS[0]])
    with h5py.File(train_files[POWERS[0]], "r") as handle:
        train_mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
    roles_all = stratified_roles_10(train_labels, 20260801)
    roles = {name: indices[(train_labels[indices, ISRJ_CLASS_INDEX] > 0) & train_mask[indices]] for name, indices in roles_all.items()}
    if np.intersect1d(roles["fit"], roles["selection"]).size:
        raise RuntimeError("P2z training roles overlap")

    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded_paths = [Path(protocol["data_file"])]
    excluded_paths.extend(Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files)
    train_exclusions = []
    for path in excluded_paths:
        prior, _ = read_identity_metadata(path)
        train_exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(train_identities, prior).size)})
    if any(row["overlap_count"] for row in train_exclusions):
        raise RuntimeError("P2z training identity overlap")

    development = {}
    development_identity_reference = None
    for regime in REGIMES:
        files = transition_files(project / args.development_root_template.format(regime=regime), "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, ISRJ_CLASS_INDEX] > 0) & mask)
        if len(selected) != 320:
            raise RuntimeError(f"P2z development count mismatch in {regime}")
        if development_identity_reference is None:
            development_identity_reference = identities
        elif not np.array_equal(development_identity_reference, identities):
            raise RuntimeError("P2z development regimes are not identity paired")
        if np.intersect1d(train_identities, identities).size:
            raise RuntimeError("P2z train/development identity overlap")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P2z development identity overlap")
        development[regime] = {"files": files, "identities": identities, "labels": labels, "selected": selected, "alignment": alignment, "exclusions": exclusions}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in MODEL_SEEDS:
        variants = {}
        for variant, reset_weight in VARIANTS.items():
            run_dir = output_root / f"seed_{seed}" / variant
            separator, history, elapsed = train_variant(train_files, roles, seed, reset_weight, run_dir, device, args)
            regime_rows = {}
            for regime, record in development.items():
                arrays = evaluate_separator(separator, record["files"], record["selected"], record["labels"], device, args.batch_size)
                result = variant_metrics(arrays)
                regime_dir = run_dir / regime
                regime_dir.mkdir()
                np.savez_compressed(regime_dir / "predictions.npz", **arrays)
                (regime_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
                regime_rows[regime] = result
            receipt = {"variant": variant, "reset_weight": reset_weight, "epochs_ran": len(history), "train_seconds": elapsed, "checkpoint_sha256": sha256_file(run_dir / "best.pt"), "regimes": regime_rows}
            (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
            (run_dir / "receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
            variants[variant] = receipt
            del separator
            if device.type == "cuda":
                torch.cuda.empty_cache()
        s0_mae = np.mean([variants["S0_waveform_only"]["regimes"][regime]["separated"]["overall"]["mae_s"] for regime in REGIMES])
        s1_mae = np.mean([variants["S1_waveform_plus_reset"]["regimes"][regime]["separated"]["overall"]["mae_s"] for regime in REGIMES])
        run = {"seed": seed, "variants": variants, "S1_mean_mae_reduction_vs_S0": float(1.0 - s1_mae / max(s0_mae, 1.0e-12))}
        runs.append(run)
        print(json.dumps({"seed": seed, "S1_mean_mae_reduction_vs_S0": run["S1_mean_mae_reduction_vs_S0"], "S1": {regime: variants["S1_waveform_plus_reset"]["regimes"][regime]["separated"]["overall"] for regime in REGIMES}}), flush=True)

    s1_rows = [(run, regime, run["variants"]["S1_waveform_plus_reset"]["regimes"][regime]) for run in runs for regime in REGIMES]
    exact_rows = [runs[0]["variants"]["S1_waveform_plus_reset"]["regimes"][regime]["exact"]["overall"] for regime in REGIMES]
    gates = {
        "identity_overlap_zero": bool(all(row["overlap_count"] == 0 for row in train_exclusions) and all(ex["overlap_count"] == 0 for record in development.values() for ex in record["exclusions"])),
        "exact_each_regime_reduction_vs_constant_at_least_0p10": bool(all(row["mae_reduction_vs_constant"] >= 0.10 for row in exact_rows)),
        "exact_each_regime_pearson_at_least_0p30": bool(all(row["pearson"] >= 0.30 for row in exact_rows)),
        "S1_all_15_reduction_vs_constant_at_least_0p10": bool(all(row["separated"]["overall"]["mae_reduction_vs_constant"] >= 0.10 for _, _, row in s1_rows)),
        "S1_all_15_pearson_at_least_0p30": bool(all(row["separated"]["overall"]["pearson"] >= 0.30 for _, _, row in s1_rows)),
        "S1_compound_all_15_reduction_vs_constant_at_least_0p10": bool(all(row["separated"]["compound"]["mae_reduction_vs_constant"] >= 0.10 for _, _, row in s1_rows)),
        "S1_compound_all_15_pearson_at_least_0p30": bool(all(row["separated"]["compound"]["pearson"] >= 0.30 for _, _, row in s1_rows)),
        "S1_all_15_mae_better_than_raw": bool(all(row["separated"]["overall"]["mae_s"] < row["raw"]["overall"]["mae_s"] for _, _, row in s1_rows)),
        "S1_each_seed_mean_mae_reduction_vs_S0_at_least_0p05": bool(all(run["S1_mean_mae_reduction_vs_S0"] >= 0.05 for run in runs)),
        "S1_mean_mae_reduction_vs_S0_at_least_0p10": bool(np.mean([run["S1_mean_mae_reduction_vs_S0"] for run in runs]) >= 0.10),
        "S1_all_15_waveform_cosine_at_least_0p30": bool(all(row["waveform_cosine_mean"] >= 0.30 for _, _, row in s1_rows)),
    }
    passed = bool(all(gates.values()))
    s0_pass = bool(all(run["variants"]["S0_waveform_only"]["regimes"][regime]["separated"]["overall"]["mae_reduction_vs_constant"] >= 0.10 and run["variants"]["S0_waveform_only"]["regimes"][regime]["separated"]["overall"]["pearson"] >= 0.30 for run in runs for regime in REGIMES))
    interpretation = (
        "Reset-consistent dedicated separation passed; freeze S1 and run seed400 independent confirmation."
        if passed else
        "Waveform-only separation passed but reset consistency did not add stable value; do not claim the reset loss."
        if s0_pass else
        "Neither dedicated variant transferred the oracle mechanism across identities; stop separator tuning on seed380."
    )
    summary = {
        "experiment": "P2z dedicated ISRJ separator with phase-reset consistency",
        "protocol": vars(args), "preregister_sha256": sha256_file(preregister), "device": str(device),
        "identity_audit": {
            "train": {"identity_count": int(len(train_identities)), "identity_sha256": _hash_int64(train_identities), "alignment": train_alignment, "exclusions": train_exclusions},
            "development": {"identity_count": int(len(development_identity_reference)), "identity_sha256": _hash_int64(development_identity_reference), "cross_train_overlap": int(np.intersect1d(train_identities, development_identity_reference).size), "per_regime": {regime: {"alignment": record["alignment"], "exclusions": record["exclusions"]} for regime, record in development.items()}},
        },
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "role_hashes": {name: _hash_int64(train_identities[indices]) for name, indices in roles.items()},
        "promotion_gates": gates, "development_passed": passed, "S0_ablation_passed": s0_pass,
        "mean_S1_reduction_vs_S0": float(np.mean([run["S1_mean_mae_reduction_vs_S0"] for run in runs])),
        "runs": runs, "interpretation": interpretation,
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "S0_ablation_passed": s0_pass, "promotion_gates": gates, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
