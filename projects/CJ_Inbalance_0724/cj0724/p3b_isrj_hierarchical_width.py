"""Evaluate a hierarchical ISRJ width-band and local-regression expert."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import FS_HZ, ISRJ_CLASS_INDEX, POWERS, REGIMES, WIDTH_COLUMN, metrics, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3a_isrj_multiphysics_ordinal import (
    SequenceOrdinalNet, collect_role, evaluate_predictions, predict_model, train_model,
)
from cj0724.train import set_seed


MODEL_SEEDS = (2041, 2042, 2043)
CHANNELS = (0, 1, 4)
BAND_LOW = np.asarray([10, 20, 50, 200, 300], dtype=np.float32)
BAND_HIGH = np.asarray([19, 49, 199, 299, 500], dtype=np.float32)


def width_band_numpy(width_samples: np.ndarray) -> np.ndarray:
    values = np.asarray(width_samples)
    return np.select(
        [values <= 19, values <= 49, values <= 199, values <= 299],
        [0, 1, 2, 3], default=4,
    ).astype(np.int64)


class HierarchicalWidthNet(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 24, 7, padding=3, bias=False), nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(48), nn.GELU(),
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(48 * 123, 128), nn.GELU(), nn.Dropout(0.1))
        self.band_head = nn.Linear(128, 5)
        self.local_head = nn.Linear(128, 5)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(self.features(sequence))
        return self.band_head(hidden), self.local_head(hidden)


def hierarchical_loss(band_logits: torch.Tensor, local_logits: torch.Tensor, target_samples: torch.Tensor) -> torch.Tensor:
    target_np = target_samples.detach().cpu().numpy()
    target_band = torch.from_numpy(width_band_numpy(target_np)).to(target_samples.device)
    low = torch.from_numpy(BAND_LOW).to(target_samples.device)[target_band]
    high = torch.from_numpy(BAND_HIGH).to(target_samples.device)[target_band]
    local_target = (target_samples - low) / (high - low).clamp_min(1.0)
    selected_local = local_logits.gather(1, target_band[:, None]).squeeze(1)
    return F.cross_entropy(band_logits, target_band) + F.smooth_l1_loss(torch.sigmoid(selected_local), local_target)


def decode_hierarchical(band_logits: torch.Tensor, local_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    band = band_logits.argmax(dim=1)
    low = torch.as_tensor(BAND_LOW, device=band_logits.device)[band]
    high = torch.as_tensor(BAND_HIGH, device=band_logits.device)[band]
    local = torch.sigmoid(local_logits.gather(1, band[:, None]).squeeze(1))
    return low + local * (high - low), band


def train_hierarchical(
    fit_features: np.ndarray, fit_target_s: np.ndarray, selection_features: np.ndarray, selection_target_s: np.ndarray,
    seed: int, run_dir: Path, device: torch.device, args: argparse.Namespace,
) -> tuple[HierarchicalWidthNet, list[dict[str, float]], float]:
    set_seed(seed)
    mean = fit_features[:, CHANNELS].mean(axis=(0, 2), keepdims=True)
    scale = np.maximum(fit_features[:, CHANNELS].std(axis=(0, 2), keepdims=True), 1.0e-4)
    fit_x = torch.from_numpy(((fit_features[:, CHANNELS] - mean) / scale).astype(np.float32))
    fit_y = torch.from_numpy(np.rint(fit_target_s * FS_HZ).astype(np.float32))
    selection_x = torch.from_numpy(((selection_features[:, CHANNELS] - mean) / scale).astype(np.float32))
    selection_y = torch.from_numpy(np.rint(selection_target_s * FS_HZ).astype(np.float32))
    fit_loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    selection_loader = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size, shuffle=False)
    model = HierarchicalWidthNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    run_dir.mkdir(parents=True)
    np.savez_compressed(run_dir / "normalization.npz", mean=mean, scale=scale, channels=np.asarray(CHANNELS, dtype=np.int8))
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = hierarchical_loss(*model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        selection_losses = []
        with torch.inference_mode():
            for x, y in selection_loader:
                selection_losses.append(float(hierarchical_loss(*model(x.to(device)), y.to(device))))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "selection_loss": float(np.mean(selection_losses))}
        history.append(row)
        if row["selection_loss"] < best:
            best, stale = row["selection_loss"], 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    return model, history, float(time.perf_counter() - started)


@torch.inference_mode()
def predict_hierarchical(model: HierarchicalWidthNet, features: np.ndarray, normalization_path: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    normalization = np.load(normalization_path)
    channels = normalization["channels"].astype(np.int64)
    standardized = ((features[:, channels] - normalization["mean"]) / normalization["scale"]).astype(np.float32)
    loader = DataLoader(torch.from_numpy(standardized), batch_size=batch_size, shuffle=False)
    predictions, bands = [], []
    model.eval()
    for x in loader:
        prediction, band = decode_hierarchical(*model(x.to(device)))
        predictions.append(prediction.cpu().numpy())
        bands.append(band.cpu().numpy())
    return np.concatenate(predictions) / FS_HZ, np.concatenate(bands)


def cross_band_fraction(prediction_band: np.ndarray, target_s: np.ndarray) -> float:
    target_band = width_band_numpy(np.rint(target_s * FS_HZ))
    return float(np.mean(np.asarray(prediction_band) != target_band))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3b_isrj_hierarchical_train_seed200000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3b_isrj_hierarchical_dev_seed250000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p3b_isrj_hierarchical_width_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3b_isrj_hierarchical_width_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    preregister = project / args.preregister
    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded_paths = [Path(protocol["data_file"])]
    excluded_paths.extend(Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files)
    roots = {
        "train": {regime: transition_files(project / args.train_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES},
        "development": {regime: transition_files(project / args.development_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES},
    }
    metadata, references = {"train": {}, "development": {}}, {}
    for role in ("train", "development"):
        reference = None
        for regime in REGIMES:
            files = roots[role][regime]
            alignment = audit_power_alignment(files)
            identities, labels = read_identity_metadata(files[POWERS[0]])
            with h5py.File(files[POWERS[0]], "r") as handle:
                values = np.asarray(handle["parameter_values"], dtype=np.float64)
                mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
            selected = np.flatnonzero((labels[:, ISRJ_CLASS_INDEX] > 0) & mask & np.isfinite(values[WIDTH_COLUMN]))
            if len(selected) != 320:
                raise RuntimeError(f"P3b {role} count mismatch in {regime}")
            if reference is None:
                reference = identities
            elif not np.array_equal(reference, identities):
                raise RuntimeError(f"P3b {role} regimes are not paired")
            exclusions = []
            for path in excluded_paths:
                prior, _ = read_identity_metadata(path)
                exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
            if any(row["overlap_count"] for row in exclusions):
                raise RuntimeError(f"P3b {role} identity overlap")
            metadata[role][regime] = {"identities": identities, "labels": labels, "values": values, "selected": selected, "alignment": alignment, "exclusions": exclusions}
        references[role] = reference
    if np.intersect1d(references["train"], references["development"]).size:
        raise RuntimeError("P3b train/development identity overlap")

    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801)
    fit_data = collect_role(roots["train"], metadata["train"], roles["fit"])
    selection_data = collect_role(roots["train"], metadata["train"], roles["selection"])
    development_data = collect_role(roots["development"], metadata["development"], None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in MODEL_SEEDS:
        variants = {}
        flat_dir = output_root / f"seed_{seed}" / "B0_flat_acf_ordinal"
        flat_model, flat_history, flat_elapsed = train_model(fit_data["features"], fit_data["target_s"], selection_data["features"], selection_data["target_s"], CHANNELS, seed, flat_dir, device, args)
        flat_prediction = predict_model(flat_model, development_data["features"], flat_dir / "normalization.npz", device, args.batch_size)
        variants["B0_flat_acf_ordinal"] = {"history": flat_history, "elapsed": flat_elapsed, "prediction": flat_prediction, "checkpoint_sha256": sha256_file(flat_dir / "best.pt")}
        del flat_model

        hierarchical_dir = output_root / f"seed_{seed}" / "B1_hierarchical"
        hierarchical_model, hierarchical_history, hierarchical_elapsed = train_hierarchical(fit_data["features"], fit_data["target_s"], selection_data["features"], selection_data["target_s"], seed, hierarchical_dir, device, args)
        hierarchical_prediction, predicted_band = predict_hierarchical(hierarchical_model, development_data["features"], hierarchical_dir / "normalization.npz", device, args.batch_size)
        variants["B1_hierarchical"] = {"history": hierarchical_history, "elapsed": hierarchical_elapsed, "prediction": hierarchical_prediction, "predicted_band": predicted_band, "checkpoint_sha256": sha256_file(hierarchical_dir / "best.pt")}
        del hierarchical_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        receipts = {}
        for variant, payload in variants.items():
            by_regime = {}
            run_dir = output_root / f"seed_{seed}" / variant
            for regime_index, regime in enumerate(REGIMES):
                mask = development_data["regime"] == regime_index
                subset = {name: values[mask] for name, values in development_data.items() if name != "features"}
                result = evaluate_predictions(payload["prediction"][mask], subset)
                result["hard_acf"] = evaluate_predictions(subset["hard_acf"], subset)
                if variant == "B1_hierarchical":
                    result["cross_band_error_fraction"] = cross_band_fraction(payload["predicted_band"][mask], subset["target_s"])
                by_regime[regime] = result
                regime_dir = run_dir / regime
                regime_dir.mkdir(parents=True)
                save = {"identity": subset["identity"], "power": subset["power"], "combination_size": subset["combination_size"], "target_s": subset["target_s"], "prediction": payload["prediction"][mask], "hard_acf": subset["hard_acf"]}
                if variant == "B1_hierarchical":
                    save["predicted_band"] = payload["predicted_band"][mask]
                np.savez_compressed(regime_dir / "predictions.npz", **save)
                (regime_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
            receipt = {"variant": variant, "epochs_ran": len(payload["history"]), "train_seconds": payload["elapsed"], "checkpoint_sha256": payload["checkpoint_sha256"], "regimes": by_regime}
            (run_dir / "history.json").write_text(json.dumps(_jsonable(payload["history"]), indent=2), encoding="utf-8")
            (run_dir / "receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
            receipts[variant] = receipt
        b0 = np.mean([receipts["B0_flat_acf_ordinal"]["regimes"][regime]["overall"]["mae_s"] for regime in REGIMES])
        b1 = np.mean([receipts["B1_hierarchical"]["regimes"][regime]["overall"]["mae_s"] for regime in REGIMES])
        run = {"seed": seed, "variants": receipts, "B1_mean_mae_reduction_vs_B0": float(1.0 - b1 / max(b0, 1.0e-12))}
        runs.append(run)
        print(json.dumps({"seed": seed, "B1_mean_mae_reduction_vs_B0": run["B1_mean_mae_reduction_vs_B0"], "B1": {regime: receipts["B1_hierarchical"]["regimes"][regime]["overall"] for regime in REGIMES}}), flush=True)

    b1_rows = [(run, regime, run["variants"]["B1_hierarchical"]["regimes"][regime]) for run in runs for regime in REGIMES]
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for role in metadata.values() for record in role.values() for ex in record["exclusions"])),
        "B1_all_15_reduction_vs_constant_at_least_0p10": bool(all(row["overall"]["mae_reduction_vs_constant"] >= 0.10 for _, _, row in b1_rows)),
        "B1_all_15_pearson_at_least_0p30": bool(all(row["overall"]["pearson"] >= 0.30 for _, _, row in b1_rows)),
        "B1_compound_all_15_reduction_vs_constant_at_least_0p10": bool(all(row["compound"]["mae_reduction_vs_constant"] >= 0.10 for _, _, row in b1_rows)),
        "B1_compound_all_15_pearson_at_least_0p30": bool(all(row["compound"]["pearson"] >= 0.30 for _, _, row in b1_rows)),
        "B1_all_15_mae_better_than_hard_acf": bool(all(row["overall"]["mae_s"] < row["hard_acf"]["overall"]["mae_s"] for _, _, row in b1_rows)),
        "B1_each_seed_mean_mae_reduction_vs_B0_at_least_0p02": bool(all(run["B1_mean_mae_reduction_vs_B0"] >= 0.02 for run in runs)),
        "B1_mean_mae_reduction_vs_B0_at_least_0p05": bool(np.mean([run["B1_mean_mae_reduction_vs_B0"] for run in runs]) >= 0.05),
        "B1_cross_band_error_fraction_all_15_at_most_0p10": bool(all(row["cross_band_error_fraction"] <= 0.10 for _, _, row in b1_rows)),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P3b hierarchical ISRJ width-band and local-regression expert",
        "protocol": vars(args), "preregister_sha256": sha256_file(preregister), "device": str(device),
        "identity_audit": {"train_sha256": _hash_int64(references["train"]), "development_sha256": _hash_int64(references["development"]), "cross_overlap": int(np.intersect1d(references["train"], references["development"]).size)},
        "promotion_gates": gates, "development_passed": passed,
        "mean_B1_reduction_vs_B0": float(np.mean([run["B1_mean_mae_reduction_vs_B0"] for run in runs])),
        "runs": runs,
        "interpretation": "Hierarchical width expert passed; freeze it for range-holdout validation." if passed else "Hierarchical gating did not meet the frozen gates; stop learned ISRJ-width development and retain only oracle sufficiency.",
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
