"""Train an ISRJ multi-physics periodic-sequence ordinal width expert."""

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
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader, TensorDataset

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import (
    FS_HZ, ISRJ_CLASS_INDEX, POWERS, REGIMES, SIGNAL_LENGTH, WIDTH_COLUMN,
    estimate_width_acf, metrics, sha256_file,
)
from cj0724.p2x_isrj_phase_reset_ood import estimate_width_phase_resets
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.train import set_seed


MODEL_SEEDS = (2041, 2042, 2043)
VARIANT_CHANNELS = {"A0_acf_ordinal": (0, 1, 4), "A1_multiphysics_ordinal": (0, 1, 2, 3, 4)}
LOW_LAG, HIGH_LAG = 10, 501
SEQUENCE_LENGTH = HIGH_LAG - LOW_LAG
THRESHOLDS = np.arange(11, 501, dtype=np.float32)


def _residual_normalize(sequence: np.ndarray) -> np.ndarray:
    residual = sequence - uniform_filter1d(sequence, size=31, axis=1, mode="nearest")
    scale = np.max(np.abs(residual), axis=1, keepdims=True)
    return residual / np.maximum(scale, 1.0e-6)


def extract_multiphysics_sequences(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [N,2,{SIGNAL_LENGTH}]")
    signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    spectrum = np.fft.fft(signal, n=2 * SIGNAL_LENGTH, axis=1)
    complex_acf = np.fft.ifft(spectrum * spectrum.conj(), axis=1)[:, :SIGNAL_LENGTH]
    complex_acf /= np.maximum(np.abs(complex_acf[:, :1]), 1.0e-12)
    acf_real = _residual_normalize(complex_acf.real[:, LOW_LAG:HIGH_LAG])
    acf_magnitude = _residual_normalize(np.abs(complex_acf[:, LOW_LAG:HIGH_LAG]))

    envelope = np.abs(signal)
    envelope_spectrum = np.fft.fft(envelope, n=2 * SIGNAL_LENGTH, axis=1)
    envelope_acf = np.fft.ifft(envelope_spectrum * envelope_spectrum.conj(), axis=1).real[:, :SIGNAL_LENGTH]
    envelope_acf /= np.maximum(envelope_acf[:, :1], 1.0e-12)
    envelope_sequence = _residual_normalize(envelope_acf[:, LOW_LAG:HIGH_LAG])

    phase_increment = np.angle(signal[:, 1:] * signal[:, :-1].conj())
    curvature = np.abs(np.angle(np.exp(1j * (phase_increment[:, 1:] - phase_increment[:, :-1])))) / np.pi
    curvature = np.pad(curvature, ((0, 0), (1, 1)))
    curvature_spectrum = np.fft.fft(curvature, n=2 * SIGNAL_LENGTH, axis=1)
    curvature_acf = np.fft.ifft(curvature_spectrum * curvature_spectrum.conj(), axis=1).real[:, :SIGNAL_LENGTH]
    curvature_acf /= np.maximum(curvature_acf[:, :1], 1.0e-12)
    curvature_sequence = _residual_normalize(curvature_acf[:, LOW_LAG:HIGH_LAG])

    position = np.linspace(LOW_LAG / (HIGH_LAG - 1), 1.0, SEQUENCE_LENGTH, dtype=np.float64)
    position = np.broadcast_to(position[None, :], acf_real.shape)
    return np.stack((acf_real, acf_magnitude, envelope_sequence, curvature_sequence, position), axis=1).astype(np.float32)


class SequenceOrdinalNet(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 24, 7, padding=3, bias=False), nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(48), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 123, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, len(THRESHOLDS)))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(sequence))


def ordinal_loss(logits: torch.Tensor, target_samples: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(11, 501, device=logits.device, dtype=target_samples.dtype)
    truth = (target_samples[:, None] >= thresholds[None, :]).float()
    bce = F.binary_cross_entropy_with_logits(logits, truth)
    monotonic = F.relu(logits[:, 1:] - logits[:, :-1]).mean()
    return bce + 0.05 * monotonic


def decode_ordinal(logits: torch.Tensor) -> torch.Tensor:
    return 10.0 + torch.sigmoid(logits).sum(dim=1)


def load_iq(path: Path, indices: np.ndarray, batch_size: int = 64) -> np.ndarray:
    dataset = V73IQDataset(path, indices, [WIDTH_COLUMN])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    parts = []
    try:
        for batch in loader:
            parts.append(batch["iq"].numpy())
    finally:
        dataset.close()
    return np.concatenate(parts)


def collect_role(
    regime_files: dict[str, dict[int, Path]], metadata: dict[str, dict[str, np.ndarray]],
    role_indices: np.ndarray | None,
) -> dict[str, np.ndarray]:
    arrays = {"features": [], "target_s": [], "identity": [], "power": [], "combination_size": [], "regime": [], "hard_acf": [], "hard_reset": []}
    for regime in REGIMES:
        record = metadata[regime]
        selected = record["selected"] if role_indices is None else np.intersect1d(record["selected"], role_indices)
        target = record["values"][WIDTH_COLUMN, selected]
        for power in POWERS:
            iq = load_iq(regime_files[regime][power], selected)
            arrays["features"].append(extract_multiphysics_sequences(iq))
            arrays["target_s"].append(target)
            arrays["identity"].append(record["identities"][selected])
            arrays["power"].append(np.full(len(selected), power, dtype=np.int16))
            arrays["combination_size"].append(record["labels"][selected].sum(axis=1).astype(np.int8))
            arrays["regime"].append(np.full(len(selected), REGIMES.index(regime), dtype=np.int8))
            arrays["hard_acf"].append(estimate_width_acf(iq)["prediction"])
            arrays["hard_reset"].append(estimate_width_phase_resets(iq)["prediction"])
    return {name: np.concatenate(parts) for name, parts in arrays.items()}


def train_model(
    features: np.ndarray, target_s: np.ndarray, selection_features: np.ndarray, selection_target_s: np.ndarray,
    channels: tuple[int, ...], seed: int, run_dir: Path, device: torch.device, args: argparse.Namespace,
) -> tuple[SequenceOrdinalNet, list[dict[str, float]], float]:
    set_seed(seed)
    model = SequenceOrdinalNet(len(channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mean = features[:, channels].mean(axis=(0, 2), keepdims=True)
    scale = features[:, channels].std(axis=(0, 2), keepdims=True)
    scale = np.maximum(scale, 1.0e-4)
    fit_x = torch.from_numpy(((features[:, channels] - mean) / scale).astype(np.float32))
    fit_y = torch.from_numpy(np.rint(target_s * FS_HZ).astype(np.float32))
    selection_x = torch.from_numpy(((selection_features[:, channels] - mean) / scale).astype(np.float32))
    selection_y = torch.from_numpy(np.rint(selection_target_s * FS_HZ).astype(np.float32))
    fit_loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    selection_loader = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size, shuffle=False)
    run_dir.mkdir(parents=True)
    np.savez_compressed(run_dir / "normalization.npz", mean=mean, scale=scale, channels=np.asarray(channels, dtype=np.int8))
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = ordinal_loss(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        selection_losses = []
        with torch.inference_mode():
            for x, y in selection_loader:
                selection_losses.append(float(ordinal_loss(model(x.to(device)), y.to(device))))
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
def predict_model(model: SequenceOrdinalNet, features: np.ndarray, normalization_path: Path, device: torch.device, batch_size: int) -> np.ndarray:
    normalization = np.load(normalization_path)
    channels = normalization["channels"].astype(np.int64)
    standardized = ((features[:, channels] - normalization["mean"]) / normalization["scale"]).astype(np.float32)
    loader = DataLoader(torch.from_numpy(standardized), batch_size=batch_size, shuffle=False)
    model.eval()
    return np.concatenate([decode_ordinal(model(x.to(device))).cpu().numpy() for x in loader]) / FS_HZ


def evaluate_predictions(prediction: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, object]:
    output = {"overall": metrics(prediction, data["target_s"])}
    compound = data["combination_size"] >= 2
    output["compound"] = metrics(prediction[compound], data["target_s"][compound])
    output["by_power"] = {str(power): metrics(prediction[data["power"] == power], data["target_s"][data["power"] == power]) for power in POWERS}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3a_isrj_ordinal_train_seed3400000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3a_isrj_ordinal_dev_seed3500000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p3a_isrj_multiphysics_ordinal_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3a_isrj_multiphysics_ordinal_development_20260801")
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
    metadata = {"train": {}, "development": {}}
    identity_references = {}
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
                raise RuntimeError(f"P3a {role} count mismatch in {regime}")
            if reference is None:
                reference = identities
            elif not np.array_equal(reference, identities):
                raise RuntimeError(f"P3a {role} regimes are not identity paired")
            exclusions = []
            for path in excluded_paths:
                prior, _ = read_identity_metadata(path)
                exclusions.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, prior).size)})
            if any(row["overlap_count"] for row in exclusions):
                raise RuntimeError(f"P3a {role} identity overlap")
            metadata[role][regime] = {"identities": identities, "labels": labels, "values": values, "selected": selected, "alignment": alignment, "exclusions": exclusions}
        identity_references[role] = reference
    if np.intersect1d(identity_references["train"], identity_references["development"]).size:
        raise RuntimeError("P3a train/development identity overlap")

    train_labels = metadata["train"]["ID_control"]["labels"]
    roles = stratified_roles_10(train_labels, 20260801)
    fit_data = collect_role(roots["train"], metadata["train"], roles["fit"])
    selection_data = collect_role(roots["train"], metadata["train"], roles["selection"])
    development_data = collect_role(roots["development"], metadata["development"], None)
    np.savez_compressed(output_root / "feature_receipt.npz", fit_identity=fit_data["identity"], selection_identity=selection_data["identity"], development_identity=development_data["identity"], development_target_s=development_data["target_s"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in MODEL_SEEDS:
        variants = {}
        for variant, channels in VARIANT_CHANNELS.items():
            run_dir = output_root / f"seed_{seed}" / variant
            model, history, elapsed = train_model(fit_data["features"], fit_data["target_s"], selection_data["features"], selection_data["target_s"], channels, seed, run_dir, device, args)
            prediction = predict_model(model, development_data["features"], run_dir / "normalization.npz", device, args.batch_size)
            by_regime = {}
            for regime_index, regime in enumerate(REGIMES):
                mask = development_data["regime"] == regime_index
                subset = {name: values[mask] for name, values in development_data.items() if name != "features"}
                result = evaluate_predictions(prediction[mask], subset)
                result["hard_acf"] = evaluate_predictions(subset["hard_acf"], subset)
                result["hard_phase_reset"] = evaluate_predictions(subset["hard_reset"], subset)
                by_regime[regime] = result
                regime_dir = run_dir / regime
                regime_dir.mkdir(parents=True)
                np.savez_compressed(regime_dir / "predictions.npz", identity=subset["identity"], power=subset["power"], combination_size=subset["combination_size"], target_s=subset["target_s"], prediction=prediction[mask], hard_acf=subset["hard_acf"], hard_reset=subset["hard_reset"])
                (regime_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
            receipt = {"variant": variant, "channels": channels, "epochs_ran": len(history), "train_seconds": elapsed, "checkpoint_sha256": sha256_file(run_dir / "best.pt"), "regimes": by_regime}
            (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
            (run_dir / "receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
            variants[variant] = receipt
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        a0 = np.mean([variants["A0_acf_ordinal"]["regimes"][regime]["overall"]["mae_s"] for regime in REGIMES])
        a1 = np.mean([variants["A1_multiphysics_ordinal"]["regimes"][regime]["overall"]["mae_s"] for regime in REGIMES])
        run = {"seed": seed, "variants": variants, "A1_mean_mae_reduction_vs_A0": float(1.0 - a1 / max(a0, 1.0e-12))}
        runs.append(run)
        print(json.dumps({"seed": seed, "A1_mean_mae_reduction_vs_A0": run["A1_mean_mae_reduction_vs_A0"], "A1": {regime: variants["A1_multiphysics_ordinal"]["regimes"][regime]["overall"] for regime in REGIMES}}), flush=True)

    a1_rows = [(run, regime, run["variants"]["A1_multiphysics_ordinal"]["regimes"][regime]) for run in runs for regime in REGIMES]
    gates = {
        "identity_overlap_zero": bool(all(ex["overlap_count"] == 0 for role in metadata.values() for record in role.values() for ex in record["exclusions"])),
        "A1_all_15_reduction_vs_constant_at_least_0p10": bool(all(row["overall"]["mae_reduction_vs_constant"] >= 0.10 for _, _, row in a1_rows)),
        "A1_all_15_pearson_at_least_0p30": bool(all(row["overall"]["pearson"] >= 0.30 for _, _, row in a1_rows)),
        "A1_compound_all_15_reduction_vs_constant_at_least_0p10": bool(all(row["compound"]["mae_reduction_vs_constant"] >= 0.10 for _, _, row in a1_rows)),
        "A1_compound_all_15_pearson_at_least_0p30": bool(all(row["compound"]["pearson"] >= 0.30 for _, _, row in a1_rows)),
        "A1_all_15_mae_better_than_hard_acf": bool(all(row["overall"]["mae_s"] < row["hard_acf"]["overall"]["mae_s"] for _, _, row in a1_rows)),
        "A1_each_seed_mean_mae_reduction_vs_A0_at_least_0p02": bool(all(run["A1_mean_mae_reduction_vs_A0"] >= 0.02 for run in runs)),
        "A1_mean_mae_reduction_vs_A0_at_least_0p05": bool(np.mean([run["A1_mean_mae_reduction_vs_A0"] for run in runs]) >= 0.05),
    }
    passed = bool(all(gates.values()))
    a0_pass = bool(all(run["variants"]["A0_acf_ordinal"]["regimes"][regime]["overall"]["mae_reduction_vs_constant"] >= 0.10 and run["variants"]["A0_acf_ordinal"]["regimes"][regime]["overall"]["pearson"] >= 0.30 for run in runs for regime in REGIMES))
    summary = {
        "experiment": "P3a ISRJ multi-physics periodic-sequence ordinal expert",
        "protocol": vars(args), "preregister_sha256": sha256_file(preregister), "device": str(device),
        "identity_audit": {
            "train_identity_sha256": _hash_int64(identity_references["train"]),
            "development_identity_sha256": _hash_int64(identity_references["development"]),
            "cross_role_overlap": int(np.intersect1d(identity_references["train"], identity_references["development"]).size),
            "role_hashes": {name: _hash_int64(identity_references["train"][indices]) for name, indices in roles.items()},
        },
        "promotion_gates": gates, "development_passed": passed, "A0_ablation_passed": a0_pass,
        "mean_A1_reduction_vs_A0": float(np.mean([run["A1_mean_mae_reduction_vs_A0"] for run in runs])),
        "runs": runs,
        "interpretation": (
            "Multi-physics ordinal expert passed; freeze it for leave-range-out and new-correlation validation."
            if passed else
            "ACF ordinal passed but extra physical channels lacked stable value; retain A0 only."
            if a0_pass else
            "Neither ordinal expert met the frozen gates; stop tuning on seed350 and inspect feature-target sufficiency."
        ),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "A0_ablation_passed": a0_pass, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
