"""Train an identity-safe MFT inverse-filter delay-sequence ordinal expert."""

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
from cj0724.p2w_isrj_width_oracle_ood import POWERS, SIGNAL_LENGTH, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3c_mft_sparse_deconvolution import (
    COUNT_COLUMN, MFT_CLASS_INDEX, REGIMES, reference_signal,
)
from cj0724.train import set_seed


MODEL_SEEDS = (2041, 2042, 2043)
SEARCH_START, SEARCH_STOP = 350, 1101
LEVELS = np.arange(1, 8, dtype=np.float32)


def extract_inverse_delay_sequences(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [N,2,{SIGNAL_LENGTH}]")
    signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    reference_spectrum = np.fft.fft(reference_signal())
    power = np.abs(reference_spectrum) ** 2
    impulse = np.fft.ifft(
        np.fft.fft(signal, axis=1) * reference_spectrum.conj()[None, :]
        / (power[None, :] + 1.0e-6 * float(power.max())), axis=1,
    )
    magnitude = np.abs(impulse[:, SEARCH_START:SEARCH_STOP])
    normalized = magnitude / np.maximum(magnitude.max(axis=1, keepdims=True), 1.0e-12)
    log_magnitude = np.log1p(100.0 * normalized) / np.log(101.0)
    background = uniform_filter1d(normalized, size=31, axis=1, mode="nearest")
    contrast = np.maximum(normalized - background, 0.0)
    contrast /= np.maximum(contrast.max(axis=1, keepdims=True), 1.0e-12)
    local = uniform_filter1d(normalized, size=9, axis=1, mode="nearest")
    sharpness = np.clip(normalized / np.maximum(local, 1.0e-4), 0.0, 10.0) / 10.0
    position = np.broadcast_to(np.linspace(0.0, 1.0, normalized.shape[1])[None, :], normalized.shape)
    return np.stack((log_magnitude, contrast, sharpness, position), axis=1).astype(np.float32)


class InverseOrdinalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 24, 9, padding=4, bias=False), nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 7))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(sequence))


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    truth = (target[:, None] > torch.arange(1, 8, device=target.device)[None, :]).float()
    monotonic = F.relu(logits[:, 1:] - logits[:, :-1]).mean()
    return F.binary_cross_entropy_with_logits(logits, truth) + 0.05 * monotonic


def decode_ordinal(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 + torch.sigmoid(logits).sum(dim=1)


def load_iq(path: Path, indices: np.ndarray) -> np.ndarray:
    dataset = V73IQDataset(path, indices, [COUNT_COLUMN])
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    try:
        return np.concatenate([batch["iq"].numpy() for batch in loader])
    finally:
        dataset.close()


def collect(files_by_regime: dict[str, dict[int, Path]], metadata: dict[str, dict[str, np.ndarray]], role: np.ndarray | None) -> dict[str, np.ndarray]:
    parts = {name: [] for name in ("features", "target", "identity", "power", "combination_size", "regime")}
    for regime_index, regime in enumerate(REGIMES):
        record = metadata[regime]
        selected = record["selected"] if role is None else np.intersect1d(record["selected"], role)
        target = record["values"][COUNT_COLUMN, selected]
        for power in POWERS:
            parts["features"].append(extract_inverse_delay_sequences(load_iq(files_by_regime[regime][power], selected)))
            parts["target"].append(target)
            parts["identity"].append(record["identities"][selected])
            parts["power"].append(np.full(len(selected), power, dtype=np.int16))
            parts["combination_size"].append(record["labels"][selected].sum(axis=1).astype(np.int8))
            parts["regime"].append(np.full(len(selected), regime_index, dtype=np.int8))
    return {name: np.concatenate(values) for name, values in parts.items()}


def metrics(prediction: np.ndarray, target: np.ndarray, constant: float) -> dict[str, float]:
    prediction, target = np.asarray(prediction), np.asarray(target)
    mae = float(np.abs(prediction - target).mean())
    baseline = float(np.abs(constant - target).mean())
    pearson = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) > 0 and np.std(target) > 0 else 0.0
    return {"count": int(len(target)), "mae": mae, "constant_mae": baseline, "mae_reduction_vs_constant": float(1.0 - mae / max(baseline, 1.0e-12)), "pearson": pearson, "rounded_accuracy": float(np.mean(np.rint(prediction) == target))}


def evaluate(prediction: np.ndarray, data: dict[str, np.ndarray], constant: float) -> dict[str, object]:
    compound = data["combination_size"] >= 2
    return {
        "overall": metrics(prediction, data["target"], constant),
        "compound": metrics(prediction[compound], data["target"][compound], constant),
        "by_power": {str(power): metrics(prediction[data["power"] == power], data["target"][data["power"] == power], constant) for power in POWERS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3d_mft_ordinal_train_seed460000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3d_mft_ordinal_dev_seed470000000_{regime}_rechunked")
    parser.add_argument("--training-protocol", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json")
    parser.add_argument("--preregister", default="configs/p3d_mft_inverse_ordinal_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3d_mft_inverse_ordinal_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    preregister = project / args.preregister
    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded = [Path(protocol["data_file"])] + [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    roots = {
        "train": {regime: transition_files(project / args.train_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES},
        "development": {regime: transition_files(project / args.development_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES},
    }
    metadata, identity_reference = {"train": {}, "development": {}}, {}
    for role_name in ("train", "development"):
        reference = None
        for regime in REGIMES:
            files = roots[role_name][regime]
            alignment = audit_power_alignment(files)
            identities, labels = read_identity_metadata(files[POWERS[0]])
            with h5py.File(files[POWERS[0]], "r") as handle:
                values = np.asarray(handle["parameter_values"], dtype=np.float64)
                mask = np.asarray(handle["parameter_mask"][COUNT_COLUMN], dtype=np.uint8).astype(bool)
            selected = np.flatnonzero((labels[:, MFT_CLASS_INDEX] > 0) & mask & np.isfinite(values[COUNT_COLUMN]))
            if len(identities) != 630 or len(selected) != 320:
                raise RuntimeError(f"P3d {role_name} count mismatch in {regime}")
            if reference is None:
                reference = identities
            elif not np.array_equal(reference, identities):
                raise RuntimeError(f"P3d {role_name} regimes are not identity paired")
            overlap = []
            for path in excluded:
                previous, _ = read_identity_metadata(path)
                overlap.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, previous).size)})
            if any(row["overlap_count"] for row in overlap):
                raise RuntimeError(f"P3d {role_name} identity overlap")
            metadata[role_name][regime] = {"identities": identities, "labels": labels, "values": values, "selected": selected, "alignment": alignment, "exclusions": overlap}
        identity_reference[role_name] = reference
    if np.intersect1d(identity_reference["train"], identity_reference["development"]).size:
        raise RuntimeError("P3d train/development identity overlap")

    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801)
    fit = collect(roots["train"], metadata["train"], roles["fit"])
    selection = collect(roots["train"], metadata["train"], roles["selection"])
    development = collect(roots["development"], metadata["development"], None)
    constant = float(np.median(fit["target"]))
    mean = fit["features"].mean(axis=(0, 2), keepdims=True)
    scale = np.maximum(fit["features"].std(axis=(0, 2), keepdims=True), 1.0e-4)
    np.savez_compressed(output_root / "feature_receipt.npz", mean=mean, scale=scale, constant=constant, fit_identity=fit["identity"], selection_identity=selection["identity"], development_identity=development["identity"])
    fit_x = torch.from_numpy(((fit["features"] - mean) / scale).astype(np.float32))
    fit_y = torch.from_numpy(fit["target"].astype(np.float32))
    selection_x = torch.from_numpy(((selection["features"] - mean) / scale).astype(np.float32))
    selection_y = torch.from_numpy(selection["target"].astype(np.float32))
    development_x = torch.from_numpy(((development["features"] - mean) / scale).astype(np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in MODEL_SEEDS:
        set_seed(seed)
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir()
        model = InverseOrdinalNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        fit_loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        selection_loader = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size, shuffle=False)
        best, stale, history, started = float("inf"), 0, [], time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train(); training_losses = []
            for x, y in fit_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = ordinal_loss(model(x.to(device)), y.to(device))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                training_losses.append(float(loss.detach()))
            model.eval(); selection_losses = []
            with torch.inference_mode():
                for x, y in selection_loader:
                    selection_losses.append(float(ordinal_loss(model(x.to(device)), y.to(device))))
            row = {"epoch": epoch, "train_loss": float(np.mean(training_losses)), "selection_loss": float(np.mean(selection_losses))}
            history.append(row)
            if row["selection_loss"] < best:
                best, stale = row["selection_loss"], 0; torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience: break
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)); model.eval()
        loader = DataLoader(development_x, batch_size=args.batch_size, shuffle=False)
        with torch.inference_mode():
            prediction = np.concatenate([decode_ordinal(model(x.to(device))).cpu().numpy() for x in loader])
        regimes = {}
        for regime_index, regime in enumerate(REGIMES):
            chosen = development["regime"] == regime_index
            subset = {name: values[chosen] for name, values in development.items() if name != "features"}
            result = evaluate(prediction[chosen], subset, constant)
            regimes[regime] = result
            np.savez_compressed(run_dir / f"{regime}_predictions.npz", prediction=prediction[chosen], target=subset["target"], identity=subset["identity"], power=subset["power"], combination_size=subset["combination_size"])
        receipt = {"seed": seed, "epochs_ran": len(history), "best_selection_loss": best, "train_seconds": float(time.perf_counter() - started), "checkpoint_sha256": sha256_file(run_dir / "best.pt"), "regimes": regimes}
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (run_dir / "receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
        runs.append(receipt)
        print(json.dumps({"seed": seed, "regimes": {name: value["overall"] for name, value in regimes.items()}}), flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    gates = {
        "identity_overlap_zero": bool(all(row["overlap_count"] == 0 for role in metadata.values() for record in role.values() for row in record["exclusions"])),
        "all_15_reduction_vs_constant_at_least_0p10": bool(all(run["regimes"][regime]["overall"]["mae_reduction_vs_constant"] >= 0.10 for run in runs for regime in REGIMES)),
        "all_15_pearson_at_least_0p30": bool(all(run["regimes"][regime]["overall"]["pearson"] >= 0.30 for run in runs for regime in REGIMES)),
        "compound_all_15_reduction_vs_constant_at_least_0p10": bool(all(run["regimes"][regime]["compound"]["mae_reduction_vs_constant"] >= 0.10 for run in runs for regime in REGIMES)),
        "compound_all_15_pearson_at_least_0p30": bool(all(run["regimes"][regime]["compound"]["pearson"] >= 0.30 for run in runs for regime in REGIMES)),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P3d MFT inverse-filter delay-sequence ordinal expert", "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister), "device": str(device), "constant_baseline": constant,
        "identity_audit": {"train_sha256": _hash_int64(identity_reference["train"]), "development_sha256": _hash_int64(identity_reference["development"]), "cross_role_overlap": int(np.intersect1d(identity_reference["train"], identity_reference["development"]).size)},
        "promotion_gates": gates, "development_passed": passed, "runs": runs,
        "interpretation": "Freeze for predicted-label-gated attachment on fresh identities." if passed else "Frozen MFT ordinal gates failed; do not attach and do not tune on seed470000000.",
        "classification_loss_used": False, "oracle_positive_only": True, "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
