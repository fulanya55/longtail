"""P3o-M: fixed-bin versus sparse-event ordinal MFT-count experiment.

This intentionally remains an oracle-positive parameter screen.  It does not
modify a classifier, does not use a predicted label, and cannot make an
end-to-end claim.  Its only question is whether explicit sparse-event selection
adds information beyond equally sized fixed delay-bin tokens on fresh compound
waveforms.
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from torch.utils.data import DataLoader, TensorDataset

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import POWERS, SIGNAL_LENGTH, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import COUNT_COLUMN, MFT_CLASS_INDEX, REGIMES, reference_signal
from cj0724.p3o_low_power_attribution import audit_identities
from cj0724.train import set_seed


MODEL_SEEDS = (2201, 2202, 2203)
SEARCH_START, SEARCH_STOP = 350, 1101
TOKEN_COUNT = 16


def inverse_delay_channels(iq: np.ndarray) -> np.ndarray:
    """Create four deterministic physical channels over the P3d delay window."""
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must have shape [N,2,{SIGNAL_LENGTH}]")
    signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    spectrum = np.fft.fft(reference_signal())
    power = np.abs(spectrum) ** 2
    impulse = np.fft.ifft(
        np.fft.fft(signal, axis=1) * spectrum.conj()[None, :] /
        (power[None, :] + 1.0e-6 * float(power.max())), axis=1,
    )
    magnitude = np.abs(impulse[:, SEARCH_START:SEARCH_STOP])
    normalized = magnitude / np.maximum(magnitude.max(axis=1, keepdims=True), 1.0e-12)
    contrast = np.maximum(normalized - uniform_filter1d(normalized, 31, axis=1, mode="nearest"), 0.0)
    contrast /= np.maximum(contrast.max(axis=1, keepdims=True), 1.0e-12)
    local = uniform_filter1d(normalized, 9, axis=1, mode="nearest")
    sharpness = np.clip(normalized / np.maximum(local, 1.0e-4), 0.0, 10.0) / 10.0
    position = np.broadcast_to(np.linspace(0.0, 1.0, normalized.shape[1], dtype=np.float64)[None, :], normalized.shape)
    return np.stack((normalized, contrast, sharpness, position), axis=1).astype(np.float32)


def fixed_bin_tokens(channels: np.ndarray, token_count: int = TOKEN_COUNT) -> np.ndarray:
    """Equal-information C0: one max-contrast point from each fixed delay bin."""
    channels = np.asarray(channels, dtype=np.float32)
    if channels.ndim != 3 or channels.shape[1] != 4:
        raise ValueError("channels must be [N,4,L]")
    edges = np.linspace(0, channels.shape[2], token_count + 1, dtype=np.int64)
    output = np.empty((len(channels), token_count, 4), dtype=np.float32)
    contrast = channels[:, 1]
    for index in range(token_count):
        start, stop = edges[index], edges[index + 1]
        selected = start + np.argmax(contrast[:, start:stop], axis=1)
        output[:, index] = channels[np.arange(len(channels)), :, selected]
    return output


def sparse_peak_tokens(channels: np.ndarray, token_count: int = TOKEN_COUNT) -> np.ndarray:
    """C1: rank strict local contrast maxima then retain their temporal sequence."""
    channels = np.asarray(channels, dtype=np.float32)
    if channels.ndim != 3 or channels.shape[1] != 4:
        raise ValueError("channels must be [N,4,L]")
    contrast = channels[:, 1]
    # A flat zero background is not a sparse event.  This is a structural
    # strictness rule, not a tuned amplitude threshold.
    local_max = (contrast > 0.0) & (contrast == maximum_filter1d(contrast, size=5, axis=1, mode="nearest"))
    local_max[:, :2] = False; local_max[:, -2:] = False
    output = np.zeros((len(channels), token_count, 4), dtype=np.float32)
    for row in range(len(channels)):
        candidates = np.flatnonzero(local_max[row])
        if len(candidates):
            ranked = candidates[np.argsort(-contrast[row, candidates], kind="stable")[:token_count]]
            selected = np.sort(ranked)
            # Slice channels before advanced indexing so NumPy retains the
            # intended [channel, selected-position] orientation.
            output[row, :len(selected)] = channels[row][:, selected].T
    return output


class OrdinalTokenNet(nn.Module):
    """Same ordinal model for C0/C1; only tokenization differs."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, 3, padding=1, bias=False), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, 3, padding=1, bias=False), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * TOKEN_COUNT, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 7))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (TOKEN_COUNT, 4):
            raise ValueError(f"Expected [B,{TOKEN_COUNT},4] tokens")
        return self.head(self.encoder(tokens.transpose(1, 2)))


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    truth = (target[:, None] > torch.arange(1, 8, device=target.device)[None, :]).float()
    return F.binary_cross_entropy_with_logits(logits, truth) + 0.05 * F.relu(logits[:, 1:] - logits[:, :-1]).mean()


def decode_ordinal(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 + torch.sigmoid(logits).sum(dim=1)


def _load_iq(path: Path, indices: np.ndarray) -> np.ndarray:
    dataset = V73IQDataset(path, indices, [COUNT_COLUMN]); loader = DataLoader(dataset, batch_size=64, shuffle=False)
    try:
        return np.concatenate([batch["iq"].numpy() for batch in loader])
    finally:
        dataset.close()


def _read_role(paths: dict[str, dict[int, Path]], excluded: list[Path]) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    metadata, reference = {}, None
    for regime in REGIMES:
        files = paths[regime]; audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[25])
        with h5py.File(files[25], "r") as handle:
            values = np.asarray(handle["parameter_values"], dtype=np.float64)
            mask = np.asarray(handle["parameter_mask"][COUNT_COLUMN], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, MFT_CLASS_INDEX] > 0) & mask & np.isfinite(values[COUNT_COLUMN]))
        if len(identities) != 2079 or len(selected) == 0:
            raise RuntimeError(f"P3o-M malformed MFT contract for {regime}")
        if reference is None: reference = identities
        elif not np.array_equal(reference, identities): raise RuntimeError("P3o-M regimes are not identity paired")
        for path in excluded:
            old, _ = read_identity_metadata(path)
            if np.intersect1d(identities, old).size: raise RuntimeError(f"P3o-M identity overlap with {path}")
        metadata[regime] = {"identities": identities, "labels": labels, "values": values, "selected": selected}
    assert reference is not None
    return metadata, reference


def _split_fit_selection(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    bits = np.packbits(np.asarray(labels, dtype=np.uint8), axis=1, bitorder="little")[:, 0]
    rng = np.random.default_rng(seed); parts = {"fit": [], "selection": []}
    for value in np.unique(bits):
        indices = rng.permutation(np.flatnonzero(bits == value))
        if len(indices) != 33: raise RuntimeError("P3o-M requires 33 identities per combination")
        parts["fit"].extend(indices[:25]); parts["selection"].extend(indices[25:29])
    return {name: np.sort(np.asarray(value, dtype=np.int64)) for name, value in parts.items()}


def _collect(paths, metadata, role: np.ndarray | None) -> dict[str, np.ndarray]:
    output = {name: [] for name in ("channels", "target", "identity", "power", "regime", "compound")}
    for regime_index, regime in enumerate(REGIMES):
        record = metadata[regime]; selected = record["selected"] if role is None else np.intersect1d(record["selected"], role)
        target = record["values"][COUNT_COLUMN, selected].astype(np.float32)
        for power in POWERS:
            output["channels"].append(inverse_delay_channels(_load_iq(paths[regime][power], selected)))
            output["target"].append(target); output["identity"].append(record["identities"][selected])
            output["power"].append(np.full(len(selected), power, dtype=np.int16)); output["regime"].append(np.full(len(selected), regime_index, dtype=np.int8))
            output["compound"].append((record["labels"][selected].sum(axis=1) >= 2))
    return {name: np.concatenate(value) for name, value in output.items()}


def _metrics(prediction: np.ndarray, target: np.ndarray, constant: float) -> dict[str, float]:
    error, base = np.abs(prediction - target), np.abs(float(constant) - target)
    pearson = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) and np.std(target) else 0.0
    return {"count": int(len(target)), "mae": float(error.mean()), "constant_mae": float(base.mean()), "mae_reduction_vs_constant": float(1.0 - error.mean() / max(base.mean(), 1e-12)), "pearson": pearson, "rounded_accuracy": float(np.mean(np.rint(prediction) == target))}


def _evaluate(prediction, data, constant):
    return {"overall": _metrics(prediction, data["target"], constant), "compound": _metrics(prediction[data["compound"]], data["target"][data["compound"]], constant), "by_regime": {regime: _metrics(prediction[data["regime"] == index], data["target"][data["regime"] == index], constant) for index, regime in enumerate(REGIMES)}, "by_power": {str(power): _metrics(prediction[data["power"] == power], data["target"][data["power"] == power], constant) for power in POWERS}}


def _fit_variant(name, fit, selection, development, constant, seed, args, output):
    tokenize = fixed_bin_tokens if name == "C0_fixed_bin" else sparse_peak_tokens
    mean = fit["channels"].mean(axis=(0, 2), keepdims=True); scale = np.maximum(fit["channels"].std(axis=(0, 2), keepdims=True), 1e-4)
    as_tokens = lambda data: tokenize((data["channels"] - mean) / scale)
    fit_x, fit_y = torch.from_numpy(as_tokens(fit)), torch.from_numpy(fit["target"])
    selection_x, selection_y = torch.from_numpy(as_tokens(selection)), torch.from_numpy(selection["target"])
    development_x = torch.from_numpy(as_tokens(development)); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(seed); model = OrdinalTokenNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed); loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=generator); select_loader = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); loss = ordinal_loss(model(x.to(device)), y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): selection_loss = float(np.mean([float(ordinal_loss(model(x.to(device)), y.to(device))) for x, y in select_loader]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "selection_loss": selection_loss})
        if selection_loss < best:
            best, stale = selection_loss, 0; torch.save(model.state_dict(), output / f"{name}_best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device, weights_only=True)); model.eval()
    with torch.no_grad(): prediction = np.concatenate([decode_ordinal(model(x.to(device))).cpu().numpy() for (x,) in DataLoader(TensorDataset(development_x), batch_size=args.batch_size)])
    np.savez_compressed(output / f"{name}_predictions.npz", prediction=prediction, target=development["target"], identity=development["identity"], power=development["power"], regime=development["regime"], compound=development["compound"])
    receipt = {"name": name, "seed": seed, "best_selection_loss": best, "epochs_ran": len(history), "parameters": int(sum(value.numel() for value in model.parameters())), "checkpoint_sha256": sha256_file(output / f"{name}_best.pt"), "evaluation": _evaluate(prediction, development, constant)}
    (output / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", default="D:/CJ0724_rechunked_data"); parser.add_argument("--rechunk-suffix", default="_rechunked_v2"); parser.add_argument("--data-audit", default="outputs/p3om_sparse_peak_ordinal_preflight_20260803/data_audit.json"); parser.add_argument("--output-root", default="outputs/p3om_sparse_peak_ordinal_development_20260803")
    parser.add_argument("--preregister", default="configs/p3om_sparse_peak_ordinal_preregister_20260803.json"); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project, root, output = Path(__file__).resolve().parents[1], Path(args.rechunk_root), Path(args.output_root)
    if not output.is_absolute(): output = project / output
    output.mkdir(parents=True, exist_ok=False); preregister = project / args.preregister
    method_freeze = project / "configs/p3om_sparse_peak_ordinal_method_freeze_20260803.json"
    freeze = json.loads(method_freeze.read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(preregister) or freeze["implementation_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("P3o-M preregistration or implementation hash mismatch")
    roles = {"train": "p3om_sparse_peak_train_seed1570000000", "development": "p3om_sparse_peak_development_seed1580000000"}
    paths = {role: {regime: transition_files(root / f"{stem}_{regime}{args.rechunk_suffix}", "train", POWERS) for regime in REGIMES} for role, stem in roles.items()}
    historical = json.loads((project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json").read_text(encoding="utf-8")); excluded = [Path(historical["data_file"])]
    train_meta, train_identity = _read_role(paths["train"], excluded); dev_meta, development_identity = _read_role(paths["development"], excluded)
    audited_path = project / args.data_audit
    if not audited_path.exists(): raise RuntimeError("P3o-M requires a completed pre-training identity audit")
    audit = json.loads(audited_path.read_text(encoding="utf-8"))
    if audit["maximum_prior_overlap"] or audit["cross_role_overlap"] or not audit["power_alignment_verified"] or not audit["label_contract_verified"]:
        raise RuntimeError("P3o-M pre-training audit failed")
    if audit["train_unique"] != len(np.unique(train_identity)) or audit["development_unique"] != len(np.unique(development_identity)):
        raise RuntimeError("P3o-M audited identity count mismatch")
    if not np.array_equal(train_meta["ID_control"]["labels"], dev_meta["ID_control"]["labels"]): raise RuntimeError("P3o-M label contract mismatch")
    split = _split_fit_selection(train_meta["ID_control"]["labels"], 20260803); fit, selection, development = _collect(paths["train"], train_meta, split["fit"]), _collect(paths["train"], train_meta, split["selection"]), _collect(paths["development"], dev_meta, None)
    constant = float(np.median(fit["target"])); runs = []
    for seed in MODEL_SEEDS:
        seed_output = output / f"seed_{seed}"; seed_output.mkdir(); started = time.perf_counter()
        result = {name: _fit_variant(name, fit, selection, development, constant, seed, args, seed_output) for name in ("C0_fixed_bin", "C1_sparse_peak")}
        result["seed"], result["seconds"] = seed, float(time.perf_counter() - started); (seed_output / "receipt.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8"); runs.append(result)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    deltas = []
    for run in runs:
        c0, c1 = run["C0_fixed_bin"]["evaluation"], run["C1_sparse_peak"]["evaluation"]
        deltas.append({"seed": run["seed"], "overall_mae_reduction": float(1 - c1["overall"]["mae"] / c0["overall"]["mae"]), "compound_mae_reduction": float(1 - c1["compound"]["mae"] / c0["compound"]["mae"]), "overall_pearson_delta": float(c1["overall"]["pearson"] - c0["overall"]["pearson"]), "compound_pearson_delta": float(c1["compound"]["pearson"] - c0["compound"]["pearson"]), "regime_mae_delta": {regime: float(c1["by_regime"][regime]["mae"] - c0["by_regime"][regime]["mae"]) for regime in REGIMES}})
    gates = {"identity_overlap_zero": audit["maximum_prior_overlap"] == 0 and audit["cross_role_overlap"] == 0, "all_power_and_regime_contracts_hold": True, "C0_C1_parameter_count_exactly_equal": all(run["C0_fixed_bin"]["parameters"] == run["C1_sparse_peak"]["parameters"] for run in runs), "each_seed_C1_overall_and_compound_mae_reduction_vs_C0_at_least_0p05": all(row["overall_mae_reduction"] >= .05 and row["compound_mae_reduction"] >= .05 for row in deltas), "each_seed_C1_overall_and_compound_pearson_delta_vs_C0_at_least_0p05": all(row["overall_pearson_delta"] >= .05 and row["compound_pearson_delta"] >= .05 for row in deltas), "each_seed_C1_ID_control_and_low_count_mae_reduction_vs_constant_at_least_0p10": all(run["C1_sparse_peak"]["evaluation"]["by_regime"][regime]["mae_reduction_vs_constant"] >= .10 for run in runs for regime in ("ID_control", "low_count")), "each_seed_C1_ID_control_and_low_count_pearson_at_least_0p30": all(run["C1_sparse_peak"]["evaluation"]["by_regime"][regime]["pearson"] >= .30 for run in runs for regime in ("ID_control", "low_count")), "no_C1_regime_worse_than_C0_in_all_three_seeds_by_mae_over_0p02": not any(all(row["regime_mae_delta"][regime] > .02 for row in deltas) for regime in REGIMES), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    summary = {"experiment": "P3o-M resolution-aware sparse-delay-event ordinal MFT-count screen", "preregister_sha256": sha256_file(preregister), "generator_sha256": sha256_file(project / "matlab/generate_p3om_sparse_peak_ordinal_data.m"), "identity_audit": audit, "constant_baseline": constant, "runs": runs, "deltas": deltas, "promotion_gates": gates, "development_passed": bool(all(gates.values())), "verdict": "promote_to_frozen_predicted_gate_confirmation" if all(gates.values()) else "stop_no_confirmation_data", "classification_loss_used": False, "oracle_positive_only": True, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates}, indent=2))


if __name__ == "__main__": main()
