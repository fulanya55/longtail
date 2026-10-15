"""P3o-P: complex-coherence versus magnitude-only sparse-event MFT count."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader, TensorDataset

from cj0724.p2w_isrj_width_oracle_ood import POWERS, SIGNAL_LENGTH, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import REGIMES, reference_signal
from cj0724.p3om_data_audit import role_paths
from cj0724.p3om_sparse_peak_ordinal import _load_iq, _read_role, _split_fit_selection, ordinal_loss, sparse_peak_tokens
from cj0724.p3oo_adjacent_peak_relation import PairTokenOrdinalNet, _evaluate
from cj0724.train import set_seed


MODEL_SEEDS = (2321, 2322, 2323)
TOKEN_COUNT = 16
SEARCH_START, SEARCH_STOP = 350, 1101


def complex_inverse_delay_channels(iq: np.ndarray) -> np.ndarray:
    """Magnitude channels plus phase-coherent inverse-delay descriptors."""
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH):
        raise ValueError(f"iq must be [N,2,{SIGNAL_LENGTH}]")
    signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    reference = np.fft.fft(reference_signal()); power = np.abs(reference) ** 2
    impulse = np.fft.ifft(np.fft.fft(signal, axis=1) * reference.conj()[None, :] / (power[None, :] + 1e-6 * float(power.max())), axis=1)[:, SEARCH_START:SEARCH_STOP]
    magnitude = np.abs(impulse); maximum = np.maximum(magnitude.max(axis=1, keepdims=True), 1e-12); normalized = magnitude / maximum
    contrast = np.maximum(normalized - uniform_filter1d(normalized, 31, axis=1, mode="nearest"), 0.0); contrast /= np.maximum(contrast.max(axis=1, keepdims=True), 1e-12)
    local = uniform_filter1d(normalized, 9, axis=1, mode="nearest"); sharpness = np.clip(normalized / np.maximum(local, 1e-4), 0.0, 10.0) / 10.0
    position = np.broadcast_to(np.linspace(0.0, 1.0, normalized.shape[1], dtype=np.float64)[None, :], normalized.shape)
    local_real = uniform_filter1d(impulse.real, 5, axis=1, mode="nearest"); local_imag = uniform_filter1d(impulse.imag, 5, axis=1, mode="nearest"); local_magnitude = uniform_filter1d(magnitude, 5, axis=1, mode="nearest")
    coherence = np.hypot(local_real, local_imag) / np.maximum(local_magnitude, 1e-12)
    inphase_coherence = local_real / np.maximum(local_magnitude, 1e-12)
    return np.stack((normalized, contrast, sharpness, position, impulse.real / maximum, impulse.imag / maximum, coherence, inphase_coherence), axis=1).astype(np.float32)


def complex_coherence_tokens(channels: np.ndarray, use_coherence: bool, position_mean: float = 0.0, position_scale: float = 1.0) -> np.ndarray:
    """Keep P3o-M peak locations fixed; only C1 receives complex descriptors."""
    if channels.ndim != 3 or channels.shape[1] != 8: raise ValueError("channels must be [N,8,L]")
    base = sparse_peak_tokens(channels[:, :4], TOKEN_COUNT); output = np.zeros((len(base), TOKEN_COUNT, 8), dtype=np.float32); output[:, :, :4] = base
    if not use_coherence: return output
    raw_position = base[:, :, 3] * float(position_scale) + float(position_mean)
    locations = np.clip(np.rint(raw_position * (channels.shape[2] - 1)).astype(np.int64), 0, channels.shape[2] - 1); valid = base[:, :, 1] > 0.0
    rows, tokens = np.nonzero(valid); output[rows, tokens, 4:] = channels[rows, 4:, locations[rows, tokens]]
    return output


def _collect_raw_iq(paths: dict, metadata: dict, role: np.ndarray | None) -> dict:
    """Use P3o-M's audited indices but retain raw I/Q for complex deconvolution."""
    output = {name: [] for name in ("channels", "target", "identity", "power", "regime", "compound")}
    for regime_index, regime in enumerate(REGIMES):
        record = metadata[regime]
        selected = record["selected"] if role is None else np.intersect1d(record["selected"], role)
        target = record["values"][28, selected].astype(np.float32)
        for power in POWERS:
            output["channels"].append(_load_iq(paths[regime][power], selected))
            output["target"].append(target); output["identity"].append(record["identities"][selected])
            output["power"].append(np.full(len(selected), power, dtype=np.int16)); output["regime"].append(np.full(len(selected), regime_index, dtype=np.int8))
            output["compound"].append(record["labels"][selected].sum(axis=1) >= 2)
    return {name: np.concatenate(values) for name, values in output.items()}


def _fit(name: str, use_coherence: bool, fit: dict, selection: dict, development: dict, seed: int, args: argparse.Namespace, output: Path) -> dict:
    mean = fit["channels"].mean(axis=(0, 2), keepdims=True); scale = np.maximum(fit["channels"].std(axis=(0, 2), keepdims=True), 1e-4)
    token = lambda value: complex_coherence_tokens((value["channels"] - mean) / scale, use_coherence, float(mean[0, 3, 0]), float(scale[0, 3, 0]))
    fit_x, fit_y = torch.from_numpy(token(fit)), torch.from_numpy(fit["target"]); selection_x, selection_y = torch.from_numpy(token(selection)), torch.from_numpy(selection["target"]); development_x = torch.from_numpy(token(development))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu"); set_seed(seed); model = PairTokenOrdinalNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)); select_loader = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); loss = ordinal_loss(model(x.to(device)), y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): selection_loss = float(np.mean([float(ordinal_loss(model(x.to(device)), y.to(device))) for x, y in select_loader]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_loss": selection_loss})
        if selection_loss < best: best, stale = selection_loss, 0; torch.save(model.state_dict(), output / f"{name}_best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device, weights_only=True)); model.eval(); logits = []
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(development_x), batch_size=args.batch_size): logits.append(model(x.to(device)).cpu().numpy())
    prediction = (1.0 + (1.0 / (1.0 + np.exp(-np.concatenate(logits)))).sum(axis=1)).astype(np.float32)
    np.savez_compressed(output / f"{name}_predictions.npz", prediction=prediction, target=development["target"], identity=development["identity"], power=development["power"], regime=development["regime"], compound=development["compound"])
    (output / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"name": name, "seed": seed, "parameters": int(sum(value.numel() for value in model.parameters())), "best_selection_loss": best, "epochs_ran": len(history), "checkpoint_sha256": sha256_file(output / f"{name}_best.pt"), "evaluation": _evaluate(prediction, development)}


def _verify(project: Path, freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")); expected = {"preregister_sha256": project / "configs/p3op_complex_coherence_preregister_20260803.json", "implementation_sha256": Path(__file__), "audit_implementation_sha256": project / "cj0724/p3om_data_audit.py", "generator_sha256": project / "matlab/generate_p3op_complex_coherence_level.m", "p3om_dependency_sha256": project / "cj0724/p3om_sparse_peak_ordinal.py", "p3oo_model_dependency_sha256": project / "cj0724/p3oo_adjacent_peak_relation.py"}
    mismatch = {name: {"expected": freeze.get(name), "actual": sha256_file(path)} for name, path in expected.items() if freeze.get(name) != sha256_file(path)}
    if mismatch: raise RuntimeError(f"P3o-P method freeze mismatch: {mismatch}")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", default="D:/CJ0724_rechunked_data"); parser.add_argument("--rechunk-suffix", default="_rechunked_v2"); parser.add_argument("--data-audit", default="outputs/p3op_complex_coherence_preflight_20260803/data_audit.json"); parser.add_argument("--output-root", default="outputs/p3op_complex_coherence_development_20260803"); parser.add_argument("--method-freeze", default="configs/p3op_complex_coherence_method_freeze_20260803.json"); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; root = Path(args.rechunk_root); freeze = _verify(project, project / args.method_freeze); output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    paths = {role: role_paths(root, role, seed, args.rechunk_suffix, "p3op_complex_coherence") for role, seed in (("train", 1800000000), ("development", 1810000000))}; train_meta, train_identity = _read_role(paths["train"], []); development_meta, development_identity = _read_role(paths["development"], [])
    audit = json.loads((project / args.data_audit).read_text(encoding="utf-8"))
    if audit["maximum_prior_overlap"] or audit["cross_role_overlap"] or not audit["power_alignment_verified"] or not audit["label_contract_verified"]: raise RuntimeError("P3o-P pre-training audit failed")
    if audit["train_unique"] != len(np.unique(train_identity)) or audit["development_unique"] != len(np.unique(development_identity)): raise RuntimeError("P3o-P audit count mismatch")
    if not np.array_equal(train_meta["ID_control"]["labels"], development_meta["ID_control"]["labels"]): raise RuntimeError("P3o-P label contract mismatch")
    split = _split_fit_selection(train_meta["ID_control"]["labels"], 20260803); fit, selection, development = _collect_raw_iq(paths["train"], train_meta, split["fit"]), _collect_raw_iq(paths["train"], train_meta, split["selection"]), _collect_raw_iq(paths["development"], development_meta, None)
    for data in (fit, selection, development): data["channels"] = complex_inverse_delay_channels(data.pop("channels"))
    runs = []
    for seed in MODEL_SEEDS:
        folder = output / f"seed_{seed}"; folder.mkdir(); started = time.perf_counter(); c0 = _fit("C0_magnitude_zero_coherence", False, fit, selection, development, seed, args, folder); c1 = _fit("C1_complex_coherence", True, fit, selection, development, seed, args, folder)
        if c0["parameters"] != c1["parameters"]: raise RuntimeError("P3o-P capacity contract failed")
        runs.append({"seed": seed, "seconds": float(time.perf_counter() - started), "C0": c0, "C1": c1});
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    deltas = []
    for run in runs:
        c0, c1 = run["C0"]["evaluation"], run["C1"]["evaluation"]
        deltas.append({"seed": run["seed"], "overall_mae_reduction": 1 - c1["overall"]["mae"] / c0["overall"]["mae"], "compound_mae_reduction": 1 - c1["compound"]["mae"] / c0["compound"]["mae"], "overall_pearson_delta": c1["overall"]["pearson"] - c0["overall"]["pearson"], "compound_pearson_delta": c1["compound"]["pearson"] - c0["compound"]["pearson"], "id_control_pearson": c1["by_regime"]["ID_control"]["pearson"], "low_count_pearson": c1["by_regime"]["low_count"]["pearson"], "ultra_close_mae_delta": c1["by_regime"]["ultra_close"]["mae"] - c0["by_regime"]["ultra_close"]["mae"]})
    gates = {"identity_overlap_zero": True, "all_power_and_parameter_contracts_hold": True, "C0_C1_parameter_count_exactly_equal": all(run["C0"]["parameters"] == run["C1"]["parameters"] for run in runs), "each_seed_C1_overall_and_compound_mae_reduction_vs_C0_at_least_0p05": all(row["overall_mae_reduction"] >= .05 and row["compound_mae_reduction"] >= .05 for row in deltas), "each_seed_C1_overall_and_compound_pearson_delta_vs_C0_at_least_0p03": all(row["overall_pearson_delta"] >= .03 and row["compound_pearson_delta"] >= .03 for row in deltas), "each_seed_C1_ID_control_and_low_count_pearson_at_least_0p30": all(row["id_control_pearson"] >= .3 and row["low_count_pearson"] >= .3 for row in deltas), "no_C1_regime_worse_than_C0_in_all_three_seeds_by_mae_over_0p02": all(not all(run["C1"]["evaluation"]["by_regime"][name]["mae"] - run["C0"]["evaluation"]["by_regime"][name]["mae"] > .02 for run in runs) for name in REGIMES), "each_seed_C1_ultra_close_mae_not_worse_than_C0": all(row["ultra_close_mae_delta"] <= 0 for row in deltas), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    summary = {"experiment": "P3o-P complex-coherence sparse-event ordinal MFT-count screen", "method_freeze": freeze, "runs": runs, "deltas": deltas, "promotion_gates": gates, "development_passed": bool(all(gates.values())), "verdict": "promote_to_two_identity_confirmation" if all(gates.values()) else "stop_no_confirmation_data", "classification_loss_used": False, "oracle_positive_only": True, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates}, indent=2))


if __name__ == "__main__": main()
