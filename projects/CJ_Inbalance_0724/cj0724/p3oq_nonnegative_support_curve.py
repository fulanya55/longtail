"""P3o-Q: global nonnegative in-phase support curve versus zero-curve control."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cj0724.p2w_isrj_width_oracle_ood import POWERS, SIGNAL_LENGTH, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import REGIMES, reference_signal
from cj0724.p3om_data_audit import role_paths
from cj0724.p3om_sparse_peak_ordinal import _read_role, _split_fit_selection, ordinal_loss, sparse_peak_tokens
from cj0724.p3oo_adjacent_peak_relation import PairTokenOrdinalNet, _evaluate
from cj0724.p3op_complex_coherence import _collect_raw_iq
from cj0724.train import set_seed


MODEL_SEEDS = (2331, 2332, 2333)
TOKEN_COUNT, SEARCH_START, SEARCH_STOP = 16, 350, 1101


def inverse_delay_magnitude_channels(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return P3o-M-compatible magnitude channels and the complex full window."""
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1:] != (2, SIGNAL_LENGTH): raise ValueError(f"iq must be [N,2,{SIGNAL_LENGTH}]")
    signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64); spectrum = np.fft.fft(reference_signal()); power = np.abs(spectrum) ** 2
    impulse = np.fft.ifft(np.fft.fft(signal, axis=1) * spectrum.conj()[None, :] / (power[None, :] + 1e-6 * float(power.max())), axis=1)[:, SEARCH_START:SEARCH_STOP]
    magnitude = np.abs(impulse); maximum = np.maximum(magnitude.max(axis=1, keepdims=True), 1e-12); normalized = magnitude / maximum
    from scipy.ndimage import uniform_filter1d
    contrast = np.maximum(normalized - uniform_filter1d(normalized, 31, axis=1, mode="nearest"), 0.0); contrast /= np.maximum(contrast.max(axis=1, keepdims=True), 1e-12)
    local = uniform_filter1d(normalized, 9, axis=1, mode="nearest"); sharpness = np.clip(normalized / np.maximum(local, 1e-4), 0.0, 10.0) / 10.0; position = np.broadcast_to(np.linspace(0.0, 1.0, normalized.shape[1])[None, :], normalized.shape)
    return np.stack((normalized, contrast, sharpness, position), axis=1).astype(np.float32), impulse


def nonnegative_support_curve(impulse: np.ndarray) -> np.ndarray:
    """Eight candidate-count features from all positive in-phase support samples."""
    impulse = np.asarray(impulse, dtype=np.complex128); support = np.maximum(impulse.real, 0.0)
    top = np.sort(support, axis=1)[:, -8:][:, ::-1]
    if top.shape[1] < 8: top = np.pad(top, ((0, 0), (0, 8 - top.shape[1])))
    total_complex = np.maximum(np.square(np.abs(impulse)).sum(axis=1), 1e-12); total_positive = np.maximum(support.sum(axis=1), 1e-12); cumulative_square = np.cumsum(np.square(top), axis=1); cumulative = np.cumsum(top, axis=1)
    return np.stack((np.maximum(total_complex[:, None] - cumulative_square, 0.0) / total_complex[:, None], cumulative / total_positive[:, None], top / np.maximum(top[:, :1], 1e-12), top / np.sqrt(total_complex[:, None])), axis=2).astype(np.float32)


def support_curve_tokens(base_channels: np.ndarray, curve: np.ndarray, use_curve: bool) -> np.ndarray:
    base = sparse_peak_tokens(base_channels, TOKEN_COUNT); output = np.zeros((len(base), TOKEN_COUNT, 8), dtype=np.float32); output[:, :, :4] = base
    if use_curve: output[:, :8, 4:] = curve
    return output


def _prepare(data: dict, mean: np.ndarray, scale: np.ndarray, use_curve: bool) -> np.ndarray:
    channels, impulse = inverse_delay_magnitude_channels(data["channels"])
    return support_curve_tokens((channels - mean) / scale, nonnegative_support_curve(impulse), use_curve)


def _fit(name: str, use_curve: bool, fit: dict, selection: dict, development: dict, seed: int, args: argparse.Namespace, output: Path) -> dict:
    fit_base, _ = inverse_delay_magnitude_channels(fit["channels"]); mean = fit_base.mean(axis=(0, 2), keepdims=True); scale = np.maximum(fit_base.std(axis=(0, 2), keepdims=True), 1e-4)
    fit_x, fit_y = torch.from_numpy(_prepare(fit, mean, scale, use_curve)), torch.from_numpy(fit["target"]); selection_x, selection_y = torch.from_numpy(_prepare(selection, mean, scale, use_curve)), torch.from_numpy(selection["target"]); development_x = torch.from_numpy(_prepare(development, mean, scale, use_curve))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu"); set_seed(seed); model = PairTokenOrdinalNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)); early = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); loss = ordinal_loss(model(x.to(device)), y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): score = float(np.mean([float(ordinal_loss(model(x.to(device)), y.to(device))) for x, y in early]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_loss": score})
        if score < best: best, stale = score, 0; torch.save(model.state_dict(), output / f"{name}_best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device, weights_only=True)); model.eval(); logits = []
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(development_x), batch_size=args.batch_size): logits.append(model(x.to(device)).cpu().numpy())
    prediction = (1 + (1 / (1 + np.exp(-np.concatenate(logits)))).sum(axis=1)).astype(np.float32); np.savez_compressed(output / f"{name}_predictions.npz", prediction=prediction, target=development["target"], identity=development["identity"], power=development["power"], regime=development["regime"], compound=development["compound"]); (output / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"name": name, "seed": seed, "parameters": int(sum(x.numel() for x in model.parameters())), "best_selection_loss": best, "epochs_ran": len(history), "checkpoint_sha256": sha256_file(output / f"{name}_best.pt"), "evaluation": _evaluate(prediction, development)}


def _verify(project: Path, freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")); expected = {"preregister_sha256": project / "configs/p3oq_nonnegative_support_curve_preregister_20260803.json", "implementation_sha256": Path(__file__), "audit_implementation_sha256": project / "cj0724/p3om_data_audit.py", "generator_sha256": project / "matlab/generate_p3oq_nonnegative_support_curve_level.m", "p3om_dependency_sha256": project / "cj0724/p3om_sparse_peak_ordinal.py", "p3oo_model_dependency_sha256": project / "cj0724/p3oo_adjacent_peak_relation.py", "p3op_raw_iq_dependency_sha256": project / "cj0724/p3op_complex_coherence.py"}; mismatch = {key: {"expected": freeze.get(key), "actual": sha256_file(value)} for key, value in expected.items() if freeze.get(key) != sha256_file(value)}
    if mismatch: raise RuntimeError(f"P3o-Q method freeze mismatch: {mismatch}")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", default="D:/CJ0724_rechunked_data"); parser.add_argument("--rechunk-suffix", default="_rechunked_v2"); parser.add_argument("--data-audit", default="outputs/p3oq_nonnegative_support_curve_preflight_20260803/data_audit.json"); parser.add_argument("--output-root", default="outputs/p3oq_nonnegative_support_curve_development_20260803"); parser.add_argument("--method-freeze", default="configs/p3oq_nonnegative_support_curve_method_freeze_20260803.json"); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; root = Path(args.rechunk_root); freeze = _verify(project, project / args.method_freeze); output = project / args.output_root; output.mkdir(parents=True, exist_ok=False); paths = {role: role_paths(root, role, seed, args.rechunk_suffix, "p3oq_nonnegative_support_curve") for role, seed in (("train", 1840000000), ("development", 1850000000))}; train_meta, train_id = _read_role(paths["train"], []); dev_meta, dev_id = _read_role(paths["development"], [])
    audit = json.loads((project / args.data_audit).read_text(encoding="utf-8"))
    if audit["maximum_prior_overlap"] or audit["cross_role_overlap"] or not audit["power_alignment_verified"] or not audit["label_contract_verified"]: raise RuntimeError("P3o-Q pre-training audit failed")
    if audit["train_unique"] != len(np.unique(train_id)) or audit["development_unique"] != len(np.unique(dev_id)): raise RuntimeError("P3o-Q audit count mismatch")
    if not np.array_equal(train_meta["ID_control"]["labels"], dev_meta["ID_control"]["labels"]): raise RuntimeError("P3o-Q label contract mismatch")
    split = _split_fit_selection(train_meta["ID_control"]["labels"], 20260803); fit, selection, development = _collect_raw_iq(paths["train"], train_meta, split["fit"]), _collect_raw_iq(paths["train"], train_meta, split["selection"]), _collect_raw_iq(paths["development"], dev_meta, None); runs = []
    for seed in MODEL_SEEDS:
        folder = output / f"seed_{seed}"; folder.mkdir(); started = time.perf_counter(); c0 = _fit("C0_sparse_zero_curve", False, fit, selection, development, seed, args, folder); c1 = _fit("C1_global_nonnegative_curve", True, fit, selection, development, seed, args, folder)
        if c0["parameters"] != c1["parameters"]: raise RuntimeError("P3o-Q capacity contract failed")
        runs.append({"seed": seed, "seconds": float(time.perf_counter() - started), "C0": c0, "C1": c1});
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    deltas = []
    for run in runs:
        c0, c1 = run["C0"]["evaluation"], run["C1"]["evaluation"]; deltas.append({"seed": run["seed"], "overall_mae_reduction": 1-c1["overall"]["mae"]/c0["overall"]["mae"], "compound_mae_reduction": 1-c1["compound"]["mae"]/c0["compound"]["mae"], "overall_pearson_delta": c1["overall"]["pearson"]-c0["overall"]["pearson"], "compound_pearson_delta": c1["compound"]["pearson"]-c0["compound"]["pearson"], "id_control_pearson": c1["by_regime"]["ID_control"]["pearson"], "low_count_pearson": c1["by_regime"]["low_count"]["pearson"], "ultra_close_mae_delta": c1["by_regime"]["ultra_close"]["mae"]-c0["by_regime"]["ultra_close"]["mae"]})
    gates = {"identity_overlap_zero": True, "all_power_and_parameter_contracts_hold": True, "C0_C1_parameter_count_exactly_equal": all(row["C0"]["parameters"] == row["C1"]["parameters"] for row in runs), "each_seed_C1_overall_and_compound_mae_reduction_vs_C0_at_least_0p05": all(row["overall_mae_reduction"] >= .05 and row["compound_mae_reduction"] >= .05 for row in deltas), "each_seed_C1_overall_and_compound_pearson_delta_vs_C0_at_least_0p03": all(row["overall_pearson_delta"] >= .03 and row["compound_pearson_delta"] >= .03 for row in deltas), "each_seed_C1_ID_control_and_low_count_pearson_at_least_0p30": all(row["id_control_pearson"] >= .3 and row["low_count_pearson"] >= .3 for row in deltas), "no_C1_regime_worse_than_C0_in_all_three_seeds_by_mae_over_0p02": all(not all(row["C1"]["evaluation"]["by_regime"][name]["mae"]-row["C0"]["evaluation"]["by_regime"][name]["mae"] > .02 for row in runs) for name in REGIMES), "each_seed_C1_ultra_close_mae_not_worse_than_C0": all(row["ultra_close_mae_delta"] <= 0 for row in deltas), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    summary = {"experiment": "P3o-Q global nonnegative in-phase support-curve MFT-count screen", "method_freeze": freeze, "runs": runs, "deltas": deltas, "promotion_gates": gates, "development_passed": bool(all(gates.values())), "verdict": "promote_to_two_identity_confirmation" if all(gates.values()) else "stop_no_confirmation_data", "classification_loss_used": False, "oracle_positive_only": True, "test_generated": False, "test_opened": False, "locked_test_used": False}; (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates}, indent=2))


if __name__ == "__main__": main()
