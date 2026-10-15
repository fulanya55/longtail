"""P3o-O: matched-capacity adjacent-peak relation MFT-count screen."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3om_data_audit import role_paths
from cj0724.p3om_sparse_peak_ordinal import (
    _collect,
    _read_role,
    _split_fit_selection,
    inverse_delay_channels,
    ordinal_loss,
    sparse_peak_tokens,
)
from cj0724.train import set_seed


MODEL_SEEDS = (2311, 2312, 2313)
TOKEN_COUNT = 16


def adjacent_peak_relation_tokens(channels: np.ndarray, use_relations: bool) -> np.ndarray:
    """Append only observable next-peak relations to frozen sparse-event tokens."""
    peaks = sparse_peak_tokens(channels, TOKEN_COUNT)
    output = np.zeros((len(peaks), TOKEN_COUNT, 8), dtype=np.float32)
    output[:, :, :4] = peaks
    if not use_relations:
        return output
    valid = peaks[:, :, 1] > 0.0
    for index in range(TOKEN_COUNT - 1):
        paired = valid[:, index] & valid[:, index + 1]
        current, following = peaks[:, index], peaks[:, index + 1]
        output[paired, index, 4] = following[paired, 0] - current[paired, 0]
        output[paired, index, 5] = following[paired, 1]
        output[paired, index, 6] = following[paired, 2] - current[paired, 2]
        output[paired, index, 7] = following[paired, 3] - current[paired, 3]
    return output


class PairTokenOrdinalNet(nn.Module):
    """Capacity-matched C0/C1 ordinal CNN over 8-channel peak tokens."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(8, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, 3, padding=1, bias=False), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, 3, padding=1, bias=False), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * TOKEN_COUNT, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 7))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (TOKEN_COUNT, 8):
            raise ValueError(f"Expected [B,{TOKEN_COUNT},8] tokens")
        return self.head(self.encoder(tokens.transpose(1, 2)))


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if not len(prediction):
        raise RuntimeError("P3o-O empty metric subset")
    pearson = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) and np.std(target) else 0.0
    return {"count": int(len(prediction)), "mae": float(np.abs(prediction - target).mean()), "pearson": pearson, "rounded_accuracy": float(np.mean(np.rint(prediction) == target))}


def _evaluate(prediction: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "overall": _metrics(prediction, data["target"]),
        "compound": _metrics(prediction[data["compound"]], data["target"][data["compound"]]),
        "by_regime": {name: _metrics(prediction[data["regime"] == index], data["target"][data["regime"] == index]) for index, name in enumerate(REGIMES)},
        "by_power": {str(power): _metrics(prediction[data["power"] == power], data["target"][data["power"] == power]) for power in POWERS},
    }


def _fit_variant(name: str, use_relations: bool, fit: dict[str, np.ndarray], selection: dict[str, np.ndarray], development: dict[str, np.ndarray], seed: int, args: argparse.Namespace, output: Path) -> dict[str, object]:
    mean = fit["channels"].mean(axis=(0, 2), keepdims=True)
    scale = np.maximum(fit["channels"].std(axis=(0, 2), keepdims=True), 1e-4)
    tokenize = lambda data: adjacent_peak_relation_tokens((data["channels"] - mean) / scale, use_relations)
    fit_x, fit_y = torch.from_numpy(tokenize(fit)), torch.from_numpy(fit["target"])
    selection_x, selection_y = torch.from_numpy(tokenize(selection)), torch.from_numpy(selection["target"])
    development_x = torch.from_numpy(tokenize(development))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(seed); model = PairTokenOrdinalNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    selection_loader = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); loss = ordinal_loss(model(x.to(device)), y.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            selection_loss = float(np.mean([float(ordinal_loss(model(x.to(device)), y.to(device))) for x, y in selection_loader]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_loss": selection_loss})
        if selection_loss < best:
            best, stale = selection_loss, 0; torch.save(model.state_dict(), output / f"{name}_best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device, weights_only=True)); model.eval()
    logits = []
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(development_x), batch_size=args.batch_size): logits.append(model(x.to(device)).cpu().numpy())
    logits = np.concatenate(logits); prediction = (1.0 + (1.0 / (1.0 + np.exp(-logits))).sum(axis=1)).astype(np.float32)
    np.savez_compressed(output / f"{name}_predictions.npz", prediction=prediction, target=development["target"], identity=development["identity"], power=development["power"], regime=development["regime"], compound=development["compound"])
    (output / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"name": name, "seed": seed, "parameters": int(sum(item.numel() for item in model.parameters())), "best_selection_loss": best, "epochs_ran": len(history), "checkpoint_sha256": sha256_file(output / f"{name}_best.pt"), "evaluation": _evaluate(prediction, development)}


def _verify_freeze(project: Path, path: Path) -> dict[str, object]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    expected = {"preregister_sha256": project / "configs/p3oo_adjacent_peak_relation_preregister_20260803.json", "implementation_sha256": Path(__file__), "audit_implementation_sha256": project / "cj0724/p3om_data_audit.py", "generator_sha256": project / "matlab/generate_p3oo_adjacent_peak_relation_level.m"}
    mismatches = {key: {"expected": freeze.get(key), "actual": sha256_file(value)} for key, value in expected.items() if freeze.get(key) != sha256_file(value)}
    if mismatches: raise RuntimeError(f"P3o-O method freeze mismatch: {mismatches}")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", default="D:/CJ0724_rechunked_data"); parser.add_argument("--rechunk-suffix", default="_rechunked_v2"); parser.add_argument("--data-audit", default="outputs/p3oo_adjacent_peak_relation_preflight_20260803/data_audit.json"); parser.add_argument("--output-root", default="outputs/p3oo_adjacent_peak_relation_development_20260803"); parser.add_argument("--method-freeze", default="configs/p3oo_adjacent_peak_relation_method_freeze_20260803.json"); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; root = Path(args.rechunk_root)
    freeze = _verify_freeze(project, project / args.method_freeze)
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    paths = {role: role_paths(root, role, seed, args.rechunk_suffix, "p3oo_adjacent_peak_relation") for role, seed in (("train", 1760000000), ("development", 1770000000))}
    train_meta, train_identity = _read_role(paths["train"], []); development_meta, development_identity = _read_role(paths["development"], [])
    audit = json.loads((project / args.data_audit).read_text(encoding="utf-8"))
    if audit["maximum_prior_overlap"] or audit["cross_role_overlap"] or not audit["power_alignment_verified"] or not audit["label_contract_verified"]: raise RuntimeError("P3o-O pre-training audit failed")
    if audit["train_unique"] != len(np.unique(train_identity)) or audit["development_unique"] != len(np.unique(development_identity)): raise RuntimeError("P3o-O audit count mismatch")
    if not np.array_equal(train_meta["ID_control"]["labels"], development_meta["ID_control"]["labels"]): raise RuntimeError("P3o-O label contract mismatch")
    split = _split_fit_selection(train_meta["ID_control"]["labels"], 20260803)
    fit, selection, development = _collect(paths["train"], train_meta, split["fit"]), _collect(paths["train"], train_meta, split["selection"]), _collect(paths["development"], development_meta, None)
    runs = []
    for seed in MODEL_SEEDS:
        seed_output = output / f"seed_{seed}"; seed_output.mkdir(); started = time.perf_counter()
        c0 = _fit_variant("C0_sparse_zero_relation", False, fit, selection, development, seed, args, seed_output)
        c1 = _fit_variant("C1_adjacent_peak_relation", True, fit, selection, development, seed, args, seed_output)
        if c0["parameters"] != c1["parameters"]: raise RuntimeError("P3o-O capacity contract failed")
        runs.append({"seed": seed, "seconds": float(time.perf_counter() - started), "C0": c0, "C1": c1})
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    deltas = []
    for run in runs:
        c0, c1 = run["C0"]["evaluation"], run["C1"]["evaluation"]
        deltas.append({"seed": run["seed"], "overall_mae_reduction": 1.0 - c1["overall"]["mae"] / c0["overall"]["mae"], "compound_mae_reduction": 1.0 - c1["compound"]["mae"] / c0["compound"]["mae"], "overall_pearson_delta": c1["overall"]["pearson"] - c0["overall"]["pearson"], "compound_pearson_delta": c1["compound"]["pearson"] - c0["compound"]["pearson"], "id_control_pearson": c1["by_regime"]["ID_control"]["pearson"], "low_count_pearson": c1["by_regime"]["low_count"]["pearson"], "ultra_close_mae_delta": c1["by_regime"]["ultra_close"]["mae"] - c0["by_regime"]["ultra_close"]["mae"]})
    gates = {"identity_overlap_zero": True, "all_power_and_parameter_contracts_hold": True, "C0_C1_parameter_count_exactly_equal": all(run["C0"]["parameters"] == run["C1"]["parameters"] for run in runs), "each_seed_C1_overall_and_compound_mae_reduction_vs_C0_at_least_0p05": all(item["overall_mae_reduction"] >= .05 and item["compound_mae_reduction"] >= .05 for item in deltas), "each_seed_C1_overall_and_compound_pearson_delta_vs_C0_at_least_0p03": all(item["overall_pearson_delta"] >= .03 and item["compound_pearson_delta"] >= .03 for item in deltas), "each_seed_C1_ID_control_and_low_count_pearson_at_least_0p30": all(item["id_control_pearson"] >= .3 and item["low_count_pearson"] >= .3 for item in deltas), "no_C1_regime_worse_than_C0_in_all_three_seeds_by_mae_over_0p02": all(not all(run["C1"]["evaluation"]["by_regime"][name]["mae"] - run["C0"]["evaluation"]["by_regime"][name]["mae"] > .02 for run in runs) for name in REGIMES), "each_seed_C1_ultra_close_mae_not_worse_than_C0": all(item["ultra_close_mae_delta"] <= 0.0 for item in deltas), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    summary = {"experiment": "P3o-O adjacent-peak relation ordinal MFT-count screen", "method_freeze": freeze, "runs": runs, "deltas": deltas, "promotion_gates": gates, "development_passed": bool(all(gates.values())), "verdict": "promote_to_two_identity_confirmation" if all(gates.values()) else "stop_no_confirmation_data", "classification_loss_used": False, "oracle_positive_only": True, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates}, indent=2))


if __name__ == "__main__": main()
