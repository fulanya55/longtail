"""P3o-R: test whether MFT component separation preserves count information."""

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
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import (
    AMPLITUDE_COLUMNS,
    COUNT_COLUMN,
    DELAY_COLUMNS,
    MFT_CLASS_INDEX,
    REGIMES,
    CONSTANT_COUNT,
    count_metrics,
    estimate_sparse_count,
    reconstruct_mft_component,
)
from cj0724.p3om_data_audit import read_role, role_paths
from cj0724.p3om_sparse_peak_ordinal import _split_fit_selection
from cj0724.train import set_seed


MODEL_SEEDS = (2341, 2342, 2343)


def _identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


class MFTComponentDataset(Dataset):
    """All regime/power compound views paired to an exact MFT-only teacher."""

    def __init__(self, paths: dict[str, dict[int, Path]], role_indices: np.ndarray) -> None:
        self.paths = paths
        self.role_indices = np.asarray(role_indices, dtype=np.int64)
        self.views: dict[str, dict[int, V73IQDataset]] = {}
        self.entries: list[tuple[str, int, np.ndarray, float, int, int]] = []
        for regime in REGIMES:
            files = paths[regime]
            with h5py.File(files[POWERS[0]], "r") as handle:
                labels = np.asarray(handle["label_selected"], dtype=np.uint8).T
                values = np.asarray(handle["parameter_values"], dtype=np.float64)
                masks = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
                identities = np.rint(np.asarray(handle["receiver_noise_seed"], dtype=np.float64).reshape(-1)).astype(np.int64)
            selected = self.role_indices[
                (labels[self.role_indices, MFT_CLASS_INDEX] > 0)
                & masks[COUNT_COLUMN, self.role_indices]
                & np.isfinite(values[COUNT_COLUMN, self.role_indices])
            ]
            if selected.size == 0:
                raise RuntimeError(f"No valid MFT positives for {regime}")
            delays = values[DELAY_COLUMNS[:, None], selected[None, :]].T
            amplitudes = values[AMPLITUDE_COLUMNS[:, None], selected[None, :]].T
            targets = reconstruct_mft_component(delays, amplitudes)
            counts = values[COUNT_COLUMN, selected]
            if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(counts)):
                raise RuntimeError(f"Non-finite MFT teacher in {regime}")
            for position, sample_index in enumerate(selected):
                self.entries.append((
                    regime,
                    int(sample_index),
                    targets[position],
                    float(counts[position]),
                    int(labels[sample_index].sum()),
                    int(identities[sample_index]),
                ))
            all_indices = np.arange(len(labels), dtype=np.int64)
            self.views[regime] = {
                power: V73IQDataset(files[power], all_indices) for power in POWERS
            }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int | str]:
        regime, sample_index, target, count, combination_size, identity = self.entries[int(position)]
        views = [self.views[regime][power][sample_index] for power in POWERS]
        if any(int(view["sample_index"]) != sample_index for view in views):
            raise RuntimeError("MFT component power-view index mismatch")
        return {
            "iq": torch.stack([view["iq"] for view in views]),
            "target_iq": torch.from_numpy(target.copy()),
            "target_count": torch.tensor(count, dtype=torch.float32),
            "sample_index": sample_index,
            "receiver_noise_seed": identity,
            "combination_size": combination_size,
            "regime": REGIMES.index(regime),
        }

    def close(self) -> None:
        for by_power in self.views.values():
            for dataset in by_power.values():
                dataset.close()


def waveform_objective(
    separator: ConditionalComplexUNetSeparator,
    iq: torch.Tensor,
    target_iq: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    batch_size, power_count = iq.shape[:2]
    flat_iq = iq.reshape(batch_size * power_count, *iq.shape[2:])
    flat_target = target_iq[:, None].expand(-1, power_count, -1, -1).reshape_as(flat_iq)
    estimate = separator.forward_components(flat_iq, None)["normalized_estimate"].float()
    cosine = 1.0 - F.cosine_similarity(estimate.flatten(1), flat_target.flatten(1), dim=1).mean()
    time_l1 = F.l1_loss(estimate, flat_target)
    spectral = multi_resolution_spectral_loss(estimate, flat_target)
    total = cosine + 0.2 * time_l1 + 0.25 * spectral
    return total, {"cosine_loss": cosine.detach(), "time_l1": time_l1.detach(), "spectral": spectral.detach()}, estimate


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def train_one(
    seed: int,
    train_paths: dict[str, dict[int, Path]],
    roles: dict[str, np.ndarray],
    output: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConditionalComplexUNetSeparator, list[dict[str, object]], float]:
    set_seed(seed)
    model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    datasets = {name: MFTComponentDataset(train_paths, roles[name]) for name in ("fit", "selection")}
    loaders = {
        "fit": DataLoader(datasets["fit"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda"),
        "selection": DataLoader(datasets["selection"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
    }
    started, history, best, stale = time.perf_counter(), [], float("inf"), 0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_rows = []
            for batch in loaders["fit"]:
                optimizer.zero_grad(set_to_none=True)
                total, parts, _ = waveform_objective(model, batch["iq"].to(device), batch["target_iq"].to(device))
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
                train_rows.append({"total": float(total.detach()), **{key: float(value) for key, value in parts.items()}})
            model.eval()
            selection_rows = []
            with torch.inference_mode():
                for batch in loaders["selection"]:
                    total, parts, _ = waveform_objective(model, batch["iq"].to(device), batch["target_iq"].to(device))
                    selection_rows.append({"total": float(total), **{key: float(value) for key, value in parts.items()}})
            train_epoch, selection_epoch = _mean(train_rows), _mean(selection_rows)
            history.append({"epoch": epoch, "fit": train_epoch, "selection": selection_epoch})
            if selection_epoch["total"] < best:
                best, stale = selection_epoch["total"], 0
                torch.save(model.state_dict(), output / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    finally:
        for dataset in datasets.values():
            dataset.close()
    return model, history, float(time.perf_counter() - started)


@torch.inference_mode()
def evaluate(
    model: ConditionalComplexUNetSeparator,
    paths: dict[str, dict[int, Path]],
    role_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    dataset = MFTComponentDataset(paths, role_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in ("raw", "separated", "exact", "target", "identity", "power", "regime", "compound", "cosine")}
    try:
        for batch in loader:
            iq = batch["iq"].to(device)
            target_iq = batch["target_iq"].to(device)
            batch_size_actual, power_count = iq.shape[:2]
            flat_iq = iq.reshape(batch_size_actual * power_count, *iq.shape[2:])
            estimate = model.forward_components(flat_iq, None)["normalized_estimate"].float()
            target_flat = target_iq[:, None].expand(-1, power_count, -1, -1).reshape_as(estimate)
            raw_prediction = estimate_sparse_count(flat_iq.cpu().numpy())["prediction"]
            separated_prediction = estimate_sparse_count(estimate.cpu().numpy())["prediction"]
            exact_prediction = estimate_sparse_count(target_flat.cpu().numpy())["prediction"]
            arrays["raw"].append(raw_prediction)
            arrays["separated"].append(separated_prediction)
            arrays["exact"].append(exact_prediction)
            arrays["target"].append(np.repeat(batch["target_count"].numpy(), power_count))
            arrays["identity"].append(np.repeat(batch["receiver_noise_seed"].numpy(), power_count))
            arrays["power"].append(np.tile(np.asarray(POWERS, dtype=np.int16), batch_size_actual))
            arrays["regime"].append(np.repeat(batch["regime"].numpy(), power_count))
            arrays["compound"].append(np.repeat((batch["combination_size"].numpy() >= 2).astype(np.uint8), power_count))
            arrays["cosine"].append(F.cosine_similarity(estimate.flatten(1), target_flat.flatten(1), dim=1).cpu().numpy())
    finally:
        dataset.close()
    merged = {name: np.concatenate(parts) for name, parts in arrays.items()}
    metrics: dict[str, object] = {"mean_waveform_cosine": float(np.mean(merged["cosine"]))}
    for name in ("raw", "separated", "exact"):
        metrics[name] = {
            "overall": count_metrics(merged[name], merged["target"]),
            "compound": count_metrics(merged[name][merged["compound"].astype(bool)], merged["target"][merged["compound"].astype(bool)]),
            "by_regime": {
                regime: {
                    "overall": count_metrics(merged[name][merged["regime"] == index], merged["target"][merged["regime"] == index]),
                    "compound": count_metrics(merged[name][(merged["regime"] == index) & merged["compound"].astype(bool)], merged["target"][(merged["regime"] == index) & merged["compound"].astype(bool)]),
                }
                for index, regime in enumerate(REGIMES)
            },
        }
    return {"arrays": merged, "metrics": metrics}


def _gate_report(run: dict[str, object]) -> dict[str, bool]:
    metrics = run["metrics"]
    assert isinstance(metrics, dict)
    result = {"mean_waveform_cosine_at_least_0p50": bool(metrics["mean_waveform_cosine"] >= 0.50)}
    for regime in REGIMES:
        raw = metrics["raw"]["by_regime"][regime]
        separated = metrics["separated"]["by_regime"][regime]
        for subset in ("overall", "compound"):
            raw_mae = raw[subset]["mae"]
            separated_metric = separated[subset]
            result[f"{regime}_{subset}_mae_reduction_at_least_0p10"] = bool(1.0 - separated_metric["mae"] / max(raw_mae, 1.0e-12) >= 0.10)
            result[f"{regime}_{subset}_pearson_at_least_0p30"] = bool(separated_metric["pearson"] >= 0.30)
    raw_ultra = metrics["raw"]["by_regime"]["ultra_close"]["overall"]["mae"]
    separated_ultra = metrics["separated"]["by_regime"]["ultra_close"]["overall"]["mae"]
    result["ultra_close_overall_mae_not_worse"] = bool(separated_ultra <= raw_ultra)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", default="D:/CJ0724_rechunked_data")
    parser.add_argument("--rechunk-suffix", default="_rechunked_v2")
    parser.add_argument("--data-audit", default="outputs/p3or_mft_component_separation_preflight_20260803/data_audit.json")
    parser.add_argument("--method-freeze", default="configs/p3or_mft_component_separation_method_freeze_20260803.json")
    parser.add_argument("--output-root", default="outputs/p3or_mft_component_separation_development_20260803")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = project / args.output_root
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((project / args.data_audit).read_text(encoding="utf-8"))
    if audit["maximum_prior_overlap"] or audit["cross_role_overlap"] or not audit["power_alignment_verified"] or not audit["label_contract_verified"]:
        raise RuntimeError("P3o-R data audit failed")
    freeze = json.loads((project / args.method_freeze).read_text(encoding="utf-8"))
    expected = {
        "preregister_sha256": project / "configs/p3or_mft_component_separation_preregister_20260803.json",
        "implementation_sha256": Path(__file__),
        "generator_sha256": project / "matlab/generate_p3or_mft_component_separation_level.m",
        "p3c_oracle_sha256": project / "cj0724/p3c_mft_sparse_deconvolution.py",
        "separator_dependency_sha256": project / "cj0724/conditional_unet_separator_experiment.py",
    }
    mismatch = {name: {"expected": freeze.get(name), "actual": sha256_file(path)} for name, path in expected.items() if freeze.get(name) != sha256_file(path)}
    if mismatch:
        raise RuntimeError(f"P3o-R method freeze mismatch: {mismatch}")
    root = Path(args.rechunk_root)
    train_paths = role_paths(root, "train", 1880000000, args.rechunk_suffix, "p3or_mft_component_separation")
    development_paths = role_paths(root, "development", 1890000000, args.rechunk_suffix, "p3or_mft_component_separation")
    train_identity, train_labels, _ = read_role(train_paths)
    development_identity, development_labels, _ = read_role(development_paths)
    if not np.array_equal(train_labels, development_labels):
        raise RuntimeError("P3o-R train/development label contract differs")
    roles = _split_fit_selection(train_labels, 20260803)
    if np.intersect1d(roles["fit"], roles["selection"]).size:
        raise RuntimeError("P3o-R role split overlap")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in MODEL_SEEDS:
        run_dir = output / f"seed_{seed}"
        run_dir.mkdir()
        model, history, elapsed = train_one(seed, train_paths, roles, run_dir, args, device)
        evaluation = evaluate(model, development_paths, np.arange(len(development_identity), dtype=np.int64), device, args.batch_size)
        arrays, metrics = evaluation["arrays"], evaluation["metrics"]
        np.savez_compressed(run_dir / "development_predictions.npz", **arrays)
        (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
        receipt = {"seed": seed, "train_seconds": elapsed, "epochs_ran": len(history), "parameters": int(sum(parameter.numel() for parameter in model.parameters())), "checkpoint_sha256": sha256_file(run_dir / "best.pt"), "metrics": metrics}
        (run_dir / "receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
        runs.append(receipt)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"seed": seed, "waveform_cosine": metrics["mean_waveform_cosine"], "raw_overall": metrics["raw"]["overall"], "separated_overall": metrics["separated"]["overall"]}), flush=True)
    per_seed_gates = [_gate_report(run) for run in runs]
    gates = {
        "identity_overlap_zero": True,
        "all_power_and_label_contracts_hold": True,
        "each_seed_all_frozen_separability_and_count_gates": bool(all(all(row.values()) for row in per_seed_gates)),
        "all_three_seeds_directionally_consistent": bool(all(
            run["metrics"]["separated"]["overall"]["mae"] < run["metrics"]["raw"]["overall"]["mae"]
            for run in runs
        )),
        "test_generated_false": True,
        "test_opened_false": True,
        "locked_test_used_false": True,
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P3o-R MFT component separability before count extraction",
        "method_freeze": freeze,
        "data_audit": audit,
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "role_identity_hashes": {name: _identity_hash(train_identity[indices]) for name, indices in roles.items()},
        "train_identity_count": int(len(train_identity)),
        "development_identity_count": int(len(development_identity)),
        "runs": runs,
        "per_seed_gates": per_seed_gates,
        "promotion_gates": gates,
        "development_passed": passed,
        "verdict": "promote_to_two_identity_confirmation" if passed else "stop_no_confirmation_data",
        "classification_loss_used": False,
        "oracle_positive_only": True,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
