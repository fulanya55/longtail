"""Five-seed, identity-safe learnability screen for ISRJ sample width.

This is a prerequisite probe only.  It does not alter classification logits,
calibration, thresholds, fusion, or any sealed test role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import random
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.split_builder import RoleCounts, build_stratified_roles  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_development_v16_isrj_width_probe import audit_train  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VFT_INDEX, ISRJ_INDEX, WIDTH_INDEX = 4, 5, 53
ROLE_COUNTS = RoleCounts(
    optimization=48, early_stop=16, calibration=0, threshold_selection=16
)
SPLIT_SEED = 2026080201
PERMUTATION_SEED = 2026080202


class WidthProbe(nn.Module):
    """Compact raw I/Q multiscale regressor with mean/std/top-k evidence."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        base = int(width)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, base, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(base), nn.GELU(),
            nn.Conv1d(base, base * 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base * 2), nn.GELU(),
            nn.Conv1d(base * 2, base * 3, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(base * 3), nn.GELU(),
        )
        feature_dim = base * 3 * 3
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, base * 4),
            nn.GELU(), nn.Linear(base * 4, 1),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        temporal = self.encoder(signal.float())
        average = temporal.mean(dim=-1)
        deviation = temporal.std(dim=-1, unbiased=False)
        k = min(4, int(temporal.shape[-1]))
        local = temporal.topk(k=k, dim=-1).values.mean(dim=-1)
        return self.head(torch.cat([average, deviation, local], dim=1)).squeeze(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _validate_config(config: dict) -> None:
    if config.get("evidence_role") != "development_v16_isrj_width_identifiability":
        raise ValueError("requires v16 ISRJ-width configuration")
    if bool(config.get("allow_locked_test", False)) or not bool(config.get("test_v3_sealed", False)):
        raise ValueError("locked-test boundary is invalid")
    if any(bool(config.get(name, False)) for name in ("test_v2_used", "test_v3_used", "test_v16_used")):
        raise ValueError("a forbidden test role is marked used")
    if bool(config.get("test_v16_generated", True)) or bool(config.get("test_v16_access_authorized", True)):
        raise ValueError("v16 must contain train role only")


def _load_signals(path: Path, sample_ids: np.ndarray) -> np.ndarray:
    """Read selected legacy I/Q columns once, then exactly apply complex RMS.

    Individual HDF5 column reads repeatedly decompress large chunks on the v7.3
    files.  The selected ids are sorted and identity-safe before this function
    is called, so a single indexed read preserves the exact samples and only
    removes redundant I/O.
    """
    ids = np.asarray(sample_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or not np.array_equal(ids, np.sort(ids)):
        raise ValueError("sample ids must be a nonempty sorted vector")
    with h5py.File(path, "r") as handle:
        raw = np.asarray(handle["data_legacy"][:, ids], dtype=np.float32).T
    signal = raw.reshape(len(ids), 2, 3000)
    power = np.mean(
        np.square(signal[:, 0]) + np.square(signal[:, 1]), axis=1
    )
    signal = signal / np.sqrt(np.maximum(power, 1.0e-12))[:, None, None]
    if not np.all(np.isfinite(signal)):
        raise FloatingPointError("complex-RMS normalized signal is non-finite")
    return np.ascontiguousarray(signal, dtype=np.float32)


def _fixed_roles(labels: np.ndarray) -> tuple:
    eligible = np.flatnonzero((labels[:, VFT_INDEX] == 0) & (labels[:, ISRJ_INDEX] == 1))
    split = build_stratified_roles(labels, eligible, SPLIT_SEED, counts=ROLE_COUNTS)
    split.assert_disjoint()
    if sum(len(values) for values in (split.optimization, split.early_stop, split.threshold_selection)) != len(eligible):
        raise ValueError("hard-negative split must cover all eligible identities")
    return split, eligible


def _permuted(values: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    permutation = np.random.RandomState(PERMUTATION_SEED).permutation(len(sample_ids))
    mapping = {int(key): float(value) for key, value in zip(sample_ids, values[permutation])}
    return np.asarray([mapping[int(value)] for value in sample_ids], dtype=np.float32)


def _metrics(true_width: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(true_width, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    span = max(float(truth.max() - truth.min()), 1.0e-12)
    return {
        "r2": float(r2_score(truth, estimate)),
        "nmae_observed_range": float(np.abs(truth - estimate).mean() / span),
        "spearman": float(spearmanr(truth, estimate).statistic),
        "mae_seconds": float(np.abs(truth - estimate).mean()),
        "true_range_seconds": [float(truth.min()), float(truth.max())],
    }


def _fit_one(
    seed: int, model_name: str, fit_signal: np.ndarray, early_signal: np.ndarray,
    holdout_signal: np.ndarray, fit_target: np.ndarray, early_target: np.ndarray,
    holdout_true: np.ndarray, mean: float, std: float, args: argparse.Namespace,
    output_dir: Path, device: torch.device,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "run_receipt.json"
    if receipt_path.is_file() and _read(receipt_path).get("status") == "completed":
        return _read(output_dir / "metrics.json")
    _set_seed(seed)
    model = WidthProbe(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    fit_dataset = TensorDataset(torch.from_numpy(fit_signal), torch.from_numpy((fit_target - mean) / std))
    early_x = torch.from_numpy(early_signal).to(device)
    early_y = torch.from_numpy((early_target - mean) / std).to(device)
    holdout_x = torch.from_numpy(holdout_signal).to(device)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(fit_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    history, best_state, best_loss, best_epoch = [], None, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for signal, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(signal.to(device))
            loss = loss_fn(prediction, target.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            early_loss = float(loss_fn(model(early_x), early_y).cpu())
        history.append({"epoch": epoch, "fit_smooth_l1": float(np.mean(losses)), "early_smooth_l1": early_loss})
        if early_loss < best_loss:
            best_loss, best_epoch = early_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        holdout_prediction = model(holdout_x).cpu().numpy() * std + mean
    metric = _metrics(holdout_true, holdout_prediction)
    metric.update({
        "status": "completed", "model": model_name, "seed": int(seed), "best_epoch": int(best_epoch),
        "best_early_smooth_l1": float(best_loss), "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v16_used": False,
    })
    torch.save({"state_dict": best_state, "seed": int(seed), "model": model_name}, output_dir / "best_model.pth")
    np.savez_compressed(output_dir / "predictions.npz", true_width=holdout_true, prediction=holdout_prediction)
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write(output_dir / "metrics.json", metric)
    _write(receipt_path, {"status": "completed", "model": model_name, "seed": int(seed), "parameter_count": metric["parameter_count"], "best_epoch": int(best_epoch), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v16_used": False})
    return metric


def _comparison(rows: list[dict]) -> dict:
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        real = next(row for row in rows if row["seed"] == seed and row["model"] == "W_real")
        perm = next(row for row in rows if row["seed"] == seed and row["model"] == "W_perm")
        paired.append({"seed": int(seed), "delta_r2": real["r2"] - perm["r2"], "delta_spearman": real["spearman"] - perm["spearman"], "delta_nmae": real["nmae_observed_range"] - perm["nmae_observed_range"]})
    aggregate = {key: _mean_ci([row[key] for row in paired], 2026080300 + index) for index, key in enumerate(("delta_r2", "delta_spearman", "delta_nmae"))}
    aggregate["positive_r2_seeds"] = int(sum(row["delta_r2"] > 0 for row in paired))
    return {"per_seed": paired, "aggregate": aggregate}


def _gate(rows: list[dict], comparison: dict) -> dict:
    real = [row for row in rows if row["model"] == "W_real"]
    delta = comparison["aggregate"]
    checks = {
        "matched_parameters_5_of_5": all(row["parameter_count"] == real[0]["parameter_count"] for row in rows),
        "real_mean_r2_ge_0p20": float(np.mean([row["r2"] for row in real])) >= 0.20,
        "real_mean_spearman_ge_0p40": float(np.mean([row["spearman"] for row in real])) >= 0.40,
        "real_minus_perm_r2_ge_0p20": delta["delta_r2"]["mean"] >= 0.20,
        "real_minus_perm_r2_ci_positive": delta["delta_r2"]["bootstrap_95_ci"][0] > 0.0,
        "positive_r2_4_of_5": delta["positive_r2_seeds"] >= 4,
    }
    passed = all(checks.values())
    return {"checks": checks, "passed": passed, "decision": "advance_to_predicted_width_risk_control_only" if passed else "stop_width_conditioned_correction"}


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); identity = audit_train(data_path)
    if not identity["identity_audit_passed"] or identity["test_files_opened"]:
        raise ValueError("v16 train-only identity audit failed")
    labels = read_labels(data_path); values, mask = read_parameters(data_path)
    split, eligible = _fixed_roles(labels)
    width = values[:, WIDTH_INDEX].astype(np.float32)
    if not np.all(mask[eligible, WIDTH_INDEX]) or not np.all(np.isfinite(width[eligible])):
        raise ValueError("ISRJ width targets are incomplete")
    signals = _load_signals(data_path, eligible)
    position = {int(sample): index for index, sample in enumerate(eligible)}
    def select(ids: np.ndarray) -> np.ndarray: return np.asarray([position[int(item)] for item in ids], dtype=np.int64)
    fit_pos, early_pos, holdout_pos = select(split.optimization), select(split.early_stop), select(split.threshold_selection)
    target = width[eligible]; permuted = _permuted(target, eligible)
    mean, std = float(target[fit_pos].mean()), float(target[fit_pos].std())
    if std <= 0.0: raise ValueError("fit width has zero variance")
    if args.smoke: args.epochs = int(args.smoke_epochs); args.seeds = [int(args.seeds[0])]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizer = {
        "target": "ISRJ.sample_width_s",
        "mean_seconds": mean,
        "std_seconds": std,
        "source_data_sha256": _sha256(data_path),
        "role_split_sha256": split.sha256,
        "fit_identity_count": int(len(split.optimization)),
        "fit_only": True,
    }
    _write(Path(args.output_root) / "width_normalizer.json", normalizer)
    rows = []
    for seed in args.seeds:
        for name, target_values in (("W_real", target), ("W_perm", permuted)):
            print(f"model={name} seed={seed}", flush=True)
            metric = _fit_one(int(seed), name, signals[fit_pos], signals[early_pos], signals[holdout_pos], target_values[fit_pos], target_values[early_pos], target[holdout_pos], mean, std, args, Path(args.output_root) / "runs" / name / f"seed_{seed}", device)
            rows.append(metric)
    comparison = _comparison(rows)
    gate = {"passed": False, "decision": "smoke_only_no_method_claim"} if args.smoke else _gate(rows, comparison)
    payload = {"status": "completed", "experiment": "route_a_development_v16_isrj_width_probe", "evidence_level": "smoke" if args.smoke else "development", "protocol": {"seeds": [int(seed) for seed in args.seeds], "split_seed": SPLIT_SEED, "permutation_seed": PERMUTATION_SEED, "role_counts": asdict(ROLE_COUNTS), "eligible_condition": "VFT=0 AND ISRJ=1", "epochs": int(args.epochs), "width": int(args.width), "loss": "SmoothL1", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v16_used": False}, "data": {"path": str(data_path), "sha256": _sha256(data_path), "identity_audit": identity, "eligible_count": int(len(eligible),), "role_split_sha256": split.sha256}, "target_normalizer": normalizer, "per_run": rows, "comparison_real_minus_perm": comparison, "gate": gate, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v16_used": False}
    root = Path(args.output_root); _write(root / "route_a_development_v16_isrj_width_probe_summary.json", payload)
    _write(root / "run_receipt.json", {"status": "completed", "evidence_level": payload["evidence_level"], "data_sha256": payload["data"]["sha256"], "identity_audit_passed": True, "gate_passed": gate["passed"], "next_decision": gate["decision"], "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v16_used": False})
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v16_isrj_width_probe.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v16_isrj_width_probe")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=16); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
