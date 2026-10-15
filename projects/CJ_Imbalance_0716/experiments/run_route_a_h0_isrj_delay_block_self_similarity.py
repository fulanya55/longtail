"""Five-seed H0 ISRJ replica-count screen for a fixed delay-block correlation map."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.models.delay_block_self_similarity import DelayBlockReplicaProbe
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write

REPLICA = 54


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _signals(path: Path, sample_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(sample_ids, dtype=np.int64)
    if not np.array_equal(ids, np.sort(ids)):
        raise ValueError("sample ids must be sorted")
    with h5py.File(path, "r") as handle:
        if "data_legacy" in handle:
            raw = np.asarray(handle["data_legacy"][:, ids], dtype=np.float32).T
            signal = raw.reshape(len(ids), 2, 3000)
        else:
            raw = np.asarray(handle["iq_complex"][:, ids]).T
            signal = np.stack([raw["real"], raw["imag"]], axis=1).astype(np.float32, copy=False)
    power = np.square(signal).sum(axis=1).mean(axis=1)
    signal = signal / np.sqrt(np.maximum(power, 1.0e-12))[:, None, None]
    return np.ascontiguousarray(signal, dtype=np.float32)


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    span = max(float(np.max(truth) - np.min(truth)), 1.0e-12)
    return {"r2": float(r2_score(truth, prediction)), "spearman": float(spearmanr(truth, prediction).statistic), "nmae_observed_range": float(np.abs(truth - prediction).mean() / span), "mae_count": float(np.abs(truth - prediction).mean())}


def _fit(name: str, seed: int, xfit: np.ndarray, yfit: np.ndarray, xearly: np.ndarray, yearly: np.ndarray, xeval: np.ndarray, yeval: np.ndarray, mean: float, std: float, args, root: Path, device: torch.device) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "run_receipt.json"
    if receipt.exists() and _read(receipt).get("status") == "completed":
        return _read(root / "metrics.json")
    _seed(seed)
    model = DelayBlockReplicaProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    generator = torch.Generator().manual_seed(seed + 1700)
    loader = DataLoader(TensorDataset(torch.from_numpy(xfit), torch.from_numpy((yfit - mean) / std)), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    early_x, early_y = torch.from_numpy(xearly).to(device), torch.from_numpy((yearly - mean) / std).to(device)
    best_loss, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for signal, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(signal.to(device)), target.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): early_loss = float(loss_fn(model(early_x), early_y).cpu())
        history.append({"epoch": epoch, "fit_smooth_l1": float(np.mean(losses)), "early_smooth_l1": early_loss})
        if early_loss < best_loss:
            best_loss, best_epoch = early_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): prediction = model(torch.from_numpy(xeval).to(device)).cpu().numpy() * std + mean
    result = _metrics(yeval, prediction)
    result.update({"status": "completed", "name": name, "seed": int(seed), "best_epoch": int(best_epoch), "best_early_smooth_l1": best_loss, "parameter_count": int(sum(value.numel() for value in model.parameters())), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_h0_used": False})
    torch.save({"state_dict": best_state, "seed": int(seed), "name": name, "architecture": "DelayBlockReplicaProbe"}, root / "best_model.pth")
    np.savez_compressed(root / "predictions.npz", true_replicas=yeval, prediction=prediction)
    _write(root / "history.json", history); _write(root / "metrics.json", result); _write(receipt, {key: result[key] for key in ("status", "name", "seed", "parameter_count", "locked_test_used", "test_v2_used", "test_v3_used", "test_h0_used")})
    return result


def run(args) -> dict:
    audit, seal = _read(args.audit), _read(args.test_v3_manifest)
    if not audit.get("identity_audit_passed") or audit.get("test_files_opened") or any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("identity audit or sealed test-v3 contract failed")
    if _sha(args.data_path).lower() != audit["sha256"].lower():
        raise ValueError("source data hash changed")
    roles = np.load(args.roles, allow_pickle=False)
    ids = {name: np.asarray(roles[name], dtype=np.int64) for name in ("fit", "early_stop", "evaluation")}
    all_ids = np.sort(np.concatenate(tuple(ids.values())))
    with h5py.File(args.data_path, "r") as handle:
        values = np.asarray(handle["parameter_values"], dtype=np.float64).T
    signal = _signals(args.data_path, all_ids)
    position = {int(value): index for index, value in enumerate(all_ids)}
    index = {name: np.asarray([position[int(value)] for value in selected], dtype=np.int64) for name, selected in ids.items()}
    target = values[all_ids, REPLICA].astype(np.float32)
    fit, early, evaluation = index["fit"], index["early_stop"], index["evaluation"]
    mean, std = float(target[fit].mean()), float(target[fit].std())
    if std <= 0: raise ValueError("zero replica target variance")
    rng = np.random.default_rng(args.permutation_seed)
    permuted = target.copy(); permuted[fit] = target[fit][rng.permutation(len(fit))]; permuted[early] = target[early][rng.permutation(len(early))]
    if args.smoke: args.seeds, args.epochs = [args.seeds[0]], args.smoke_epochs
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for seed in args.seeds:
        for name, target_values in (("R_real", target), ("R_perm", permuted)):
            rows.append(_fit(name, seed, signal[fit], target_values[fit], signal[early], target_values[early], signal[evaluation], target[evaluation], mean, std, args, args.output_root / "runs" / name / f"seed_{seed}", device))
    comparisons = []
    for seed in args.seeds:
        real = next(row for row in rows if row["seed"] == seed and row["name"] == "R_real")
        perm = next(row for row in rows if row["seed"] == seed and row["name"] == "R_perm")
        comparisons.append({"seed": int(seed), "delta_r2": real["r2"] - perm["r2"], "delta_spearman": real["spearman"] - perm["spearman"], "delta_nmae": real["nmae_observed_range"] - perm["nmae_observed_range"]})
    aggregate = {field: _mean_ci([row[field] for row in comparisons], args.permutation_seed + offset) for offset, field in enumerate(("delta_r2", "delta_spearman", "delta_nmae"))}
    real_rows = [row for row in rows if row["name"] == "R_real"]
    checks = {"matched_parameter_count": len({row["parameter_count"] for row in rows}) == 1, "real_mean_r2_ge_0p20": float(np.mean([row["r2"] for row in real_rows])) >= .20, "real_mean_spearman_ge_0p40": float(np.mean([row["spearman"] for row in real_rows])) >= .40, "real_minus_perm_r2_ge_0p20": aggregate["delta_r2"]["mean"] >= .20, "real_minus_perm_r2_ci_positive": aggregate["delta_r2"]["bootstrap_95_ci"][0] > 0, "positive_r2_4of5": sum(row["delta_r2"] > 0 for row in comparisons) >= 4, "test_v3_sealed": True}
    passed = bool(all(checks.values())) if not args.smoke else False
    payload = {"status": "completed", "experiment": "route_a_h0_isrj_delay_block_self_similarity", "evidence_role": "smoke" if args.smoke else "fresh_train_only_isrj_replica_identifiability", "data": {"path": str(args.data_path), "sha256": audit["sha256"], "roles": {key: int(len(value)) for key, value in ids.items()}}, "representation": {"map": "normalized complex delay-block correlation", "lags_samples": list(range(80, 481, 16)), "blocks": 12, "input_shape": [2, 26, 12]}, "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "loss": "SmoothL1", "batch_size": args.batch_size, "epochs": args.epochs}, "rows": rows, "real_minus_permutation": {"per_seed": comparisons, "aggregate": aggregate}, "gate": {"checks": checks, "passed": passed, "decision": "permit_separate_h1_physical_map_control_design" if passed else "stop_delay_block_map"}, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_h0_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_h0_isrj_delay_block_self_similarity_summary.json", payload)
    _write(args.output_root / "run_receipt.json", {"status": "completed", "evidence_role": payload["evidence_role"], "data_sha256": audit["sha256"], "gate_passed": passed, "decision": payload["gate"]["decision"], "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_h0_used": False})
    print(json.dumps({"passed": passed, "checks": checks, "output": str(args.output_root)}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_delay_block_h0\closedset_6class_train.mat"))
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_h0_isrj_delay_block_generation/identity_audit.json")
    parser.add_argument("--roles", type=Path, default=ROOT / "outputs/tapc/route_a_h0_isrj_delay_block_generation/role_indices.npz")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/tapc/route_a_h0_isrj_delay_block_self_similarity")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--permutation-seed", type=int, default=2026080214)
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
