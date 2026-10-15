"""Preregistered H1u training: correct-LFM residual versus phase-order control.

This runner is deliberately self-contained.  It reads only the fresh H1u
training source and the public test-v3 seal, never a test MAT file.
"""
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.models.mft_matched_filter_support import fixed_phase_permutation, lfm_reference

DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
SOURCE = DATA_ROOT / "复合干扰数据0803_h1u_mft_delay_aperture_r1" / "closedset_6class_train.mat"
BANDS = ((1600, 3000), (3500, 5000), (5400, 7000))
ARMS = ("B0", "M1", "M2")
SPLIT_SEEDS = (3101, 3102, 3103, 3104, 3105)
MODEL_SEEDS = (4101, 4102, 4103, 4104, 4105)
MFT_INDEX = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalise(iq: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.maximum(np.mean(np.square(iq[:, 0]) + np.square(iq[:, 1]), axis=1, keepdims=True), 1e-12))
    return (iq / rms[:, :, None]).astype(np.float32, copy=False)


def load_h1u_source(path: Path) -> dict[str, np.ndarray]:
    """Load only the explicit H1u development source, without test data."""
    with h5py.File(path, "r") as handle:
        real, imag = np.asarray(handle["iq_real"], dtype=np.float32), np.asarray(handle["iq_imag"], dtype=np.float32)
        labels = np.asarray(handle["label_selected"], dtype=np.float32)
        combination = np.asarray(handle["combination_mask"]).reshape(-1).astype(np.int64)
        weak = np.asarray(handle["mft_weak_mask"]).reshape(-1).astype(bool)
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    iq = _normalise(np.stack((real, imag), axis=1))
    if iq.shape != (5040, 2, 10000) or labels.shape != (5040, 6):
        raise ValueError(f"unexpected H1u input shape {iq.shape} / {labels.shape}")
    return {"iq": iq, "labels": labels, "combination": combination, "weak": weak, "receiver": receiver}


def make_roles(combination: np.ndarray, labels: np.ndarray, weak: np.ndarray, split_seed: int) -> dict[str, np.ndarray]:
    """Split every full-combination x MFT-state stratum 60/20/20."""
    state = np.where(labels[:, MFT_INDEX] == 0, 0, np.where(weak, 1, 2)).astype(np.int64)
    rng, roles = np.random.default_rng(split_seed), np.full(len(state), -1, dtype=np.int8)
    for comb in range(1, 64):
        for visibility in range(3):
            members = np.flatnonzero((combination == comb) & (state == visibility))
            if not len(members):
                continue
            members = rng.permutation(members)
            n_fit, n_validation = int(round(.60 * len(members))), int(round(.20 * len(members)))
            roles[members[:n_fit]] = 0; roles[members[n_fit:n_fit + n_validation]] = 1; roles[members[n_fit + n_validation:]] = 2
    if np.any(roles < 0):
        raise ValueError("an H1u sample did not receive a role")
    return {"fit": np.flatnonzero(roles == 0), "validation": np.flatnonzero(roles == 1), "early": np.flatnonzero(roles == 2), "state": state}


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, 9, stride=4, padding=4, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, item: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.pointwise(self.depthwise(item))))


class H1uModel(nn.Module):
    """Shared compact backbone plus equal-capacity bounded MFT residual head."""
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown H1u arm {arm}")
        self.arm = arm
        self.stem = nn.Sequential(nn.Conv1d(2, 16, 17, stride=4, padding=8, bias=False), nn.BatchNorm1d(16), nn.SiLU())
        self.stage2, self.stage3 = DepthwiseBlock(16, 32), DepthwiseBlock(32, 48)
        self.classifier = nn.Linear(96, 6)
        self.mft_residual = nn.Linear(3, 1)
        reference = lfm_reference(); self.register_buffer("reference", reference, persistent=False)
        self.register_buffer("phase_control", fixed_phase_permutation(reference), persistent=False)

    @staticmethod
    def _response(iq: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        z, length = torch.complex(iq[:, 0].float(), iq[:, 1].float()), iq.shape[-1]
        template = torch.zeros(length, device=iq.device, dtype=torch.complex64)
        template[:reference.numel()] = reference.to(device=iq.device, dtype=torch.complex64)
        return torch.fft.ifft(torch.fft.fft(z, dim=-1) * torch.fft.fft(template).conj()[None, :], dim=-1).abs()

    def residual_features(self, iq: torch.Tensor) -> torch.Tensor:
        if self.arm == "B0":
            return torch.zeros((len(iq), 3), dtype=iq.dtype, device=iq.device)
        reference = self.reference if self.arm == "M1" else self.phase_control
        with torch.no_grad():
            response = self._response(iq, reference)
            return torch.stack([response[:, left:right].amax(dim=1) for left, right in BANDS], dim=1).clamp_min(1e-12).log()

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        feature = self.stage3(self.stage2(self.stem(iq)))
        pooled = torch.cat((feature.mean(dim=-1), feature.std(dim=-1, unbiased=False)), dim=1)
        logits = self.classifier(pooled)
        logits[:, MFT_INDEX] = logits[:, MFT_INDEX] + .5 * torch.tanh(self.mft_residual(self.residual_features(iq)).squeeze(1))
        return logits


def balanced_loss(logits: torch.Tensor, labels: torch.Tensor, weak: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    other = loss[:, [0, 1, 2, 4, 5]].mean()
    target = loss[:, MFT_INDEX]
    mft_negative, mft_weak, mft_strong = labels[:, MFT_INDEX] == 0, weak.bool(), (labels[:, MFT_INDEX] == 1) & ~weak.bool()
    if not all(mask.any() for mask in (mft_negative, mft_weak, mft_strong)):
        raise ValueError("fit batch lacks one H1u MFT stratum; use stratified batch construction")
    mft = (target[mft_negative].mean() + target[mft_weak].mean() + target[mft_strong].mean()) / 3.0
    return other + mft


def _stratified_loader(data: dict[str, np.ndarray], indices: np.ndarray, *, batch_size: int, seed: int) -> DataLoader:
    labels, weak = data["labels"][indices], data["weak"][indices]
    state = np.where(labels[:, MFT_INDEX] == 0, 0, np.where(weak, 1, 2))
    groups = [indices[state == value] for value in range(3)]
    # 10 negatives + 11 weak + 11 strong avoids absent-stratum batches and retains every item by cycling.
    quotas = (10, 11, 11)
    rng, samples = np.random.default_rng(seed), []
    steps = int(np.ceil(len(indices) / batch_size))
    for _ in range(steps):
        batch = [rng.choice(group, size=count, replace=len(group) < count) for group, count in zip(groups, quotas)]
        samples.append(np.concatenate(batch))
    flattened = np.concatenate(samples)
    tensor = TensorDataset(torch.from_numpy(data["iq"][flattened]), torch.from_numpy(data["labels"][flattened]), torch.from_numpy(data["weak"][flattened]))
    return DataLoader(tensor, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


def _loader(data: dict[str, np.ndarray], indices: np.ndarray, batch_size: int) -> DataLoader:
    tensor = TensorDataset(torch.from_numpy(data["iq"][indices]), torch.from_numpy(data["labels"][indices]), torch.from_numpy(data["weak"][indices]))
    return DataLoader(tensor, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


def _rank_metrics(labels: np.ndarray, weak: np.ndarray, logits: np.ndarray) -> dict[str, object]:
    probability = 1.0 / (1.0 + np.exp(-logits))
    absent, weak_pos, strong_pos = labels[:, MFT_INDEX] == 0, weak, (labels[:, MFT_INDEX] == 1) & ~weak
    def binary(pos: np.ndarray) -> dict[str, float]:
        selected, y = absent | pos, pos[absent | pos].astype(np.uint8)
        score = probability[selected, MFT_INDEX]
        return {"auroc": float(roc_auc_score(y, score)), "aupr": float(average_precision_score(y, score))}
    per_label = [float(average_precision_score(labels[:, index], probability[:, index])) for index in range(6)]
    return {"weak_mft": binary(weak_pos), "strong_mft": binary(strong_pos), "macro_aupr": float(np.mean(per_label)), "non_mft_mean_aupr": float(np.mean([per_label[index] for index in (0, 1, 2, 4, 5)])), "per_label_aupr": per_label}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model.eval(); all_logits, all_labels, all_weak = [], [], []
    for iq, labels, weak in loader:
        all_logits.append(model(iq.to(device, non_blocking=True)).cpu().numpy())
        all_labels.append(labels.numpy()); all_weak.append(weak.numpy())
    stored = {"logits": np.concatenate(all_logits), "labels": np.concatenate(all_labels), "weak": np.concatenate(all_weak).astype(bool)}
    return _rank_metrics(stored["labels"], stored["weak"], stored["logits"]), stored


def _mean_bootstrap(values: list[float], seed: int, draws: int = 5000) -> dict[str, object]:
    values_array = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    samples = values_array[rng.integers(0, len(values_array), size=(draws, len(values_array)))].mean(axis=1)
    return {"mean": float(values_array.mean()), "per_seed": values_array.tolist(), "bootstrap_95_ci": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], "positive_seed_count": int((values_array > 0).sum())}


def _train_arm(data: dict[str, np.ndarray], roles: dict[str, np.ndarray], arm: str, split_seed: int, model_seed: int, args: argparse.Namespace, root: Path, device: torch.device) -> dict[str, object]:
    _set_seed(model_seed)
    arm_root = root / f"split{split_seed}_model{model_seed}" / arm; arm_root.mkdir(parents=True, exist_ok=True)
    model = H1uModel(arm).to(device); params = int(sum(item.numel() for item in model.parameters()))
    train_loader = _stratified_loader(data, roles["fit"], batch_size=args.batch_size, seed=model_seed)
    validation_loader, early_loader = _loader(data, roles["validation"], args.batch_size), _loader(data, roles["early"], args.batch_size)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, history, stale = -np.inf, [], 0
    checkpoint = arm_root / "best.pt"
    for epoch in range(1, args.max_epochs + 1):
        model.train(); losses = []
        for iq, labels, weak in train_loader:
            optimiser.zero_grad(set_to_none=True)
            value = balanced_loss(model(iq.to(device, non_blocking=True)), labels.to(device, non_blocking=True), weak.to(device, non_blocking=True))
            value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step(); losses.append(float(value.detach().cpu()))
        validation, _ = evaluate(model, validation_loader, device)
        current = float(validation["weak_mft"]["auroc"])
        history.append({"epoch": epoch, "fit_loss": float(np.mean(losses)), "validation": validation})
        if current > best + 1e-12:
            best, stale = current, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_weak_mft_auroc": best}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True); model.load_state_dict(saved["state_dict"])
    early, stored = evaluate(model, early_loader, device)
    np.savez_compressed(arm_root / "early_predictions.npz", logits=stored["logits"], labels=stored["labels"], mft_weak_mask=stored["weak"], receiver_noise_seed=data["receiver"][roles["early"]])
    result = {"arm": arm, "split_seed": split_seed, "model_seed": model_seed, "parameter_count": params, "best_epoch": int(saved["epoch"]), "best_validation_weak_mft_auroc": float(best), "early": early, "epochs_ran": len(history), "checkpoint": str(checkpoint), "no_calibration": True, "no_threshold": True}
    _json(history, arm_root / "history.json"); _json(result, arm_root / "metrics.json")
    return result


def _verify_boundary(audit: Path, seal: Path) -> dict[str, object]:
    source_audit, manifest = json.loads(audit.read_text(encoding="utf-8")), json.loads(seal.read_text(encoding="utf-8"))
    if not source_audit.get("identity_audit_passed") or not all(source_audit["contracts"].values()):
        raise ValueError("H1u source provenance audit did not pass")
    if any(manifest["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    return source_audit


def smoke(args: argparse.Namespace) -> dict[str, object]:
    """Synthetic code-path smoke: does not open the formal H1u source."""
    _set_seed(7); device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    iq = torch.randn(32, 2, 10000, device=device); labels = torch.zeros(32, 6, device=device); labels[10:21, MFT_INDEX] = 1; labels[21:, MFT_INDEX] = 1
    weak = torch.zeros(32, dtype=torch.bool, device=device); weak[10:21] = True
    rows = {}
    for arm in ARMS:
        model = H1uModel(arm).to(device); value = balanced_loss(model(iq), labels, weak); value.backward()
        rows[arm] = {"loss_finite": bool(torch.isfinite(value)), "parameter_count": int(sum(x.numel() for x in model.parameters()))}
    result = {"status": "completed_synthetic_smoke", "formal_h1u_source_opened": False, "rows": rows, "parameter_counts_equal": len({row["parameter_count"] for row in rows.values()}) == 1, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _json(result, args.output_dir / "smoke" / "summary.json"); return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.smoke:
        return smoke(args)
    audit = _verify_boundary(args.source_audit, args.test_v3_manifest)
    data, device = load_h1u_source(args.source), torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results, root = [], args.output_dir
    for split_seed, model_seed in zip(SPLIT_SEEDS, MODEL_SEEDS):
        roles = make_roles(data["combination"], data["labels"], data["weak"], split_seed)
        role_path = root / f"split{split_seed}_model{model_seed}" / "roles.npz"; role_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(role_path, fit=roles["fit"], validation=roles["validation"], early=roles["early"], receiver_noise_seed=data["receiver"])
        role_counts = {key: int(len(roles[key])) for key in ("fit", "validation", "early")}
        for arm in ARMS:
            row = _train_arm(data, roles, arm, split_seed, model_seed, args, root, device); row["role_counts"] = role_counts; row["roles_sha256"] = _sha256(role_path); results.append(row)
    table = {arm: {row["split_seed"]: row for row in results if row["arm"] == arm} for arm in ARMS}
    def delta(left: str, right: str, path: tuple[str, ...], key: int) -> dict[str, object]:
        values = []
        for seed in SPLIT_SEEDS:
            lhs, rhs = table[left][seed], table[right][seed]
            for part in path:
                lhs, rhs = lhs[part], rhs[part]
            values.append(float(lhs - rhs))
        return _mean_bootstrap(values, key)
    comparison = {
        "m1_minus_m2_weak_auroc": delta("M1", "M2", ("early", "weak_mft", "auroc"), 20260831),
        "m1_minus_m2_weak_aupr": delta("M1", "M2", ("early", "weak_mft", "aupr"), 20260832),
        "m1_minus_b0_weak_auroc": delta("M1", "B0", ("early", "weak_mft", "auroc"), 20260833),
        "m1_minus_b0_strong_auroc": delta("M1", "B0", ("early", "strong_mft", "auroc"), 20260834),
        "m1_minus_b0_non_mft_mean_aupr": delta("M1", "B0", ("early", "non_mft_mean_aupr"), 20260835),
    }
    a, p, b, strong, non_mft = (comparison[key] for key in ("m1_minus_m2_weak_auroc", "m1_minus_m2_weak_aupr", "m1_minus_b0_weak_auroc", "m1_minus_b0_strong_auroc", "m1_minus_b0_non_mft_mean_aupr"))
    gates = {"m1_m2_weak_auroc_mean_ge_0p030": a["mean"] >= .030, "m1_m2_weak_auroc_4of5_positive": a["positive_seed_count"] >= 4, "m1_m2_weak_auroc_ci_lower_gt_zero": a["bootstrap_95_ci"][0] > 0, "m1_m2_weak_aupr_mean_ge_0p025": p["mean"] >= .025, "m1_m2_weak_aupr_ci_lower_gt_zero": p["bootstrap_95_ci"][0] > 0, "m1_b0_weak_auroc_mean_ge_0p030": b["mean"] >= .030, "m1_b0_weak_auroc_4of5_positive": b["positive_seed_count"] >= 4, "m1_b0_weak_auroc_ci_lower_gt_zero": b["bootstrap_95_ci"][0] > 0, "m1_b0_strong_auroc_not_worse_than_minus_0p010": strong["mean"] >= -.010, "m1_b0_non_mft_aupr_not_worse_than_minus_0p010": non_mft["mean"] >= -.010, "source_audit_passed": True, "test_v3_sealed": True}
    receipt = {"source": str(args.source), "source_sha256": audit["source_sha256"], "source_audit_sha256": _sha256(args.source_audit), "split_seeds": list(SPLIT_SEEDS), "model_seeds": list(MODEL_SEEDS), "arms": list(ARMS), "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "max_epochs": args.max_epochs, "patience": args.patience, "selection": "validation_mft_weak_vs_negative_auroc", "rank_only_no_calibration_or_threshold": True, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    result = {"status": "completed_fresh_source_multiseed_development", "experiment": "RouteA_H1u_MFTDelayApertureLearning", "evidence_role": "fresh_external_delay_aperture_component_learning", "rows": results, "comparison": comparison, "gates": gates, "passed": bool(all(gates.values())), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _json(receipt, root / "run_receipt.json"); _json(result, root / "summary.json"); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--source-audit", type=Path, default=ROOT / "outputs/tapc/route_a_h1u_mft_delay_aperture_generation/source_audit.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/tapc/route_a_h1u_mft_delay_aperture_learning")
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=24); parser.add_argument("--patience", type=int, default=6); parser.add_argument("--device", default=None); parser.add_argument("--smoke", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
