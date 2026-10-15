"""P8b: equal-compute random versus axis-stratified BCE training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn import metrics as sk_metrics
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cj0724.data import complex_rms_normalize
from cj0724.dual_axis import (
    CLASS_NAMES, audit_power_alignment, build_identity_roles, discover_power_train_files,
    read_identity_metadata, roles_receipt,
)
from cj0724.dual_axis_train import (
    metrics_from_probabilities, predict, run_train_epoch, select_f1_thresholds, set_seed,
)
from cj0724.models import build_benchmark_model
from cj0724.p8a_multi_axis_longtail_atlas import build_manifest, sha256_file


VFT = CLASS_NAMES.index("VFT")
POWERS = (10, 15, 20, 25, 30, 35, 40, 45, 50)


def _json_safe(value):
    """Convert NumPy containers in the training receipt to standard JSON types."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class FixedViewDataset(Dataset):
    """One frozen I/Q power view per selected identity."""

    def __init__(self, files: dict[int, Path], labels: np.ndarray, indices: np.ndarray, levels: np.ndarray):
        self.files = {int(key): Path(value) for key, value in files.items()}
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.levels = np.asarray(levels, dtype=np.int16)
        if len(self.indices) != len(self.levels):
            raise ValueError("indices and levels must have equal length")
        if not np.isin(self.levels, np.asarray(POWERS, dtype=np.int16)).all():
            raise ValueError("A requested power level is unavailable")
        self.handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return int(len(self.indices))

    def _handle(self, level: int) -> h5py.File:
        if level not in self.handles:
            self.handles[level] = h5py.File(str(self.files[level]), "r")
        return self.handles[level]

    def __getitem__(self, position: int) -> dict:
        position = int(position); sample_index = int(self.indices[position]); level = int(self.levels[position])
        raw = np.asarray(self._handle(level)["data_legacy"][:, sample_index], dtype=np.float32)
        label = self.labels[sample_index]
        weak = np.zeros(len(CLASS_NAMES), dtype=bool)
        weak[VFT] = bool(level < 30 and label[VFT] and label[:3].any())
        return {
            "iq": torch.from_numpy(complex_rms_normalize(raw.reshape(2, -1))),
            "label": torch.from_numpy(label.astype(np.float32, copy=True)),
            "weak_mask": torch.from_numpy(weak),
            "sample_index": sample_index,
            "level": level,
        }

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles = {}

    def __del__(self) -> None:
        self.close()


class ReplacementBatchSampler(Sampler[list[int]]):
    def __init__(self, size: int, batch_size: int, steps: int, seed: int):
        self.size, self.batch_size, self.steps, self.seed, self.epoch = int(size), int(batch_size), int(steps), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        for _ in range(self.steps):
            yield rng.integers(0, self.size, size=self.batch_size, endpoint=False).tolist()

    def __len__(self) -> int:
        return self.steps


class AxisStratifiedBatchSampler(Sampler[list[int]]):
    """Deterministic replacement sampling with fixed, disjoint semantic quotas."""

    QUOTAS = {"weak_vft": 16, "other_vft": 4, "hard_negative": 16, "remainder": 28}

    def __init__(self, labels: np.ndarray, levels: np.ndarray, batch_size: int, steps: int, seed: int):
        labels, levels = np.asarray(labels, dtype=np.uint8), np.asarray(levels, dtype=np.int16)
        if int(batch_size) != sum(self.QUOTAS.values()):
            raise ValueError("P8b quotas require batch size 64")
        weak = np.logical_and(labels[:, VFT] == 1, np.logical_and(labels[:, :3].any(axis=1), levels < 30))
        other_vft = np.logical_and(labels[:, VFT] == 1, ~weak)
        hard = np.logical_and(labels[:, VFT] == 0, np.logical_and.reduce((labels[:, 2] == 1, labels[:, 3] == 1, labels[:, 5] == 1)))
        remainder = ~(weak | other_vft | hard)
        self.pools = {
            "weak_vft": np.flatnonzero(weak), "other_vft": np.flatnonzero(other_vft),
            "hard_negative": np.flatnonzero(hard), "remainder": np.flatnonzero(remainder),
        }
        for name, quota in self.QUOTAS.items():
            if len(self.pools[name]) < quota:
                raise ValueError(f"Insufficient {name} pool: {len(self.pools[name])} < {quota}")
        self.steps, self.seed, self.epoch = int(steps), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        for _ in range(self.steps):
            batch = np.concatenate([rng.choice(self.pools[name], size=quota, replace=False) for name, quota in self.QUOTAS.items()])
            rng.shuffle(batch)
            yield batch.astype(np.int64).tolist()

    def __len__(self) -> int:
        return self.steps


def _loader(dataset: Dataset, batch_size: int | None = None, sampler: Sampler | None = None) -> DataLoader:
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    return DataLoader(dataset, batch_size=int(batch_size or 64), shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


def _metric_summary(labels: np.ndarray, prob: np.ndarray, thresholds: np.ndarray, weak: np.ndarray) -> dict:
    base = metrics_from_probabilities(labels, prob, thresholds, weak)
    hard = np.logical_and(labels[:, VFT] == 0, np.logical_and.reduce((labels[:, 2] == 1, labels[:, 3] == 1, labels[:, 5] == 1)))
    positive = labels[:, VFT] == 1
    decision = prob[:, VFT] >= thresholds[VFT]
    base.update({
        "vft_hard_negative_count": int(hard.sum()),
        "vft_hard_negative_fpr": float(decision[hard].mean()) if hard.any() else float("nan"),
        "vft_separation": float(prob[positive, VFT].mean() - prob[hard, VFT].mean()) if positive.any() and hard.any() else float("nan"),
    })
    return base


def _bootstrap_vft_aupr_delta(labels: np.ndarray, c0: np.ndarray, c1: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(int(seed)); y = labels[:, VFT]; draws = []
    for _ in range(300):
        take = rng.integers(0, len(y), len(y))
        draws.append(float(sk_metrics.average_precision_score(y[take], c1[take, VFT]) - sk_metrics.average_precision_score(y[take], c0[take, VFT])))
    return {"samples": 300, "mean": float(np.mean(draws)), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975))}


def _predict_power(model: nn.Module, files: dict[int, Path], labels: np.ndarray, indices: np.ndarray, level: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = FixedViewDataset(files, labels, indices, np.full(len(indices), int(level), dtype=np.int16))
    true, probability, weak, sample_indices, _ = predict(model, _loader(dataset, 64), device)
    dataset.close()
    return true, probability, weak, sample_indices


def _train_one(name: str, seed: int, files: dict[int, Path], labels: np.ndarray, train_manifest: dict, roles: dict[str, np.ndarray], cfg: dict, output: Path, device: torch.device) -> dict:
    set_seed(seed)
    selected = np.asarray(train_manifest["selected_indices"], dtype=np.int64)
    levels = np.asarray(train_manifest["view_level_by_selected_position_db"], dtype=np.int16)
    train_dataset = FixedViewDataset(files, labels, selected, levels)
    steps, batch = int(cfg["models"]["optimization"]["steps_per_epoch"]), int(cfg["models"]["optimization"]["batch_size"])
    if name == "C0_random_BCE":
        sampler: Sampler = ReplacementBatchSampler(len(selected), batch, steps, seed)
    else:
        sampler = AxisStratifiedBatchSampler(labels[selected], levels, batch, steps, seed)
    train_loader = _loader(train_dataset, sampler=sampler)
    early_dataset = FixedViewDataset(files, labels, roles["early_stop"], np.full(len(roles["early_stop"]), 10, dtype=np.int16))
    early_loader = _loader(early_dataset, 64)
    model = build_benchmark_model("efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256).to(device)
    criterion = nn.BCEWithLogitsLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["models"]["optimization"]["lr"]), weight_decay=float(cfg["models"]["optimization"]["weight_decay"]))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_key, best_epoch, best_state = (-np.inf, -np.inf, -np.inf), 0, None
    history = []
    for epoch in range(1, int(cfg["models"]["optimization"]["epochs"]) + 1):
        sampler.set_epoch(epoch)
        loss = run_train_epoch(model, train_loader, criterion, optimizer, scaler, device, "single", 0.0)
        y, p, weak, _, _ = predict(model, early_loader, device)
        report = metrics_from_probabilities(y, p, np.full(6, 0.5), weak)
        key = (float(report["macro_aupr"]), float(report["per_label_aupr"][VFT]), -epoch)
        history.append({"epoch": epoch, "loss": loss, "macro_aupr": key[0], "vft_aupr": key[1]})
        if key > best_key:
            best_key, best_epoch, best_state = key, epoch, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state); model.eval()
    calibration_dataset = FixedViewDataset(files, labels, roles["calibration"], np.full(len(roles["calibration"]), 10, dtype=np.int16))
    cy, cp, _, _, _ = predict(model, _loader(calibration_dataset, 64), device); calibration_dataset.close()
    thresholds = select_f1_thresholds(cy, cp)
    powers = {}; raw = {}
    for power in POWERS:
        y, p, weak, idx = _predict_power(model, files, labels, roles["development"], power, device)
        powers[str(power)] = _metric_summary(y, p, thresholds, weak)
        raw[str(power)] = {"labels": y, "probability": p, "indices": idx}
    checkpoint = output / f"{name}__seed{seed}.pt"
    torch.save({"model_state": model.state_dict(), "thresholds": thresholds, "best_epoch": best_epoch, "history": history}, checkpoint)
    np.savez_compressed(output / f"{name}__seed{seed}_development.npz", **{f"{power}_{key}": value for power, values in raw.items() for key, value in values.items()})
    train_dataset.close(); early_dataset.close()
    return {"name": name, "seed": int(seed), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "best_epoch": int(best_epoch), "thresholds": thresholds.tolist(), "history": history, "powers": powers, "raw_10": raw["10"]}


def _gates(pairs: list[dict]) -> dict:
    rows = [pair["delta"] for pair in pairs]
    non_vft = [all(value <= 0.02 for value in row["non_vft_fpr_delta"]) for row in rows]
    return {
        "all_three_seeds_vft_10db_aupr_nonnegative_vs_C0": all(row["vft_10db_aupr_delta"] >= 0.0 for row in rows),
        "mean_vft_10db_aupr_gain_vs_C0_min": float(np.mean([row["vft_10db_aupr_delta"] for row in rows])) >= 0.02,
        "all_three_seeds_macro_aupr_delta_vs_C0_min": all(row["macro_10db_aupr_delta"] >= -0.005 for row in rows),
        "all_three_seeds_hamming_delta_vs_C0_max": all(row["hamming_10db_delta"] <= 0.005 for row in rows),
        "all_three_seeds_vft_hard_negative_fpr_delta_vs_C0_max": all(row["hard_fpr_10db_delta"] <= 0.02 for row in rows),
        "all_three_seeds_worst_power_vft_fpr_delta_vs_C0_max": all(row["worst_power_vft_fpr_delta"] <= 0.02 for row in rows),
        "all_three_seeds_vft_separation_delta_vs_C0_min": all(row["vft_separation_10db_delta"] >= 0.0 for row in rows),
        "no_non_vft_label_has_systemic_fpr_increase_over_0p02": all(non_vft),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.preregister.read_text(encoding="utf-8")); freeze = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    if sha256_file(args.preregister) != freeze["preregister_sha256"] or sha256_file(Path(__file__)) != freeze["runner_sha256"]:
        raise RuntimeError("P8b preregistration or runner hash mismatch")
    if args.output_root.exists():
        raise FileExistsError(f"Output exists: {args.output_root}")
    files = discover_power_train_files(args.data_root)
    if any("test" in str(path).lower() for path in files.values()):
        raise RuntimeError("P8b permits only train paths")
    reference = Path(cfg["parent_protocol"]["p8a_result"])
    p8a = json.loads(reference.read_text(encoding="utf-8"))
    _, old_labels = read_identity_metadata(Path(p8a["train_only_power_alignment"]["levels"]["30"]["path"]))
    fresh_ids, labels = read_identity_metadata(files[30])
    old_ids, _ = read_identity_metadata(Path(p8a["train_only_power_alignment"]["levels"]["30"]["path"]))
    if np.intersect1d(fresh_ids, old_ids).size:
        raise RuntimeError("Fresh P8b identities overlap P8a root")
    roles = build_identity_roles(labels, int(cfg["roles"]["split_seed"]))
    manifest = build_manifest(labels, roles["optimization"], "mixed_lt", cfg["roles"]["training_manifest"]["label_targets"], "combo_lt", 1310000003, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); args.output_root.mkdir(parents=True)
    runs = []
    for seed in cfg["models"]["model_seeds_development"]:
        c0 = _train_one("C0_random_BCE", int(seed), files, labels, manifest, roles, cfg, args.output_root, device)
        c1 = _train_one("C1_axis_stratified_BCE", int(seed), files, labels, manifest, roles, cfg, args.output_root, device)
        a, b = c0["powers"]["10"], c1["powers"]["10"]
        worst = max(c1["powers"][str(power)]["per_label_fpr"][VFT] - c0["powers"][str(power)]["per_label_fpr"][VFT] for power in POWERS)
        delta = {
            "vft_10db_aupr_delta": b["per_label_aupr"][VFT] - a["per_label_aupr"][VFT],
            "macro_10db_aupr_delta": b["macro_aupr"] - a["macro_aupr"],
            "hamming_10db_delta": b["hamming_loss"] - a["hamming_loss"],
            "hard_fpr_10db_delta": b["vft_hard_negative_fpr"] - a["vft_hard_negative_fpr"],
            "vft_separation_10db_delta": b["vft_separation"] - a["vft_separation"],
            "worst_power_vft_fpr_delta": worst,
            "non_vft_fpr_delta": [b["per_label_fpr"][i] - a["per_label_fpr"][i] for i in range(6) if i != VFT],
        }
        bootstrap = _bootstrap_vft_aupr_delta(c0["raw_10"]["labels"], c0["raw_10"]["probability"], c1["raw_10"]["probability"], int(seed) + 500)
        c0.pop("raw_10"); c1.pop("raw_10"); runs.append({"seed": int(seed), "C0": c0, "C1": c1, "delta": delta, "bootstrap_vft_10db_aupr": bootstrap})
    gates = _gates(runs)
    result = {
        "experiment": cfg["experiment"], "preregister_sha256": sha256_file(args.preregister), "method_freeze_sha256": sha256_file(args.method_freeze),
        "train_only_alignment": audit_power_alignment(files), "roles": roles_receipt(roles, labels, int(cfg["roles"]["split_seed"])),
        "manifest": manifest, "runs": runs, "gates": gates, "development_passed": bool(all(gates.values())),
        "verdict": "promote_to_two_model_seed_extension" if all(gates.values()) else "stop_no_confirmation_data",
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (args.output_root / "result.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": result["development_passed"], "gates": gates, "verdict": result["verdict"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
