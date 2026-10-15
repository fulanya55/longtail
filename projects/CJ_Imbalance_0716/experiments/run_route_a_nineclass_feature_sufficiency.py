"""Route-A nine-class feature-sufficiency gate (development/smoke only).

This runner deliberately uses only the E-drive nine-class development source. It
never resolves or opens a locked-test path.  The first implementation is a
small, resumable development screen: an untrained frozen-F2 linear probe (A0),
an end-to-end F2+9-head model (A1), and EfficientNet-B1-1D (A2) are trained on
the 30 dB train view and evaluated on disjoint identities from every power
level.  The artifact is a readiness result, not a locked-test result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CJ0724_ROOT = Path(r"E:\LJQ_code\CJ_Inbalance_0724")
if str(CJ0724_ROOT) not in sys.path:
    sys.path.insert(0, str(CJ0724_ROOT))

from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cj0724.models import build_model  # noqa: E402

CLASS_NAMES = (
    "FMB", "AMA", "SawFM", "SinFM", "SN", "MFT", "VFT", "ISRJ", "RangeFakeTarget"
)
POWERS_DB = tuple(range(10, 51, 5))
SEEDS = (2039, 2040, 2041)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_file(data_root: Path, power_db: int, role: str) -> Path:
    folder = data_root / "power_sweep_9level_v1" / f"level_{POWERS_DB.index(power_db)+1:02d}__jnr_{power_db}dB"
    path = folder / f"closedset_9class_{role}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def receiver_seeds(path: Path, indices: np.ndarray) -> list[int]:
    with h5py.File(str(path), "r") as handle:
        values = np.asarray(handle["receiver_noise_seed"][:, indices], dtype=np.int64).reshape(-1)
    return [int(value) for value in values.tolist()]


class NineClassDataset(Dataset):
    """Lazy HDF5 reader for iq_complex, returning [2, 3000] tensors."""

    def __init__(self, path: Path, indices: np.ndarray, label_indices: np.ndarray | None = None):
        self.path = Path(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_indices = np.arange(9, dtype=np.int64) if label_indices is None else np.asarray(label_indices, dtype=np.int64)
        if self.label_indices.ndim != 1 or np.any(self.label_indices < 0) or np.any(self.label_indices >= 9):
            raise ValueError("label_indices must select rows from the nine-class label matrix")
        self._handle = None
        with h5py.File(str(self.path), "r") as handle:
            labels = np.asarray(handle["known_label"][self.label_indices.tolist(), :][:, self.indices], dtype=np.uint8).T
            seeds = np.asarray(handle["receiver_noise_seed"][:, self.indices], dtype=np.int64).reshape(-1)
            self.sample_count = int(handle["iq_complex"].shape[1])
        if labels.shape != (len(self.indices), len(self.label_indices)):
            raise ValueError(f"nine-class labels shape drift: {labels.shape}")
        self.labels = labels
        self.receiver_seeds = seeds

    def __len__(self):
        return int(len(self.indices))

    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(str(self.path), "r")
        return self._handle

    def __getitem__(self, position):
        pos = int(position)
        index = int(self.indices[pos])
        stored = np.asarray(self._get_handle()["iq_complex"][:, index])
        if stored.dtype.fields is None:
            raise TypeError("iq_complex must be an HDF5 compound real/imag dataset")
        real = np.asarray(stored["real"], dtype=np.float32)
        imag = np.asarray(stored["imag"], dtype=np.float32)
        signal = np.stack([real, imag], axis=0)
        rms = np.sqrt(np.mean(signal[0] ** 2 + signal[1] ** 2))
        if np.isfinite(rms) and rms > 0:
            signal = signal / np.sqrt(rms * rms + 1e-12)
        return {
            "signal": torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)),
            "label": torch.from_numpy(self.labels[pos].astype(np.float32)),
            "sample_index": index,
        }

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()


class F2HeadWrapper(nn.Module):
    """Frozen/shared F2 feature extractor plus an arbitrary nine-class head."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, values):
        return self.classifier(self.backbone.forward_features(values))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(targets: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    predictions = (probabilities >= threshold).astype(np.uint8)
    per_label = {}
    auprs, aurocs = [], []
    for index, name in enumerate(CLASS_NAMES):
        truth = targets[:, index]
        score = probabilities[:, index]
        aupr = float(average_precision_score(truth, score))
        auroc = float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else float("nan")
        positive = truth == 1
        negative = ~positive
        per_label[name] = {
            "aupr": aupr,
            "auroc": auroc,
            "recall": float(predictions[positive, index].mean()) if positive.any() else float("nan"),
            "negative_fpr": float(predictions[negative, index].mean()) if negative.any() else float("nan"),
            "f1": float(f1_score(truth, predictions[:, index], zero_division=0)),
            "positive_count": int(positive.sum()),
        }
        auprs.append(aupr)
        aurocs.append(auroc)
    powers = (2 ** np.arange(targets.shape[1], dtype=np.int64))[None, :]
    target_codes = (targets.astype(np.int64) * powers).sum(axis=1).astype(int)
    prediction_codes = (predictions.astype(np.int64) * powers).sum(axis=1).astype(int)
    pair_counts = {}
    for truth, predicted in zip(target_codes.tolist(), prediction_codes.tolist()):
        key = f"{truth:03d}->{predicted:03d}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    top_errors = sorted(
        [{"transition": key, "count": count} for key, count in pair_counts.items() if key.split("->")[0] != key.split("->")[1]],
        key=lambda row: (-row["count"], row["transition"]),
    )[:20]
    return {
        "macro_aupr": float(np.nanmean(auprs)),
        "macro_auroc": float(np.nanmean(aurocs)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(np.mean(predictions != targets)),
        "subset_accuracy": float(np.mean(np.all(predictions == targets, axis=1))),
        "label_cardinality_error": float(np.mean(np.abs(predictions.sum(1) - targets.sum(1)))),
        "target_unique_combinations": int(len(np.unique(target_codes))),
        "predicted_unique_combinations": int(len(np.unique(prediction_codes))),
        "combination_exact_accuracy": float(np.mean(target_codes == prediction_codes)),
        "combination_error_top20": top_errors,
        "per_label": per_label,
    }


def profile_cost(model: nn.Module, device: torch.device) -> dict:
    macs = 0
    handles = []
    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            batch, channels, length = output.shape
            macs += int(batch * channels * length * (module.in_channels // module.groups) * module.kernel_size[0])
        elif isinstance(module, nn.Linear):
            macs += int(np.prod(output.shape[:-1]) * module.in_features * module.out_features)
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    values = torch.randn(1, 2, 3000, device=device)
    with torch.no_grad():
        model(values)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(20):
            model(values)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0 / 20.0
    for handle in handles:
        handle.remove()
    return {"conv_linear_macs_per_sample": int(macs), "cuda_batch1_latency_ms": float(latency_ms)}


def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch["signal"].to(device, non_blocking=True)).detach().cpu().numpy())
            labels.append(batch["label"].numpy())
    return np.concatenate(labels).astype(np.uint8), np.concatenate(logits).astype(np.float32)


def _load_f2_initialization(model: nn.Module, checkpoint_path: Path) -> dict:
    """Load only the shared F2 feature extractor from a validated six-class checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    full_state = checkpoint.get("model_state_dict", checkpoint)
    state = {}
    for key, value in full_state.items():
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        if key.startswith("classifier."):
            continue
        state[key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith(("classifier.", "cross_stage_pyramid.", "local_statistics_bypass."))]
    allowed_extra = ("label_feature_bank.", "prototype_memory.")
    unexpected_core = [key for key in unexpected if not key.startswith(allowed_extra)]
    if missing or unexpected_core:
        raise RuntimeError(f"F2 checkpoint incompatibility: missing={missing[:5]}, unexpected={unexpected_core[:5]}")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
    }


def train_one(name: str, seed: int, train_path: Path, train_indices: np.ndarray, eval_sets: dict[int, tuple[Path, np.ndarray]], args, output: Path) -> dict:
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("This experiment is GPU-only; no CPU fallback is allowed")
    if name in ("A0_F2_frozen_linear", "A0b_F2_frozen_mlp"):
        backbone = build_backbone("dfms_iq", num_classes=9, width=16, dropout=0.2).to(device)
        init_meta = _load_f2_initialization(backbone, args.f2_checkpoint_root / f"seed_{seed}" / "backbone" / "best_model.pth")
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        # Preserve the exact F2 feature extractor and replace only the 9-class head.
        if name == "A0_F2_frozen_linear":
            head = nn.Sequential(nn.Dropout(0.2), nn.Linear(backbone.feature_dim, 9)).to(device)
        else:
            head = nn.Sequential(
                nn.Linear(backbone.feature_dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 9)
            ).to(device)
        model = F2HeadWrapper(backbone, head)
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif name == "A1_F2_finetune":
        model = build_backbone("dfms_iq", num_classes=9, width=16, dropout=0.2).to(device)
        init_meta = _load_f2_initialization(model, args.f2_checkpoint_root / f"seed_{seed}" / "backbone" / "best_model.pth")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif name in ("A3_F2_CSFP", "A4_F2_LocalStatsK16"):
        variant = "dfms_iq_csfp" if name == "A3_F2_CSFP" else "dfms_iq_local_topk16"
        model = build_backbone(variant, num_classes=9, width=16, dropout=0.2).to(device)
        init_meta = _load_f2_initialization(model, args.f2_checkpoint_root / f"seed_{seed}" / "backbone" / "best_model.pth")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif name == "A2_EfficientNetB1":
        model = build_model("efficientnet_b1_1d", num_classes=9, dropout=0.2).to(device)
        init_meta = {"checkpoint": None, "checkpoint_sha256": None, "checkpoint_best_epoch": None, "checkpoint_selection_metric": None}
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        raise ValueError(name)
    criterion = nn.BCEWithLogitsLoss()
    train_ds = NineClassDataset(train_path, train_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    history = []
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                values = batch["signal"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(values), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    finally:
        train_ds.close()
    result = {"model": name, "seed": seed, "status": "completed", "initialization": init_meta, "trainable_parameters": trainable, "epochs": args.epochs, "history": history, "power_metrics": {}, "elapsed_seconds": time.time() - started}
    model_path = output / f"{name}__seed_{seed}.pth"
    torch.save(model.state_dict(), model_path)
    for power_db, (path, indices) in eval_sets.items():
        ds = NineClassDataset(path, indices)
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        try:
            targets, logits = predict(model, loader, device)
        finally:
            ds.close()
        result["power_metrics"][str(power_db)] = metrics(targets, logits)
    result["checkpoint_sha256"] = sha256(model_path)
    result["total_parameters"] = int(sum(p.numel() for p in model.parameters()))
    result["cost_profile"] = profile_cost(model, device)
    result["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    write_json(output / f"{name}__seed_{seed}.json", result)
    return result


def run(args):
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root
    train_path = find_file(data_root, 30, "train")
    # Disjoint identity groups: training uses a fixed prefix of 30 dB train;
    # every evaluation role is from the independent test MAT and never reused.
    with h5py.File(str(train_path), "r") as handle:
        n_train = int(handle["iq_complex"].shape[1])
    rng = np.random.default_rng(7000 + int(args.seed_offset))
    permutation = rng.permutation(n_train)
    train_indices = np.sort(permutation[: args.train_samples])
    eval_sets = {}
    for power_db in POWERS_DB:
        path = find_file(data_root, power_db, "test")
        with h5py.File(str(path), "r") as handle:
            count = int(handle["iq_complex"].shape[1])
        local_rng = np.random.default_rng(9000 + power_db + int(args.seed_offset))
        eval_sets[power_db] = (path, np.sort(local_rng.permutation(count)[: args.eval_samples]))
    # Explicit role manifest.  The development evaluation samples come from
    # the ordinary nine-class test MATs, never from a locked-test artifact.
    role_manifest = {
        "schema": "route-a-nineclass-role-manifest-v1",
        "fit": {"source_role": "train", "path": str(train_path), "count": int(len(train_indices)), "sample_indices": train_indices.tolist(), "receiver_noise_seed": receiver_seeds(train_path, train_indices)},
        "validation": {"source_role": "train", "path": str(train_path), "count": 0, "sample_indices": [], "used": False},
        "early": {"source_role": "development", "count": 0, "sample_indices": [], "used": False},
        "calibration": {"source_role": "development", "count": 0, "sample_indices": [], "used": False},
        "development": {"source_role": "test_mat_development_only", "powers_db": list(POWERS_DB), "count_per_power": int(args.eval_samples), "sample_indices_by_power": {str(power): indices.tolist() for power, (_, indices) in eval_sets.items()}, "receiver_noise_seed_by_power": {str(power): receiver_seeds(path, indices) for power, (path, indices) in eval_sets.items()}, "used": True},
        "test": {"source_role": "locked_test", "count": 0, "sample_indices": [], "used": False, "locked_test_used": False},
        "identity_policy": "receiver_noise_seed is audited; train and development evaluation are disjoint MAT roles; no power view is counted as an independent identity",
        "locked_test_used": False,
    }
    write_json(output / "role_manifest.json", role_manifest)
    split_payload = {"fit_indices": train_indices}
    split_payload.update({f"development_{power}dB_indices": indices for power, (_, indices) in eval_sets.items()})
    np.savez_compressed(output / "split_manifest.npz", **split_payload)
    config = {
        "schema": "route-a-nineclass-feature-sufficiency-screen-v1",
        "data_root": str(data_root), "train_path": str(train_path),
        "class_names": list(CLASS_NAMES), "powers_db": list(POWERS_DB),
        "train_samples": int(args.train_samples), "eval_samples_per_power": int(args.eval_samples),
        "batch_size": int(args.batch_size), "eval_batch_size": int(args.eval_batch_size),
        "epochs": int(args.epochs), "lr": float(args.lr), "lr_head": float(args.lr_head),
        "weight_decay": float(args.weight_decay), "seeds": list(args.seeds),
        "f2_checkpoint_root": str(args.f2_checkpoint_root),
        "locked_test_used": False, "test_files_opened": False,
        "role_policy": "30dB train prefix for fit; independent test MAT per power for development evaluation; validation/early/calibration/test roles are explicitly empty or sealed",
        "threshold_policy": "fixed probability 0.5 for readiness screen; no threshold tuning",
        "gate_policy": {"frozen_to_finetune_gain_tolerance_macro_aupr": 0.02, "strong_reference_gain_tolerance_macro_aupr": 0.02, "status": "not_validated_until_3_seed_formal_run", "frozen_f2_checkpoint_available": False},
    }
    write_json(output / "experiment_config.json", config)
    # Run the three requested model cells.  The command is resumable per JSON artifact.
    rows = []
    model_names = ("A0_F2_frozen_linear", "A0b_F2_frozen_mlp", "A1_F2_finetune", "A2_EfficientNetB1", "A3_F2_CSFP", "A4_F2_LocalStatsK16")
    for seed in args.seeds:
        for name in model_names:
            artifact = output / f"{name}__seed_{seed}.json"
            if args.resume and artifact.is_file():
                rows.append(json.loads(artifact.read_text(encoding="utf-8")))
            else:
                rows.append(train_one(name, int(seed), train_path, train_indices, eval_sets, args, output))
    # Summarize per-model/per-power curves and make a conservative feature gate.
    summary = {"schema": "route-a-nineclass-feature-sufficiency-summary-v1", "status": "development_screen_completed", "locked_test_used": False, "models": {}}
    for name in model_names:
        cells = [row for row in rows if row["model"] == name]
        summary["models"][name] = {
            "seeds": [int(row["seed"]) for row in cells],
            "macro_aupr_by_power": {str(power): [row["power_metrics"][str(power)]["macro_aupr"] for row in cells] for power in POWERS_DB},
            "macro_f1_by_power": {str(power): [row["power_metrics"][str(power)]["macro_f1"] for row in cells] for power in POWERS_DB},
            "mean_macro_aupr_30dB": float(np.mean([row["power_metrics"]["30"]["macro_aupr"] for row in cells])),
            "mean_worst_power_macro_aupr": float(np.mean([min(row["power_metrics"][str(power)]["macro_aupr"] for power in POWERS_DB) for row in cells])),
            "mean_min_class_aupr_30dB": float(np.mean([min(v["aupr"] for v in row["power_metrics"]["30"]["per_label"].values()) for row in cells])),
            "parameter_counts": [int(row["trainable_parameters"]) for row in cells],
        }
    a0 = summary["models"]["A0_F2_frozen_linear"]["mean_macro_aupr_30dB"]
    a0b = summary["models"]["A0b_F2_frozen_mlp"]["mean_macro_aupr_30dB"]
    a1 = summary["models"]["A1_F2_finetune"]["mean_macro_aupr_30dB"]
    a2 = summary["models"]["A2_EfficientNetB1"]["mean_macro_aupr_30dB"]
    frozen_gap = a1 - a0
    frozen_mlp_gap = a1 - a0b
    gain = a2 - a1
    by_seed = {}
    for seed in args.seeds:
        values = {name: next(row for row in rows if row["model"] == name and int(row["seed"]) == int(seed))["power_metrics"]["30"]["macro_aupr"] for name in ("A0_F2_frozen_linear", "A0b_F2_frozen_mlp", "A1_F2_finetune", "A2_EfficientNetB1")}
        by_seed[str(seed)] = {
            **values,
            "a1_minus_a0": float(values["A1_F2_finetune"] - values["A0_F2_frozen_linear"]),
            "a1_minus_a0b": float(values["A1_F2_finetune"] - values["A0b_F2_frozen_mlp"]),
            "a2_minus_a1": float(values["A2_EfficientNetB1"] - values["A1_F2_finetune"]),
        }
    summary["feature_sufficiency_gate"] = {
        "decision": "not_validated_formal_3seed_required" if len(args.seeds) < 3 else ("provisional_insufficient" if (frozen_gap > 0.02 or frozen_mlp_gap > 0.02 or gain > 0.02) else "provisional_sufficient"),
        "a1_minus_a0_macro_aupr_30dB": float(frozen_gap),
        "a1_minus_a0b_mlp_macro_aupr_30dB": float(frozen_mlp_gap),
        "a2_minus_a1_macro_aupr_30dB": float(gain),
        "per_seed_30dB_differences": by_seed,
        "a1_minus_a0_positive_seed_count": int(sum(row["a1_minus_a0"] > 0 for row in by_seed.values())),
        "a1_minus_a0b_positive_seed_count": int(sum(row["a1_minus_a0b"] > 0 for row in by_seed.values())),
        "frozen_to_finetune_tolerance": 0.02,
        "pre_registered_tolerance": 0.02,
        "six_class_selection_allowed": bool(len(args.seeds) >= 3 and frozen_gap <= 0.02 and frozen_mlp_gap <= 0.02 and gain <= 0.02),
        "reason": "This is a development screen only. A0/A1 now use the same traceable per-seed F2 checkpoint when supplied; six-class selection remains stopped until the formal three-seed gate, worst-power, weakest-class and combination checks pass.",
    }
    write_json(output / "feature_sufficiency_summary.json", summary)
    write_json(output / "gate_decision.json", {
        "schema": "route-a-nineclass-feature-sufficiency-gate-v1",
        "decision": summary["feature_sufficiency_gate"]["decision"],
        "six_class_selection_allowed": summary["feature_sufficiency_gate"]["six_class_selection_allowed"],
        "stop_reason": summary["feature_sufficiency_gate"]["reason"],
        "locked_test_used": False,
        "test_files_opened": False,
        "next_action": "formalize F2 representation improvements (CSFP/LocalStatsK16) with equal training budget before any 84-subset screen",
    })
    write_json(output / "terminal_receipt.json", {"status": "completed", "summary": str((output / 'feature_sufficiency_summary.json').resolve()), "locked_test_used": False, "test_files_opened": False, "script_sha256": sha256(Path(__file__))})
    print(json.dumps(summary["feature_sufficiency_gate"], ensure_ascii=False, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_nineclass_feature_sufficiency_20260826/model_screen")
    parser.add_argument("--f2-checkpoint-root", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
