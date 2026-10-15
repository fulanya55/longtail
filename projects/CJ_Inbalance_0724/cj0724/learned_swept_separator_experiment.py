"""Deployable single-view STFT separator distilled from oracle endpoint isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader

from cj0724.dual_axis import discover_power_train_files, read_identity_metadata
from cj0724.models import IQGroupedConvClassifier
from cj0724.oracle_swept_isolation_experiment import (
    SweptIsolationDataset,
    evaluate,
    make_loader,
    select_fpr_thresholds,
    set_seed,
    sha256_indices,
)


class SeparatorPairDataset(SweptIsolationDataset):
    def __init__(self, file_10, file_50, labels, indices):
        super().__init__(file_10, file_50, labels, indices, "raw_10db")

    def __getitem__(self, position):
        raw = super().__getitem__(position)
        original_mode = self.mode
        self.mode = "oracle_swept_endpoint"
        oracle = super().__getitem__(position)
        self.mode = original_mode
        return {
            "iq": raw["iq"],
            "target_iq": oracle["iq"],
            "label": raw["label"],
            "sample_index": raw["sample_index"],
        }


class ComplexSTFTSeparatorClassifier(nn.Module):
    def __init__(self, width: int = 16, classifier_width: int = 12, mask_bound: float = 2.5):
        super().__init__()
        self.mask_network = nn.Sequential(
            nn.Conv2d(6, width, 5, padding=2, bias=False), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, 4, 1),
        )
        nn.init.zeros_(self.mask_network[-1].weight)
        nn.init.zeros_(self.mask_network[-1].bias)
        self.classifier = IQGroupedConvClassifier(num_classes=3, width=classifier_width, dropout=0.1)
        self.mask_bound = float(mask_bound)
        passthrough_bias = float(np.arctanh(1.0 / self.mask_bound))
        with torch.no_grad():
            self.mask_network[-1].bias[:2].fill_(passthrough_bias)
        self.n_fft = 128
        self.hop_length = 32
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    @staticmethod
    def normalize_iq(iq: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(iq.square().sum(1).mean(-1, keepdim=True).clamp_min(1e-12))
        return iq / rms[:, None, :]

    def forward_components(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, length = iq.shape
        spectrum = torch.stft(
            iq.float().reshape(batch_size * 2, length),
            n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
            return_complex=True, center=True,
        ).reshape(batch_size, 2, self.n_fft // 2 + 1, -1)
        scale = spectrum.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        normalized = spectrum / scale
        feature = torch.cat(
            [normalized.real, normalized.imag, torch.log1p(normalized.abs())], dim=1
        )
        raw_mask = self.mask_network(feature)
        complex_mask = torch.complex(
            self.mask_bound * torch.tanh(raw_mask[:, :2]),
            self.mask_bound * torch.tanh(raw_mask[:, 2:]),
        )
        estimated_spectrum = spectrum * complex_mask
        estimated = torch.istft(
            estimated_spectrum.reshape(batch_size * 2, self.n_fft // 2 + 1, -1),
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, length=length, center=True,
        ).reshape(batch_size, 2, length)
        estimated_iq = estimated
        normalized_estimate = self.normalize_iq(estimated_iq)
        logits = self.classifier(normalized_estimate)
        return {
            "logits": logits,
            "estimated_iq": estimated_iq,
            "normalized_estimate": normalized_estimate,
            "mask": complex_mask,
        }

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.forward_components(iq)["logits"]


def separation_loss(
    components: dict[str, torch.Tensor], target_iq: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positive = labels.sum(1) > 0
    estimate = components["normalized_estimate"].float()
    target = target_iq.float()
    zero = estimate.sum() * 0.0
    if positive.any():
        estimate_flat = estimate[positive].flatten(1)
        target_flat = target[positive].flatten(1)
        cosine = 1.0 - F.cosine_similarity(estimate_flat, target_flat, dim=1).mean()
        time_l1 = F.l1_loss(estimate[positive], target[positive])
    else:
        cosine, time_l1 = zero, zero
    negative_energy = components["estimated_iq"][~positive].square().mean() if (~positive).any() else zero
    total = cosine + 0.25 * time_l1 + 0.1 * negative_energy
    return total, {
        "cosine": cosine.detach(), "time_l1": time_l1.detach(),
        "negative_energy": negative_energy.detach(),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probabilities, cosine_parts = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        target = batch["target_iq"].to(device)
        output = model.forward_components(iq)
        labels.append(batch["label"].numpy())
        probabilities.append(torch.sigmoid(output["logits"]).cpu().numpy())
        positive = batch["label"].sum(1) > 0
        if positive.any():
            cosine_parts.append(
                F.cosine_similarity(
                    output["normalized_estimate"][positive].flatten(1),
                    target[positive].flatten(1), dim=1,
                ).cpu().numpy()
            )
    return np.concatenate(labels).astype(np.uint8), np.concatenate(probabilities), float(np.mean(np.concatenate(cosine_parts)))


def train_one(seed, files, labels, roles, args, device):
    set_seed(seed)
    run_dir = Path(args.output_root) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        role: SeparatorPairDataset(files[10], files[50], labels, roles[role])
        for role in ("train", "early", "calibration", "confirm")
    }
    loaders = {
        role: make_loader(dataset, args.batch_size, role == "train", seed)
        for role, dataset in datasets.items()
    }
    model = ComplexSTFTSeparatorClassifier(args.separator_width, args.classifier_width, args.mask_bound).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = (-float("inf"), -float("inf"))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("loss", "bce", "separation", "cosine", "time_l1", "negative_energy")}
        samples = 0
        for batch in loaders["train"]:
            iq = batch["iq"].to(device)
            target_iq = batch["target_iq"].to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                components = model.forward_components(iq)
                bce = F.binary_cross_entropy_with_logits(components["logits"], target)
                sep, parts = separation_loss(components, target_iq, target)
                loss = bce + args.separation_weight * sep
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(target)
            totals["loss"] += float(loss.detach()) * count
            totals["bce"] += float(bce.detach()) * count
            totals["separation"] += float(sep.detach()) * count
            for name, value in parts.items():
                totals[name] += float(value) * count
            samples += count
        early_true, early_probability, early_cosine = predict(model, loaders["early"], device)
        auprs = [sk_metrics.average_precision_score(early_true[:, i], early_probability[:, i]) for i in range(3)]
        row = {
            "epoch": epoch, **{name: value / samples for name, value in totals.items()},
            "early_macro_aupr": float(np.mean(auprs)),
            "early_min_aupr": float(np.min(auprs)),
            "early_separator_cosine": early_cosine,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps({"seed": seed, **row}))
        key = (row["early_macro_aupr"], row["early_separator_cosine"])
        if key > best:
            best = key
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        scheduler.step()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    calibration_true, calibration_probability, _ = predict(model, loaders["calibration"], device)
    thresholds = select_fpr_thresholds(calibration_true, calibration_probability, args.fpr_cap)
    confirm_true, confirm_probability, confirm_cosine = predict(model, loaders["confirm"], device)
    result = {
        "seed": seed, "best_epoch": int(checkpoint["epoch"]),
        "thresholds": thresholds.tolist(), "separator_cosine": confirm_cosine,
        "confirm": evaluate(confirm_true, confirm_probability, thresholds),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for dataset in datasets.values():
        dataset.close()
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--oracle-root", default="outputs/oracle_swept_isolation_20260725")
    parser.add_argument("--output-root", default="outputs/learned_swept_separator_20260725")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--separator-width", type=int, default=16)
    parser.add_argument("--classifier-width", type=int, default=12)
    parser.add_argument("--mask-bound", type=float, default=2.5)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fpr-cap", type=float, default=0.10)
    args = parser.parse_args()
    roles_npz = np.load(Path(args.run_root) / "role_indices.npz")
    roles = {name: roles_npz[name].astype(np.int64) for name in roles_npz.files}
    files = discover_power_train_files(args.power_root)
    _, labels = read_identity_metadata(files[10])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "teacher": "oracle endpoint swept isolation, train-only",
        "inference": "single current 10 dB I/Q view",
        "separator": "complex STFT mask n_fft=128 hop=32",
        "roles": {name: {"count": len(value), "sha256": sha256_indices(value)} for name, value in roles.items()},
        "seeds": args.seeds, "epochs": args.epochs, "fpr_cap": args.fpr_cap,
        "locked_test_used": False,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [train_one(seed, files, labels, roles, args, device) for seed in args.seeds]
    aggregate = {"seed_count": len(results)}
    for metric in ("macro_aupr", "macro_recall", "macro_fpr", "macro_f1"):
        values = [row["confirm"][metric] for row in results]
        aggregate[f"{metric}_mean"] = float(np.mean(values))
        aggregate[f"{metric}_std"] = float(np.std(values))
    aggregate["separator_cosine_mean"] = float(np.mean([row["separator_cosine"] for row in results]))
    for i, name in enumerate(("SawFM", "VFT", "ISRJ")):
        for metric in ("aupr", "recall", "fpr"):
            values = [row["confirm"]["per_class"][i][metric] for row in results]
            aggregate[f"{name}_{metric}_mean"] = float(np.mean(values))
    oracle = json.loads((Path(args.oracle_root) / "results.json").read_text(encoding="utf-8"))
    raw = next(row for row in oracle["aggregates"] if row["mode"] == "raw_10db")
    upper = next(row for row in oracle["aggregates"] if row["mode"] == "oracle_swept_endpoint")
    aggregate["recall_gain_vs_raw"] = float(aggregate["macro_recall_mean"] - raw["macro_recall_mean"])
    aggregate["fraction_of_oracle_recall_gap_closed"] = float(
        (aggregate["macro_recall_mean"] - raw["macro_recall_mean"])
        / max(upper["macro_recall_mean"] - raw["macro_recall_mean"], 1e-12)
    )
    aggregate["promotion_pass"] = bool(
        aggregate["macro_recall_mean"] >= 0.70 and aggregate["macro_fpr_mean"] <= 0.10
        and all(aggregate[f"{name}_recall_mean"] >= 0.70 for name in ("SawFM", "VFT", "ISRJ"))
    )
    payload = {"aggregate": aggregate, "per_seed": results}
    (output_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
