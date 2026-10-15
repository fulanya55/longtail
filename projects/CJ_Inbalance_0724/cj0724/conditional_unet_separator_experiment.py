"""Conditional multi-scale STFT U-Net distilled from oracle swept isolation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sk_metrics

from cj0724.dual_axis import discover_power_train_files, read_identity_metadata
from cj0724.learned_swept_separator_experiment import SeparatorPairDataset
from cj0724.models import IQGroupedConvClassifier
from cj0724.multitask import build_multitask_model
from cj0724.oracle_swept_isolation_experiment import (
    evaluate,
    make_loader,
    select_fpr_thresholds,
    set_seed,
    sha256_indices,
)


PROFILES = {
    "unet_wave": {"condition": False, "distill_weight": 0.0},
    "unet_condition": {"condition": True, "distill_weight": 0.0},
    "unet_condition_distill": {"condition": True, "distill_weight": 0.25},
}


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ConditionalComplexUNetSeparator(nn.Module):
    def __init__(
        self,
        width: int = 12,
        classifier_width: int = 12,
        condition: bool = True,
        maximum_phase: float = 1.5707963267948966,
    ) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(6, width)
        self.encoder2 = ConvBlock(width, width * 2)
        self.bottleneck = ConvBlock(width * 2, width * 4)
        self.decoder2 = ConvBlock(width * 6, width * 2)
        self.decoder1 = ConvBlock(width * 3, width)
        self.output = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.condition_projection = nn.Linear(6, width * 8)
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        self.classifier = IQGroupedConvClassifier(num_classes=3, width=classifier_width, dropout=0.1)
        self.condition = bool(condition)
        self.maximum_phase = float(maximum_phase)
        self.n_fft = 128
        self.hop_length = 32
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    @staticmethod
    def normalize_iq(iq: torch.Tensor) -> torch.Tensor:
        energy = iq.square().sum(1).mean(-1, keepdim=True).clamp_min(1e-12)
        return iq / torch.sqrt(energy)[:, None, :]

    def forward_components(
        self, iq: torch.Tensor, condition_logits: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
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
        encoder1 = self.encoder1(feature)
        encoder2 = self.encoder2(F.avg_pool2d(encoder1, 2, ceil_mode=True))
        bottleneck = self.bottleneck(F.avg_pool2d(encoder2, 2, ceil_mode=True))
        if self.condition:
            if condition_logits is None:
                raise ValueError("condition logits are required")
            affine = self.condition_projection(condition_logits.detach())
            gamma, beta = affine.chunk(2, dim=1)
            bottleneck = bottleneck * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        up2 = F.interpolate(bottleneck, size=encoder2.shape[-2:], mode="bilinear", align_corners=False)
        decoder2 = self.decoder2(torch.cat([up2, encoder2], dim=1))
        up1 = F.interpolate(decoder2, size=encoder1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decoder1(torch.cat([up1, encoder1], dim=1))
        raw = self.output(decoded)
        magnitude_mask = 2.0 * torch.sigmoid(raw[:, :2])
        phase_residual = self.maximum_phase * torch.tanh(raw[:, 2:])
        complex_mask = torch.polar(magnitude_mask, phase_residual)
        estimated_spectrum = spectrum * complex_mask
        estimated = torch.istft(
            estimated_spectrum.reshape(batch_size * 2, self.n_fft // 2 + 1, -1),
            n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
            length=length, center=True,
        ).reshape(batch_size, 2, length)
        normalized_estimate = self.normalize_iq(estimated)
        return {
            "logits": self.classifier(normalized_estimate),
            "estimated_iq": estimated,
            "normalized_estimate": normalized_estimate,
            "magnitude_mask": magnitude_mask,
            "phase_residual": phase_residual,
        }


def classifier_logits(model: nn.Module, iq: torch.Tensor) -> torch.Tensor:
    output = model(iq)
    return output[0] if isinstance(output, tuple) else output


def multi_resolution_spectral_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size, channels, length = estimate.shape
    losses = []
    for n_fft, hop in ((64, 16), (128, 32), (256, 64)):
        window = torch.hann_window(n_fft, device=estimate.device)
        estimate_spectrum = torch.stft(
            estimate.reshape(batch_size * channels, length), n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True,
        )
        target_spectrum = torch.stft(
            target.reshape(batch_size * channels, length), n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True,
        )
        losses.append(
            F.l1_loss(torch.log1p(estimate_spectrum.abs()), torch.log1p(target_spectrum.abs()))
        )
    return torch.stack(losses).mean()


def objective(
    components: dict[str, torch.Tensor],
    target_iq: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    separation_weight: float,
    distill_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bce = F.binary_cross_entropy_with_logits(components["logits"], labels)
    positive = labels.sum(1) > 0
    zero = bce * 0.0
    if positive.any():
        estimate = components["normalized_estimate"][positive]
        target = target_iq[positive]
        cosine = 1.0 - F.cosine_similarity(estimate.flatten(1), target.flatten(1), dim=1).mean()
        time_l1 = F.l1_loss(estimate, target)
        spectral = multi_resolution_spectral_loss(estimate, target)
    else:
        cosine = time_l1 = spectral = zero
    negative_energy = (
        components["estimated_iq"][~positive].square().mean() if (~positive).any() else zero
    )
    separation = cosine + 0.2 * time_l1 + 0.25 * spectral + 0.1 * negative_energy
    distillation = F.binary_cross_entropy_with_logits(
        components["logits"], torch.sigmoid(teacher_logits.detach())
    )
    total = bce + float(separation_weight) * separation + float(distill_weight) * distillation
    return total, {
        "bce": bce.detach(), "separation": separation.detach(), "cosine": cosine.detach(),
        "time_l1": time_l1.detach(), "spectral": spectral.detach(),
        "negative_energy": negative_energy.detach(), "distillation": distillation.detach(),
    }


@torch.no_grad()
def predict(
    model: ConditionalComplexUNetSeparator,
    base_model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    base_model.eval()
    labels, probabilities, cosines = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device)
        target = batch["target_iq"].to(device)
        base_logits = classifier_logits(base_model, iq)
        components = model.forward_components(iq, base_logits)
        labels.append(batch["label"].numpy())
        probabilities.append(torch.sigmoid(components["logits"]).cpu().numpy())
        positive = batch["label"].sum(1) > 0
        if positive.any():
            cosines.append(F.cosine_similarity(
                components["normalized_estimate"][positive].flatten(1),
                target[positive].flatten(1), dim=1,
            ).cpu().numpy())
    return np.concatenate(labels).astype(np.uint8), np.concatenate(probabilities), float(np.mean(np.concatenate(cosines)))


def build_frozen_models(seed, selected_row, protocol, oracle_root, device):
    base = build_multitask_model(
        selected_row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
        protocol["parameter_hidden_dim"], protocol["legacy_attention"],
    ).to(device)
    base.load_state_dict(torch.load(selected_row["checkpoint"], map_location=device, weights_only=False)["model_state"])
    teacher = IQGroupedConvClassifier(num_classes=3, width=12, dropout=0.1).to(device)
    teacher.load_state_dict(torch.load(
        Path(oracle_root) / "oracle_swept_endpoint" / f"seed_{seed}" / "best.pt",
        map_location=device, weights_only=False,
    )["model_state"])
    for frozen in (base, teacher):
        frozen.eval()
        for parameter in frozen.parameters(): parameter.requires_grad_(False)
    return base, teacher


def train_one(profile_name, seed, selected_row, protocol, files, labels, roles, args, device):
    set_seed(seed)
    config = PROFILES[profile_name]
    run_dir = Path(args.output_root) / profile_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = {role: SeparatorPairDataset(files[10], files[50], labels, roles[role]) for role in ("train", "early", "calibration", "tune", "confirm")}
    loaders = {role: make_loader(dataset, args.batch_size, role == "train", seed) for role, dataset in datasets.items()}
    base, teacher = build_frozen_models(seed, selected_row, protocol, args.oracle_root, device)
    model = ConditionalComplexUNetSeparator(
        width=args.width, classifier_width=args.classifier_width, condition=config["condition"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_key = (-float("inf"), -float("inf"))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals, samples = {}, 0
        for batch in loaders["train"]:
            iq = batch["iq"].to(device)
            target_iq = batch["target_iq"].to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                base_logits = classifier_logits(base, iq)
                teacher_logits = teacher(target_iq)
            components = model.forward_components(iq, base_logits)
            loss, parts = objective(
                components, target_iq, target, teacher_logits,
                args.separation_weight, config["distill_weight"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(target)
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * count
            for name, value in parts.items(): totals[name] = totals.get(name, 0.0) + float(value) * count
            samples += count
        early_true, early_probability, early_cosine = predict(model, base, loaders["early"], device)
        auprs = [sk_metrics.average_precision_score(early_true[:, i], early_probability[:, i]) for i in range(3)]
        row = {
            "epoch": epoch, **{name: value / samples for name, value in totals.items()},
            "early_macro_aupr": float(np.mean(auprs)), "early_min_aupr": float(np.min(auprs)),
            "early_separator_cosine": early_cosine,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps({"profile": profile_name, "seed": seed, **row}))
        key = (row["early_macro_aupr"], row["early_separator_cosine"])
        if key > best_key:
            best_key = key
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "profile": profile_name}, run_dir / "best.pt")
        scheduler.step()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    calibration_true, calibration_probability, _ = predict(model, base, loaders["calibration"], device)
    thresholds = select_fpr_thresholds(calibration_true, calibration_probability, args.fpr_cap)
    tune_true, tune_probability, tune_cosine = predict(model, base, loaders["tune"], device)
    result = {
        "profile": profile_name, "seed": seed, "best_epoch": int(checkpoint["epoch"]),
        "thresholds": thresholds.tolist(), "tune_separator_cosine": tune_cosine,
        "tune": evaluate(tune_true, tune_probability, thresholds),
    }
    if args.evaluate_confirm:
        confirm_true, confirm_probability, confirm_cosine = predict(model, base, loaders["confirm"], device)
        result["confirm_separator_cosine"] = confirm_cosine
        result["confirm"] = evaluate(confirm_true, confirm_probability, thresholds)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for dataset in datasets.values(): dataset.close()
    del model, base, teacher
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--oracle-root", default="outputs/oracle_swept_isolation_20260725")
    parser.add_argument("--baseline-root", default="outputs/learned_swept_separator_fp32_v2_20260725")
    parser.add_argument("--output-root", default="outputs/conditional_unet_separator_20260725")
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=["unet_condition_distill"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--classifier-width", type=int, default=12)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fpr-cap", type=float, default=0.10)
    parser.add_argument("--evaluate-confirm", action="store_true")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    selected = {int(row["seed"]): row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}
    roles_npz = np.load(run_root / "role_indices.npz")
    roles = {name: roles_npz[name].astype(np.int64) for name in roles_npz.files}
    files = discover_power_train_files(args.power_root)
    _, labels = read_identity_metadata(files[10])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "profiles": {name: PROFILES[name] for name in args.profiles},
        "seeds": args.seeds, "epochs": args.epochs,
        "checkpoint_role": "early", "threshold_role": "calibration",
        "profile_selection_role": "tune", "confirm_evaluated": args.evaluate_confirm,
        "inference": "single current I/Q plus frozen current-view logits; no power input",
        "oracle_teacher": "training only", "locked_test_used": False,
        "roles": {name: {"count": len(value), "sha256": sha256_indices(value)} for name, value in roles.items()},
    }
    (output_root / "protocol.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [train_one(profile, seed, selected[seed], protocol, files, labels, roles, args, device) for profile in args.profiles for seed in args.seeds]
    (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    aggregates = []
    metric_role = "confirm" if args.evaluate_confirm else "tune"
    for profile in args.profiles:
        rows = [row for row in results if row["profile"] == profile]
        entry = {"profile": profile, "seed_count": len(rows), "metric_role": metric_role}
        for metric in ("macro_aupr", "macro_recall", "macro_fpr", "macro_f1"):
            values = [row[metric_role][metric] for row in rows]
            entry[f"{metric}_mean"] = float(np.mean(values)); entry[f"{metric}_std"] = float(np.std(values))
        for i, name in enumerate(("SawFM", "VFT", "ISRJ")):
            entry[f"{name}_recall_mean"] = float(np.mean([row[metric_role]["per_class"][i]["recall"] for row in rows]))
        entry["promotion_pass"] = bool(
            entry["macro_recall_mean"] >= 0.70 and entry["macro_fpr_mean"] <= 0.10
            and all(entry[f"{name}_recall_mean"] >= 0.70 for name in ("SawFM", "VFT", "ISRJ"))
        )
        aggregates.append(entry)
    (output_root / "aggregate.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
