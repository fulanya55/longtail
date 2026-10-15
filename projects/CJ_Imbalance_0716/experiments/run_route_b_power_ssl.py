"""Screen unlabeled paired-power pretraining with frozen-backbone probes."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.augmentations import weak_complex_augmentation  # noqa: E402
from cjlt.data.power_pair_dataset import PowerPairDataset  # noqa: E402
from cjlt.data.power_pairs import build_role_bound_power_pair_index  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.run_manager import configuration_sha256  # noqa: E402
from cjlt.losses.classification import build_classification_loss  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.power_ssl import (  # noqa: E402
    PowerContrastiveModel,
    contrastive_similarity_diagnostics,
    symmetric_contrastive_loss,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
from experiments.run_student_b import build_student_b_plan  # noqa: E402


VARIANTS = ("RandomInit", "SameViewAug", "RandomSource", "SameSourcePower")


def _write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _limited_records(records, maximum_identities):
    maximum = int(maximum_identities)
    if maximum <= 0:
        return list(records)
    selected_identities = []
    seen = set()
    for record in records:
        if record.identity not in seen:
            seen.add(record.identity)
            selected_identities.append(record.identity)
            if len(selected_identities) == maximum:
                break
    allowed = set(selected_identities)
    return [record for record in records if record.identity in allowed]


def _build_backbone(args, num_classes):
    return build_backbone(
        args.backbone,
        num_classes=num_classes,
        width=args.width,
        dropout=args.dropout,
    )


def _pretrain(args, variant, loader, num_classes, device, output_dir):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(int(args.seed) + 505)
    backbone = _build_backbone(args, num_classes)
    model = PowerContrastiveModel(
        backbone,
        projection_dim=args.projection_dim,
        hidden_dim=args.projection_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    generator = torch.Generator().manual_seed(int(args.seed) + 707)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        diagnostics = []
        for batch in loader:
            reference = batch["reference_signal"]
            paired = batch["paired_signal"]
            if variant == "SameViewAug":
                first = weak_complex_augmentation(reference, [], generator=generator).signal
                second = weak_complex_augmentation(reference, [], generator=generator).signal
            else:
                first = reference
                second = paired
                if variant == "RandomSource":
                    identities = torch.as_tensor(batch["identity"])
                    shift = next(
                        (
                            value
                            for value in range(1, len(identities))
                            if bool(
                                torch.all(
                                    identities
                                    != torch.roll(identities, shifts=value, dims=0)
                                )
                            )
                        ),
                        None,
                    )
                    if shift is None:
                        raise RuntimeError(
                            "RandomSource batch cannot form a different-identity derangement"
                        )
                    second = torch.roll(second, shifts=shift, dims=0)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _, first_projection = model(first.to(device))
                _, second_projection = model(second.to(device))
                loss = symmetric_contrastive_loss(
                    first_projection, second_projection, temperature=args.temperature
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu().item()) * len(first)
            sample_count += len(first)
            if not diagnostics:
                diagnostics.append(
                    contrastive_similarity_diagnostics(
                        first_projection.detach(), second_projection.detach()
                    )
                )
        row = {
            "epoch": epoch,
            "pretrain_loss": loss_sum / max(sample_count, 1),
            "samples": sample_count,
            **(diagnostics[0] if diagnostics else {}),
        }
        history.append(row)
    checkpoint = {
        "backbone_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.backbone.state_dict().items()
        },
        "pretrain_variant": variant,
        "history": history,
        "train_seconds": time.perf_counter() - started,
    }
    torch.save(checkpoint, str(output_dir / "pretrained_backbone.pth"))
    _write(output_dir / "pretrain_history.json", history)
    return checkpoint


@torch.no_grad()
def _extract(backbone, loader, device):
    backbone.to(device).eval()
    feature_parts = []
    target_parts = []
    for batch in loader:
        feature_parts.append(
            backbone.forward_features(batch["signal"].to(device)).cpu()
        )
        target_parts.append(batch["label"].float())
    return torch.cat(feature_parts), torch.cat(target_parts)


def _probe(args, backbone, labeled_loader, early_loader, device, output_dir):
    train_features, train_targets = _extract(backbone, labeled_loader, device)
    early_features, early_targets = _extract(backbone, early_loader, device)
    generator = torch.Generator().manual_seed(int(args.seed) + 909)
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    torch.manual_seed(int(args.seed) + 808)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + 808)
    head = nn.Linear(backbone.feature_dim, train_targets.shape[1]).to(device)
    loss_fn = build_classification_loss("asl")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.probe_learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_aupr = -float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.probe_epochs + 1):
        head.train()
        loss_sum = 0.0
        count = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            logits = head(features.to(device))
            loss = loss_fn(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * len(features)
            count += len(features)
        head.eval()
        with torch.no_grad():
            logits = head(early_features.to(device)).cpu().numpy()
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        prediction = (probability >= 0.5).astype(np.uint8)
        metrics = compute_multilabel_metrics(
            early_targets.numpy().astype(np.uint8),
            prediction,
            probabilities=probability,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "macro_aupr": metrics["macro_aupr"],
                "macro_f1": metrics["macro_f1"],
                "hamming_loss": metrics["hamming_loss"],
            }
        )
        if float(metrics["macro_aupr"]) > best_aupr:
            best_aupr = float(metrics["macro_aupr"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
    head.load_state_dict(best_state)
    torch.save(
        {"head_state_dict": best_state, "best_epoch": best_epoch, "history": history},
        str(output_dir / "linear_probe.pth"),
    )
    _write(output_dir / "probe_history.json", history)
    best = next(row for row in history if row["epoch"] == best_epoch)
    return {"best_epoch": best_epoch, **best}


def _finetune(args, backbone, labeled_loader, early_loader, device, output_dir):
    torch.manual_seed(int(args.seed) + 1001)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + 1001)
    backbone.to(device)
    loss_fn = build_classification_loss("asl")
    optimizer = torch.optim.AdamW(
        backbone.parameters(),
        lr=args.finetune_learning_rate,
        weight_decay=args.weight_decay,
    )
    best_state = None
    best_aupr = -float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.finetune_epochs + 1):
        backbone.train()
        loss_sum = 0.0
        count = 0
        for batch in labeled_loader:
            optimizer.zero_grad()
            logits = backbone(batch["signal"].to(device))
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * len(logits)
            count += len(logits)
        backbone.eval()
        logits_parts = []
        target_parts = []
        with torch.no_grad():
            for batch in early_loader:
                logits_parts.append(backbone(batch["signal"].to(device)).cpu().numpy())
                target_parts.append(batch["label"].numpy())
        logits = np.concatenate(logits_parts, axis=0)
        target = np.concatenate(target_parts, axis=0).astype(np.uint8)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        prediction = (probability >= 0.5).astype(np.uint8)
        metrics = compute_multilabel_metrics(
            target, prediction, probabilities=probability
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "macro_aupr": metrics["macro_aupr"],
                "macro_f1": metrics["macro_f1"],
                "hamming_loss": metrics["hamming_loss"],
            }
        )
        if float(metrics["macro_aupr"]) > best_aupr:
            best_aupr = float(metrics["macro_aupr"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in backbone.state_dict().items()
            }
    backbone.load_state_dict(best_state)
    torch.save(
        {"model_state_dict": best_state, "best_epoch": best_epoch, "history": history},
        str(output_dir / "finetuned_backbone.pth"),
    )
    _write(output_dir / "finetune_history.json", history)
    best = next(row for row in history if row["epoch"] == best_epoch)
    return {"best_epoch": best_epoch, **best}


def run(args):
    config = load_config(args.config)
    base_plan = build_run_plan(
        config,
        loss_name="asl",
        seed=args.seed,
        max_train_samples=0,
        max_eval_samples=args.max_eval_samples,
    )
    semi_plan = build_student_b_plan(
        config,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
        max_labeled_samples=0,
        max_unlabeled_samples=0,
        max_eval_samples=args.max_eval_samples,
    )
    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    pair_index = build_role_bound_power_pair_index(
        _data_path(config),
        power_root,
        # Keep the representation stage genuinely label-free: identities later
        # allocated to the labeled subset are excluded from SSL pretraining.
        base_indices=semi_plan["unlabeled_indices"],
        levels=args.power_levels,
        reference_level=args.reference_level,
    )
    records = _limited_records(pair_index.records, args.max_pretrain_identities)
    pair_dataset = PowerPairDataset(
        power_root,
        records,
        class_mode=base_plan["class_mode"],
        normalization=config["normalization"],
        include_parameters=False,
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed) + 505),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    common = {
        "path": _data_path(config),
        "class_mode": base_plan["class_mode"],
        "normalization": config["normalization"],
        "include_parameters": False,
    }
    labeled_dataset = V73SignalDataset(indices=semi_plan["labeled_indices"], **common)
    early_dataset = V73SignalDataset(indices=base_plan["early_stop_indices"], **common)
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    early_loader = DataLoader(early_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(args.output_root)
    results = []
    for variant in VARIANTS:
        output_dir = root / variant / "seed_{}".format(args.seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        if variant == "RandomInit":
            torch.manual_seed(args.seed)
            backbone = _build_backbone(args, base_plan["num_classes"])
            pretrain = {"pretrain_variant": variant, "history": [], "train_seconds": 0.0}
        else:
            pretrain = _pretrain(
                args, variant, pair_loader, base_plan["num_classes"], device, output_dir
            )
            backbone = _build_backbone(args, base_plan["num_classes"])
            backbone.load_state_dict(pretrain["backbone_state_dict"])
        probe = _probe(args, backbone, labeled_loader, early_loader, device, output_dir)
        finetune_backbone = _build_backbone(args, base_plan["num_classes"])
        finetune_backbone.load_state_dict(backbone.state_dict())
        finetune = _finetune(
            args,
            finetune_backbone,
            labeled_loader,
            early_loader,
            device,
            output_dir,
        )
        receipt = {
            "variant": variant,
            "seed": args.seed,
            "pretrain_epochs": 0 if variant == "RandomInit" else args.pretrain_epochs,
            "pretrain_identities": len(set(record.identity for record in records)),
            "pretrain_records": len(records),
            "linear_probe_macro_aupr": probe["macro_aupr"],
            "linear_probe_macro_f1_fixed_0p5": probe["macro_f1"],
            "linear_probe_hamming_fixed_0p5": probe["hamming_loss"],
            "best_probe_epoch": probe["best_epoch"],
            "finetune_macro_aupr": finetune["macro_aupr"],
            "finetune_macro_f1_fixed_0p5": finetune["macro_f1"],
            "finetune_hamming_fixed_0p5": finetune["hamming_loss"],
            "best_finetune_epoch": finetune["best_epoch"],
            "pretrain_seconds": pretrain["train_seconds"],
            "split_sha256": base_plan["split_sha256"],
            "semi_split_sha256": semi_plan["semi_split_sha256"],
            "pair_sha256": pair_index.sha256,
            "locked_test_used": False,
        }
        receipt["config_sha256"] = configuration_sha256(
            {"arguments": vars(args), "variant": variant, "receipt": receipt}
        )
        _write(output_dir / "run_receipt.json", receipt)
        results.append(receipt)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "runs": results,
    }
    _write(root / "route_b_power_ssl_summary.json", payload)
    pair_dataset.close()
    labeled_dataset.close()
    early_dataset.close()
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "tapc_four_class.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "student_b" / "route_b_power_ssl")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--power-levels", type=int, nargs="+", default=[25, 30, 35])
    parser.add_argument("--reference-level", type=int, default=30)
    parser.add_argument("--max-pretrain-identities", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-hidden-dim", type=int, default=128)
    parser.add_argument("--backbone", default="complex_1d_true")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
