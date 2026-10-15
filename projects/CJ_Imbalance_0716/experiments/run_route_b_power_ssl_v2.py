"""Run identity-aware power-SSL V2 screening on three new seeds."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.augmentations import weak_complex_augmentation  # noqa: E402
from cjlt.data.power_pair_dataset import PowerPairDataset  # noqa: E402
from cjlt.data.power_pairs import build_role_bound_power_pair_index  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.run_manager import configuration_sha256  # noqa: E402
from cjlt.models.power_ssl import (  # noqa: E402
    PowerContrastiveModel,
    contrastive_similarity_diagnostics,
    multi_positive_contrastive_loss,
    symmetric_contrastive_loss,
)
from experiments.run_route_b_power_ssl import (  # noqa: E402
    _build_backbone,
    _finetune,
    _limited_records,
    _probe,
    _write,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
from experiments.run_student_b import build_student_b_plan  # noqa: E402


VARIANTS = (
    "RandomInit",
    "SameViewAug",
    "NaivePower",
    "MultiPositive",
    "HybridMasked",
    "HybridPowerAux",
    "RandomSource",
)
PRETRAINED_VARIANTS = VARIANTS[1:]


class IdentityBatchSampler(Sampler):
    def __init__(self, records, batch_size, seed):
        groups = {}
        for position, record in enumerate(records):
            groups.setdefault(int(record.identity), []).append(int(position))
        self.groups = {key: tuple(value) for key, value in groups.items()}
        self.identities = tuple(sorted(self.groups))
        maximum = max(len(value) for value in self.groups.values())
        self.identities_per_batch = max(1, int(batch_size) // maximum)
        self.seed = int(seed)
        self.iteration = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.iteration)
        self.iteration += 1
        identities = np.asarray(self.identities, dtype=np.int64)
        rng.shuffle(identities)
        usable = len(identities) - len(identities) % self.identities_per_batch
        for start in range(0, usable, self.identities_per_batch):
            batch = []
            for identity in identities[start : start + self.identities_per_batch]:
                batch.extend(self.groups[int(identity)])
            yield batch

    def __len__(self):
        return len(self.identities) // self.identities_per_batch


def _identity_loader(dataset, args, seed):
    return DataLoader(
        dataset,
        batch_sampler=IdentityBatchSampler(dataset.records, args.batch_size, seed),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )


def _derangement_shift(identities):
    for shift in range(1, len(identities)):
        if bool(torch.all(identities != torch.roll(identities, shifts=shift, dims=0))):
            return shift
    raise RuntimeError("batch cannot form a different-identity derangement")


def _pretrain(args, variant, dataset, num_classes, device, output_dir):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loader = _identity_loader(dataset, args, int(args.seed) + 505)
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
        totals = {"total": 0.0, "same": 0.0, "power": 0.0, "aux": 0.0}
        count = 0
        diagnostic = None
        for batch in loader:
            reference = batch["reference_signal"]
            paired = batch["paired_signal"]
            identities = batch["identity"].long().to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                same_loss = reference.new_tensor(0.0, device=device)
                power_loss = reference.new_tensor(0.0, device=device)
                aux_loss = reference.new_tensor(0.0, device=device)

                if variant in ("SameViewAug", "HybridMasked", "HybridPowerAux"):
                    first_aug = weak_complex_augmentation(
                        reference, [], generator=generator
                    ).signal
                    second_aug = weak_complex_augmentation(
                        reference, [], generator=generator
                    ).signal
                    _, first_aug_projection = model(first_aug.to(device))
                    _, second_aug_projection = model(second_aug.to(device))
                    same_loss = multi_positive_contrastive_loss(
                        first_aug_projection,
                        second_aug_projection,
                        identities,
                        temperature=args.temperature,
                    )

                first_features = second_features = None
                first_projection = second_projection = None
                if variant != "SameViewAug":
                    first_features, first_projection = model(reference.to(device))
                    second_features, second_projection = model(paired.to(device))

                if variant == "NaivePower":
                    power_loss = symmetric_contrastive_loss(
                        first_projection, second_projection, temperature=args.temperature
                    )
                elif variant in ("MultiPositive", "HybridMasked", "HybridPowerAux"):
                    power_loss = multi_positive_contrastive_loss(
                        first_projection,
                        second_projection,
                        identities,
                        temperature=args.temperature,
                    )
                elif variant == "RandomSource":
                    shift = _derangement_shift(identities)
                    power_loss = symmetric_contrastive_loss(
                        first_projection,
                        torch.roll(second_projection, shifts=shift, dims=0),
                        temperature=args.temperature,
                    )

                if variant == "HybridPowerAux":
                    signed_gap = (
                        batch["paired_level"].float()
                        - batch["reference_level"].float()
                    ).to(device) / float(args.power_gap_scale)
                    prediction = model.predict_power_gap(first_features, second_features)
                    aux_loss = F.mse_loss(prediction, signed_gap)

                if variant == "SameViewAug":
                    loss = same_loss
                elif variant in ("NaivePower", "MultiPositive", "RandomSource"):
                    loss = power_loss
                else:
                    loss = same_loss + args.power_weight * power_loss + args.power_aux_weight * aux_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count = len(reference)
            count += batch_count
            for name, value in (
                ("total", loss), ("same", same_loss),
                ("power", power_loss), ("aux", aux_loss),
            ):
                totals[name] += float(value.detach().cpu().item()) * batch_count
            if diagnostic is None:
                if first_projection is None:
                    diagnostic = contrastive_similarity_diagnostics(
                        first_aug_projection.detach(), second_aug_projection.detach()
                    )
                else:
                    diagnostic = contrastive_similarity_diagnostics(
                        first_projection.detach(), second_projection.detach()
                    )
        history.append(
            {
                "epoch": epoch,
                "pretrain_loss": totals["total"] / max(count, 1),
                "same_view_loss": totals["same"] / max(count, 1),
                "power_loss": totals["power"] / max(count, 1),
                "power_aux_loss": totals["aux"] / max(count, 1),
                "samples": count,
                **(diagnostic or {}),
            }
        )
    checkpoint = {
        "backbone_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.backbone.state_dict().items()
        },
        "variant": variant,
        "history": history,
        "train_seconds": time.perf_counter() - started,
    }
    torch.save(checkpoint, str(output_dir / "pretrained_backbone.pth"))
    _write(output_dir / "pretrain_history.json", history)
    return checkpoint


def _metrics_from_logits(logits, target):
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    prediction = (probability >= 0.5).astype(np.uint8)
    metrics = compute_multilabel_metrics(
        target.astype(np.uint8), prediction, probabilities=probability
    )
    return {
        "macro_aupr": metrics["macro_aupr"],
        "macro_f1": metrics["macro_f1"],
        "hamming_loss": metrics["hamming_loss"],
    }


@torch.no_grad()
def _evaluate_heldout(backbone, loader, device, head=None):
    backbone.to(device).eval()
    if head is not None:
        head.to(device).eval()
    logits_parts = []
    target_parts = []
    for batch in loader:
        signal = batch["paired_signal"].to(device)
        if head is None:
            logits = backbone(signal)
        else:
            logits = head(backbone.forward_features(signal))
        logits_parts.append(logits.cpu().numpy())
        target_parts.append(batch["label"].numpy())
    return _metrics_from_logits(
        np.concatenate(logits_parts, axis=0),
        np.concatenate(target_parts, axis=0),
    )


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _aggregate(rows):
    result = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        entry = {"variant": variant, "seed_count": len(selected)}
        for metric in (
            "linear_probe_macro_aupr",
            "finetune_macro_aupr",
            "heldout_probe_macro_aupr",
            "heldout_finetune_macro_aupr",
        ):
            entry[metric] = _mean_std([row[metric] for row in selected])
        result.append(entry)
    return result


def _gates(rows):
    baseline = {row["seed"]: row for row in rows if row["variant"] == "SameViewAug"}
    result = []
    for variant in ("MultiPositive", "HybridMasked", "HybridPowerAux"):
        selected = {row["seed"]: row for row in rows if row["variant"] == variant}
        differences = {
            "linear_probe": [
                selected[seed]["linear_probe_macro_aupr"]
                - baseline[seed]["linear_probe_macro_aupr"]
                for seed in sorted(baseline)
            ],
            "finetune": [
                selected[seed]["finetune_macro_aupr"]
                - baseline[seed]["finetune_macro_aupr"]
                for seed in sorted(baseline)
            ],
            "heldout_finetune": [
                selected[seed]["heldout_finetune_macro_aupr"]
                - baseline[seed]["heldout_finetune_macro_aupr"]
                for seed in sorted(baseline)
            ],
        }
        entry = {
            "variant": variant,
            "differences_vs_same_view": differences,
            "mean_linear_probe_gain": float(np.mean(differences["linear_probe"])),
            "mean_finetune_gain": float(np.mean(differences["finetune"])),
            "mean_heldout_finetune_gain": float(np.mean(differences["heldout_finetune"])),
        }
        entry["all_seed_directions_positive"] = all(
            all(value > 0 for value in values) for values in differences.values()
        )
        entry["passed"] = bool(
            entry["mean_linear_probe_gain"] >= 0.01
            and entry["mean_finetune_gain"] >= 0.005
            and entry["mean_heldout_finetune_gain"] >= 0.01
            and entry["all_seed_directions_positive"]
        )
        result.append(entry)
    return result


def run(args):
    config = load_config(args.config)
    root = Path(args.output_root)
    rows = []
    for seed in args.seeds:
        args.seed = int(seed)
        base_plan = build_run_plan(
            config, loss_name="asl", seed=seed,
            max_train_samples=0, max_eval_samples=args.max_eval_samples,
        )
        semi_plan = build_student_b_plan(
            config, labeled_ratio=args.labeled_ratio, seed=seed,
            max_labeled_samples=0, max_unlabeled_samples=0,
            max_eval_samples=args.max_eval_samples,
        )
        power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
        pair_index = build_role_bound_power_pair_index(
            _data_path(config), power_root,
            base_indices=semi_plan["unlabeled_indices"],
            levels=args.power_levels, reference_level=args.reference_level,
        )
        records = _limited_records(pair_index.records, args.max_pretrain_identities)
        pair_dataset = PowerPairDataset(
            power_root, records, class_mode=base_plan["class_mode"],
            normalization=config["normalization"], include_parameters=False,
        )
        heldout_levels = sorted(set([args.reference_level] + list(args.heldout_power_levels)))
        heldout_index = build_role_bound_power_pair_index(
            _data_path(config), power_root,
            base_indices=base_plan["early_stop_indices"],
            levels=heldout_levels, reference_level=args.reference_level,
        )
        heldout_dataset = PowerPairDataset(
            power_root, heldout_index.records, class_mode=base_plan["class_mode"],
            normalization=config["normalization"], include_parameters=False,
        )
        heldout_loader = DataLoader(
            heldout_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
        )
        common = {
            "path": _data_path(config),
            "class_mode": base_plan["class_mode"],
            "normalization": config["normalization"],
            "include_parameters": False,
        }
        labeled_dataset = V73SignalDataset(indices=semi_plan["labeled_indices"], **common)
        early_dataset = V73SignalDataset(indices=base_plan["early_stop_indices"], **common)
        labeled_loader = DataLoader(
            labeled_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
        )
        early_loader = DataLoader(
            early_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
        )
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        for variant in VARIANTS:
            output_dir = root / variant / "seed_{}".format(seed)
            output_dir.mkdir(parents=True, exist_ok=True)
            if variant == "RandomInit":
                torch.manual_seed(seed)
                backbone = _build_backbone(args, base_plan["num_classes"])
                pretrain = {"train_seconds": 0.0, "history": []}
            else:
                pretrain = _pretrain(
                    args, variant, pair_dataset,
                    base_plan["num_classes"], device, output_dir,
                )
                backbone = _build_backbone(args, base_plan["num_classes"])
                backbone.load_state_dict(pretrain["backbone_state_dict"])

            probe = _probe(args, backbone, labeled_loader, early_loader, device, output_dir)
            probe_checkpoint = torch.load(
                str(output_dir / "linear_probe.pth"), map_location="cpu"
            )
            probe_head = nn.Linear(backbone.feature_dim, base_plan["num_classes"])
            probe_head.load_state_dict(probe_checkpoint["head_state_dict"])
            heldout_probe = _evaluate_heldout(
                backbone, heldout_loader, device, head=probe_head
            )

            finetune_backbone = _build_backbone(args, base_plan["num_classes"])
            finetune_backbone.load_state_dict(backbone.state_dict())
            finetune = _finetune(
                args, finetune_backbone, labeled_loader, early_loader, device, output_dir
            )
            heldout_finetune = _evaluate_heldout(
                finetune_backbone, heldout_loader, device, head=None
            )
            receipt = {
                "variant": variant,
                "seed": seed,
                "labeled_ratio": args.labeled_ratio,
                "labeled_samples": len(labeled_dataset),
                "unlabeled_pretrain_identities": len(set(record.identity for record in records)),
                "pretrain_records": len(records),
                "linear_probe_macro_aupr": probe["macro_aupr"],
                "finetune_macro_aupr": finetune["macro_aupr"],
                "heldout_probe_macro_aupr": heldout_probe["macro_aupr"],
                "heldout_finetune_macro_aupr": heldout_finetune["macro_aupr"],
                "heldout_probe_metrics": heldout_probe,
                "heldout_finetune_metrics": heldout_finetune,
                "pretrain_seconds": pretrain["train_seconds"],
                "split_sha256": base_plan["split_sha256"],
                "semi_split_sha256": semi_plan["semi_split_sha256"],
                "pair_sha256": pair_index.sha256,
                "heldout_pair_sha256": heldout_index.sha256,
                "locked_test_used": False,
            }
            receipt["config_sha256"] = configuration_sha256(
                {"arguments": vars(args), "variant": variant, "receipt": receipt}
            )
            _write(output_dir / "run_receipt.json", receipt)
            rows.append(receipt)

        pair_dataset.close()
        heldout_dataset.close()
        labeled_dataset.close()
        early_dataset.close()

    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": vars(args),
        "aggregate": _aggregate(rows),
        "gates": _gates(rows),
        "runs": rows,
    }
    _write(root / "route_b_power_ssl_v2_summary.json", payload)
    print(json.dumps(payload["gates"], ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "tapc_four_class.yaml",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs" / "student_b" / "route_b_power_ssl_v2",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2031, 2032, 2033])
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--labeled-ratio", type=float, default=0.05)
    parser.add_argument("--power-levels", type=int, nargs="+", default=[25, 30, 35])
    parser.add_argument("--heldout-power-levels", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--reference-level", type=int, default=30)
    parser.add_argument("--max-pretrain-identities", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--power-weight", type=float, default=0.2)
    parser.add_argument("--power-aux-weight", type=float, default=0.1)
    parser.add_argument("--power-gap-scale", type=float, default=20.0)
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
