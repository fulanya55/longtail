"""Identity-safe weak-component residual experiment on a frozen MLCNN router."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader

from cj0724.dual_axis import (
    POWER_LEVELS,
    PairedPowerViewDataset,
    PowerViewDataset,
    audit_power_alignment,
    discover_power_train_files,
    read_identity_metadata,
)
from cj0724.losses import PairedSideRoutedResidualLoss, PairedWeakVisibilityResidualLoss
from cj0724.models import SideRoutedPhysicsResidualHead, WeakVisibilityResidualHead
from cj0724.multitask import build_multitask_model
from cj0724.power_robustness_calibration import fbeta_thresholds


PROFILES = {
    "guarded": {
        "weak_weight": 1.0,
        "ranking_weight": 0.25,
        "hard_negative_weight": 1.5,
        "residual_bound": 1.5,
    },
    "balanced": {
        "weak_weight": 1.5,
        "ranking_weight": 0.5,
        "hard_negative_weight": 1.0,
        "residual_bound": 2.0,
    },
    "recall": {
        "weak_weight": 2.0,
        "ranking_weight": 0.75,
        "hard_negative_weight": 0.75,
        "residual_bound": 2.5,
    },
    "side_r01": {
        "weak_weight": 1.5, "ranking_weight": 0.5, "hard_negative_weight": 1.0,
        "residual_bound": 2.5, "route_weight": 0.1,
    },
    "side_r03": {
        "weak_weight": 1.5, "ranking_weight": 0.5, "hard_negative_weight": 1.0,
        "residual_bound": 2.5, "route_weight": 0.3,
    },
    "side_r06": {
        "weak_weight": 1.5, "ranking_weight": 0.5, "hard_negative_weight": 1.0,
        "residual_bound": 2.5, "route_weight": 0.6,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def classifier_logits(model: nn.Module, iq: torch.Tensor) -> torch.Tensor:
    output = model(iq)
    return output[0] if isinstance(output, tuple) else output


def train_epoch(
    base_model: nn.Module,
    residual_head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, float]:
    base_model.eval()
    residual_head.train()
    totals = {key: 0.0 for key in ("loss", "base", "weak", "ranking", "hard_negative", "residual", "route")}
    samples = 0
    for batch in loader:
        weak_iq = batch["iq_weak"].to(device, non_blocking=True)
        reference_iq = batch["iq_reference"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        weak_mask = batch["weak_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            base_weak = classifier_logits(base_model, weak_iq)
            base_reference = classifier_logits(base_model, reference_iq)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            components = residual_head.forward_components(weak_iq, base_weak)
            if isinstance(criterion, PairedSideRoutedResidualLoss):
                loss, parts = criterion(
                    components, labels, weak_mask, base_reference, batch["level"]
                )
            else:
                loss, parts = criterion(components, labels, weak_mask, base_reference)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(residual_head.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in parts.items():
            totals[name] += float(value) * batch_size
        samples += batch_size
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.no_grad()
def collect_level(
    base_model: nn.Module,
    residual_head: nn.Module,
    files: dict[int, Path],
    labels: np.ndarray,
    indices: np.ndarray,
    level: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = PowerViewDataset(files, labels, indices, (int(level),), seed)
    loader = make_loader(dataset, batch_size, False, seed)
    base_model.eval()
    residual_head.eval()
    outputs = {key: [] for key in ("labels", "weak_mask", "base_logits", "logits", "residual")}
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            base_logits = classifier_logits(base_model, iq)
            components = residual_head.forward_components(iq, base_logits)
        outputs["labels"].append(batch["label"].numpy())
        outputs["weak_mask"].append(batch["weak_mask"].numpy())
        outputs["base_logits"].append(base_logits.float().cpu().numpy())
        outputs["logits"].append(components["logits"].float().cpu().numpy())
        outputs["residual"].append(components["residual"].float().cpu().numpy())
    dataset.close()
    return {key: np.concatenate(value) for key, value in outputs.items()}


def collect_role(
    base_model: nn.Module,
    residual_head: nn.Module,
    files: dict[int, Path],
    labels: np.ndarray,
    indices: np.ndarray,
    levels: tuple[int, ...],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[int, dict[str, np.ndarray]]:
    return {
        level: collect_level(
            base_model, residual_head, files, labels, indices, level, batch_size, seed, device
        )
        for level in levels
    }


def probabilities(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def evaluate(
    records: dict[int, dict[str, np.ndarray]], thresholds: np.ndarray, logit_key: str
) -> dict[str, object]:
    level_rows = []
    all_labels, all_probability = [], []
    for level in sorted(records):
        record = records[level]
        truth = record["labels"].astype(bool)
        probability = probabilities(record[logit_key])
        predicted = probability >= np.asarray(thresholds)[None, :]
        weak_positive = record["weak_mask"].astype(bool) & truth
        negative = ~truth
        per_aupr = [
            float(sk_metrics.average_precision_score(truth[:, index], probability[:, index]))
            for index in range(truth.shape[1])
        ]
        level_rows.append(
            {
                "power_level_db": int(level),
                "weak_component_count": int(weak_positive.sum()),
                "weak_component_recall": float(predicted[weak_positive].mean())
                if weak_positive.any()
                else float("nan"),
                "false_positive_gate_rate": float(np.logical_and(predicted, negative).sum() / max(negative.sum(), 1)),
                "macro_aupr": float(np.mean(per_aupr)),
                "macro_f1": float(sk_metrics.f1_score(truth, predicted, average="macro", zero_division=0)),
            }
        )
        all_labels.append(truth.astype(np.uint8))
        all_probability.append(probability)
    combined_truth = np.concatenate(all_labels)
    combined_probability = np.concatenate(all_probability)
    combined_prediction = combined_probability >= np.asarray(thresholds)[None, :]
    combined_negative = ~combined_truth.astype(bool)
    by_level = {row["power_level_db"]: row for row in level_rows}
    return {
        "weak_recall_10db": float(by_level[10]["weak_component_recall"]) if 10 in by_level else float("nan"),
        "weak_recall_50db": float(by_level[50]["weak_component_recall"]) if 50 in by_level else float("nan"),
        "nine_level_mean_weak_recall": float(
            np.nanmean([row["weak_component_recall"] for row in level_rows])
        ),
        "nine_level_mean_false_positive_gate_rate": float(
            np.mean([row["false_positive_gate_rate"] for row in level_rows])
        ),
        "combined_false_positive_gate_rate": float(
            np.logical_and(combined_prediction, combined_negative).sum() / max(combined_negative.sum(), 1)
        ),
        "combined_macro_aupr": float(
            np.mean(
                [
                    sk_metrics.average_precision_score(combined_truth[:, index], combined_probability[:, index])
                    for index in range(combined_truth.shape[1])
                ]
            )
        ),
        "combined_macro_f1": float(
            sk_metrics.f1_score(combined_truth, combined_prediction, average="macro", zero_division=0)
        ),
        "per_level": level_rows,
    }


def combined_arrays(records: dict[int, dict[str, np.ndarray]], logit_key: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([records[level]["labels"] for level in sorted(records)])
    probability = np.concatenate([probabilities(records[level][logit_key]) for level in sorted(records)])
    return labels, probability


def train_one(
    seed: int,
    profile_name: str,
    args: argparse.Namespace,
    protocol: dict,
    selected_row: dict,
    roles: dict[str, np.ndarray],
    files: dict[int, Path],
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed)
    profile = PROFILES[profile_name]
    base_model = build_multitask_model(
        selected_row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
        protocol["parameter_hidden_dim"], protocol["legacy_attention"],
    ).to(device)
    checkpoint = torch.load(selected_row["checkpoint"], map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint["model_state"])
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    if args.head == "side_physics":
        residual_head = SideRoutedPhysicsResidualHead(
            width=args.width,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            residual_bound=profile["residual_bound"],
        ).to(device)
        criterion = PairedSideRoutedResidualLoss(
            weak_weight=profile["weak_weight"],
            ranking_weight=profile["ranking_weight"],
            hard_negative_weight=profile["hard_negative_weight"],
            residual_weight=args.residual_weight,
            route_weight=profile.get("route_weight", args.route_weight),
        )
    else:
        residual_head = WeakVisibilityResidualHead(
            width=args.width,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            residual_bound=profile["residual_bound"],
        ).to(device)
        criterion = PairedWeakVisibilityResidualLoss(
            weak_weight=profile["weak_weight"],
            ranking_weight=profile["ranking_weight"],
            hard_negative_weight=profile["hard_negative_weight"],
            residual_weight=args.residual_weight,
        )
    optimizer = torch.optim.AdamW(
        residual_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    train_dataset = PairedPowerViewDataset(files, labels, roles["train"], tuple(args.train_levels), seed)
    train_loader = make_loader(train_dataset, args.batch_size, True, seed)
    run_dir = Path(args.output_root) / profile_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = train_epoch(
            base_model, residual_head, train_loader, criterion, optimizer, scaler, device
        )
        early_records = collect_role(
            base_model, residual_head, files, labels, roles["early"], tuple(args.early_levels),
            args.batch_size, seed, device,
        )
        early_labels, early_probability = combined_arrays(early_records, "logits")
        weak_aupr = []
        for level, record in early_records.items():
            weak_classes = range(3, 6) if level < 30 else range(0, 3)
            for class_index in weak_classes:
                weak_aupr.append(
                    sk_metrics.average_precision_score(
                        record["labels"][:, class_index], probabilities(record["logits"])[:, class_index]
                    )
                )
        macro_aupr = float(np.mean([
            sk_metrics.average_precision_score(early_labels[:, i], early_probability[:, i])
            for i in range(6)
        ]))
        mean_weak_aupr = float(np.mean(weak_aupr))
        mean_abs_residual = float(np.mean([
            np.abs(record["residual"]).mean() for record in early_records.values()
        ]))
        row = {
            "epoch": epoch,
            **train_metrics,
            "early_macro_aupr": macro_aupr,
            "early_weak_aupr": mean_weak_aupr,
            "early_mean_abs_residual": mean_abs_residual,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps({"profile": profile_name, "seed": seed, **row}))
        key = (mean_weak_aupr, macro_aupr, -train_metrics["loss"])
        if key > best_key:
            best_key = key
            torch.save(
                {
                    "model_state": residual_head.state_dict(),
                    "epoch": epoch,
                    "profile": profile_name,
                    "profile_config": profile,
                    "head": args.head,
                    "base_checkpoint": selected_row["checkpoint"],
                },
                run_dir / "best.pt",
            )
        scheduler.step()
    train_dataset.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    residual_head.load_state_dict(best["model_state"])

    role_records = {
        role: collect_role(
            base_model, residual_head, files, labels, roles[role], POWER_LEVELS,
            args.batch_size, seed, device,
        )
        for role in ("calibration", "tune", "confirm")
    }
    base_thresholds = np.asarray(json.loads(selected_row["thresholds"]), dtype=np.float32)
    base_tune = evaluate(role_records["tune"], base_thresholds, "base_logits")
    fpr_cap = float(base_tune["nine_level_mean_false_positive_gate_rate"] + args.fpr_slack)
    calibration_labels, calibration_probability = combined_arrays(role_records["calibration"], "logits")
    candidates = []
    for beta in args.betas:
        thresholds = fbeta_thresholds(calibration_labels, calibration_probability, beta)
        metrics = evaluate(role_records["tune"], thresholds, "logits")
        candidate = {
            "beta": float(beta),
            "thresholds": thresholds.tolist(),
            "tune": metrics,
            "within_fpr_cap": bool(metrics["nine_level_mean_false_positive_gate_rate"] <= fpr_cap),
        }
        candidates.append(candidate)
    feasible = [candidate for candidate in candidates if candidate["within_fpr_cap"]]
    pool = feasible if feasible else candidates
    selected = max(
        pool,
        key=lambda candidate: (
            min(candidate["tune"]["weak_recall_10db"], candidate["tune"]["weak_recall_50db"]),
            candidate["tune"]["nine_level_mean_weak_recall"],
            -candidate["tune"]["nine_level_mean_false_positive_gate_rate"],
        ),
    )
    selected_thresholds = np.asarray(selected["thresholds"], dtype=np.float32)
    result = {
        "profile": profile_name,
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "train_levels": list(args.train_levels),
        "base_checkpoint": selected_row["checkpoint"],
        "base_thresholds": base_thresholds.tolist(),
        "fpr_slack": float(args.fpr_slack),
        "tune_fpr_cap": fpr_cap,
        "threshold_candidates": candidates,
        "selected_beta": selected["beta"],
        "selected_thresholds": selected["thresholds"],
        "base_confirm": evaluate(role_records["confirm"], base_thresholds, "base_logits"),
        "residual_confirm": evaluate(role_records["confirm"], selected_thresholds, "logits"),
        "residual_diagnostics": {
            "confirm_mean_abs_residual": float(np.mean([
                np.abs(record["residual"]).mean() for record in role_records["confirm"].values()
            ])),
            "confirm_max_abs_residual": float(np.max([
                np.abs(record["residual"]).max() for record in role_records["confirm"].values()
            ])),
        },
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del base_model, residual_head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/weak_visibility_residual_20260725")
    parser.add_argument("--head", choices=("shared_iq", "side_physics"), default="shared_iq")
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=["balanced"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--route-weight", type=float, default=0.3)
    parser.add_argument("--fpr-slack", type=float, default=0.05)
    parser.add_argument("--train-levels", nargs="+", type=int, default=[10, 15, 45, 50])
    parser.add_argument("--early-levels", nargs="+", type=int, default=[10, 15, 45, 50])
    parser.add_argument("--betas", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0])
    args = parser.parse_args()

    run_root = Path(args.run_root)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    roles_npz = np.load(run_root / "role_indices.npz")
    roles = {name: roles_npz[name].astype(np.int64) for name in roles_npz.files}
    files = discover_power_train_files(args.power_root)
    alignment = audit_power_alignment(files)
    root_seeds, root_labels = read_identity_metadata(protocol["data_file"])
    power_seeds, power_labels = read_identity_metadata(files[30])
    if not np.array_equal(root_seeds, power_seeds) or not np.array_equal(root_labels, power_labels):
        raise RuntimeError("root and power identity order differs")
    selected_rows = {
        int(row["seed"]): row
        for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed"
        and row.get("checkpoint_objective") == "parameter"
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_protocol = {
        "base_run_root": str(run_root),
        "power_root": str(args.power_root),
        "profiles": args.profiles,
        "head": args.head,
        "profile_configs": {name: PROFILES[name] for name in args.profiles},
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_levels": args.train_levels,
        "early_levels": args.early_levels,
        "threshold_fit_role": "calibration",
        "profile_and_threshold_selection_role": "tune",
        "final_development_role": "confirm",
        "role_hashes": {name: array_sha256(values) for name, values in roles.items()},
        "identity_alignment": alignment,
        "inference_inputs": "current I/Q view and frozen current-view logits only",
        "reference_view_usage": "training-only ranking teacher at 30 dB",
        "locked_test_used": False,
    }
    (output_root / "protocol.json").write_text(json.dumps(experiment_protocol, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for profile_name in args.profiles:
        for seed in args.seeds:
            results.append(
                train_one(
                    seed, profile_name, args, protocol, selected_rows[seed], roles, files,
                    root_labels.astype(np.uint8), device,
                )
            )
    (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    aggregates = []
    for profile_name in args.profiles:
        rows = [row for row in results if row["profile"] == profile_name]
        entry = {"profile": profile_name, "seed_count": len(rows)}
        for method in ("base_confirm", "residual_confirm"):
            for metric in (
                "weak_recall_10db", "weak_recall_50db", "nine_level_mean_weak_recall",
                "nine_level_mean_false_positive_gate_rate", "combined_macro_aupr", "combined_macro_f1",
            ):
                values = [row[method][metric] for row in rows]
                entry[f"{method}_{metric}_mean"] = float(np.mean(values))
                entry[f"{method}_{metric}_std"] = float(np.std(values))
        entry["three_seed_10db_same_direction"] = bool(
            all(row["residual_confirm"]["weak_recall_10db"] > row["base_confirm"]["weak_recall_10db"] for row in rows)
        )
        entry["three_seed_50db_same_direction"] = bool(
            all(row["residual_confirm"]["weak_recall_50db"] > row["base_confirm"]["weak_recall_50db"] for row in rows)
        )
        entry["fpr_guard_pass"] = bool(
            entry["residual_confirm_nine_level_mean_false_positive_gate_rate_mean"]
            <= entry["base_confirm_nine_level_mean_false_positive_gate_rate_mean"] + args.fpr_slack
        )
        entry["promotion_pass"] = bool(
            entry["residual_confirm_weak_recall_10db_mean"] >= 0.8
            and entry["residual_confirm_weak_recall_50db_mean"] >= 0.8
            and entry["three_seed_10db_same_direction"]
            and entry["three_seed_50db_same_direction"]
            and entry["fpr_guard_pass"]
        )
        aggregates.append(entry)
    (output_root / "aggregate.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
