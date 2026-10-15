"""Frozen F2 bounded local-contrast evidence study on development-v2."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.thresholding import select_per_label_weak_recall_fpr_thresholds  # noqa: E402
from cjlt.models.evidence_readout import BoundedLocalContrastReadout  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import (  # noqa: E402
    _component_power,
    _load_backbone,
)
from experiments.run_route_a_h5_background_ablation_v2 import (  # noqa: E402
    ALL_NAMES,
    COMMON_INDICES,
    COMMON_NAMES,
    TARGET_LOCAL,
    _metrics,
    _sha256,
    _sigmoid,
    _write,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


F2_MODEL_NAME = "F2_DFMS_Full"
VARIANTS = ("B0_FrozenC6", "L1_LocalBCE", "L2_LocalWeakRank")


def weak_hard_negative_ranking_loss(logits, targets, weak_mask, margin=0.20):
    losses = []
    positive = targets > 0.5
    weak_positive = positive & weak_mask.bool()
    negative = ~positive
    for index in range(logits.shape[1]):
        weak_score = logits[weak_positive[:, index], index]
        negative_score = logits[negative[:, index], index]
        if not weak_score.numel() or not negative_score.numel():
            continue
        count = min(int(weak_score.numel()), int(negative_score.numel()))
        hard_negative = torch.topk(negative_score, k=count).values
        losses.append(
            F.softplus(
                float(margin) - weak_score[:, None] + hard_negative[None, :]
            ).mean()
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _load_frozen_c6(seed, root, device):
    directory = root / f"seed_{seed}" / "C6_PosBCE"
    checkpoint_path = directory / "best_head.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = nn.Linear(128, 6)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, checkpoint_path


def _extract_maps(backbone, base_head, mean, std, data_path, indices,
                  normalization, device, batch_size):
    dataset = V73SignalDataset(data_path, indices, "six", normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    temporal, base_logits, targets, sample_ids = [], [], [], []
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type,
                                        enabled=device.type == "cuda"):
                    maps = backbone.forward_structure_maps(inputs)
                global_feature = maps["global"].float()
                logits = base_head((global_feature - mean_tensor) / std_tensor)
                temporal.append(maps["temporal"].detach().cpu().half().numpy())
                base_logits.append(logits.detach().cpu().float().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {
        "temporal": np.concatenate(temporal),
        "base_logits": np.concatenate(base_logits),
        "targets": np.concatenate(targets),
        "sample_ids": np.concatenate(sample_ids),
    }


def _addon_logits(model, arrays, device, batch_size=128, diagnostics=False):
    model.eval()
    logits, residuals = [], []
    with torch.no_grad():
        for start in range(0, len(arrays["targets"]), batch_size):
            stop = start + batch_size
            temporal = torch.from_numpy(arrays["temporal"][start:stop]).to(device).float()
            base = torch.from_numpy(arrays["base_logits"][start:stop]).to(device)
            if diagnostics:
                output, info = model(temporal, base, return_diagnostics=True)
                residuals.append(info["residual"].cpu().float().numpy())
            else:
                output = model(temporal, base)
            logits.append(output.cpu().float().numpy())
    result = {"logits": np.concatenate(logits)}
    if diagnostics:
        result["residual"] = np.concatenate(residuals)
    return result


def _target_weak_auroc(targets5, probabilities5, weak5):
    values = []
    per_label = {}
    for index in TARGET_LOCAL:
        positive = (targets5[:, index] == 1) & weak5[:, index]
        negative = targets5[:, index] == 0
        if positive.any() and negative.any():
            y = np.concatenate([np.ones(positive.sum()), np.zeros(negative.sum())])
            p = np.concatenate([probabilities5[positive, index], probabilities5[negative, index]])
            score = float(roc_auc_score(y, p))
        else:
            score = float("nan")
        values.append(score)
        per_label[COMMON_NAMES[index]] = score
    return float(np.nanmean(values)), per_label


def _train_variant(variant, train_arrays, fit, validation, weak_train, seed,
                   device, args):
    train_seed = 112000 + int(seed)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    model = BoundedLocalContrastReadout(
        channels=128, num_labels=6, hidden_channels=args.hidden_channels,
        topk=args.topk, residual_cap=args.residual_cap,
    ).to(device)
    fit_targets = train_arrays["targets"][fit].astype(np.float32)
    positives = fit_targets.sum(axis=0)
    pos_weight = torch.as_tensor(
        (len(fit_targets) - positives) / np.maximum(positives, 1.0),
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    dataset = TensorDataset(
        torch.from_numpy(train_arrays["temporal"][fit]),
        torch.from_numpy(train_arrays["base_logits"][fit]),
        torch.from_numpy(fit_targets),
        torch.from_numpy(weak_train[fit].astype(np.uint8)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    best_state, best_epoch = None, 0
    best_key = (-float("inf"), -float("inf"))
    stale, history = 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rows = []
        for temporal, base, target, weak in loader:
            temporal = temporal.to(device).float()
            base, target, weak = base.to(device), target.to(device), weak.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, info = model(temporal, base, return_diagnostics=True)
            residual = info["residual"]
            bce = criterion(logits, target)
            residual_l2 = residual.pow(2).mean()
            mean_penalty = residual.mean(dim=0).pow(2).mean()
            strong = (target > 0.5) & ~weak.bool()
            strong_anchor = residual[strong].pow(2).mean() if strong.any() else residual.sum() * 0.0
            rank = (weak_hard_negative_ranking_loss(logits, target, weak, args.rank_margin)
                    if variant == "L2_LocalWeakRank" else logits.sum() * 0.0)
            loss = (bce + args.residual_l2_weight * residual_l2
                    + args.mean_penalty_weight * mean_penalty)
            if variant == "L2_LocalWeakRank":
                loss = loss + args.rank_weight * rank + args.strong_anchor_weight * strong_anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            epoch_rows.append([float(loss.detach().cpu()), float(bce.detach().cpu()),
                               float(rank.detach().cpu()), float(residual.abs().mean().detach().cpu())])
        validation_output = _addon_logits(model, {
            key: value[validation] for key, value in train_arrays.items()
        }, device)
        probability = _sigmoid(validation_output["logits"])
        targets5 = train_arrays["targets"][validation][:, COMMON_INDICES]
        prob5 = probability[:, COMMON_INDICES]
        weak5 = weak_train[validation][:, COMMON_INDICES]
        macro_aupr = float(np.mean([
            average_precision_score(targets5[:, i], prob5[:, i]) for i in range(5)
        ]))
        weak_auroc, _ = _target_weak_auroc(targets5, prob5, weak5)
        means = np.mean(epoch_rows, axis=0)
        history.append({"epoch": epoch, "train_total_loss": means[0],
                        "train_bce": means[1], "train_rank": means[2],
                        "train_residual_abs_mean": means[3],
                        "validation_macro_aupr_5": macro_aupr,
                        "validation_target_weak_auroc": weak_auroc})
        key = (macro_aupr, weak_auroc)
        improved = key[0] > best_key[0] + 1e-8 or (
            np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8)
        if improved:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("local readout did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, train_seed, history


def _evaluate(variant, logits, residual, arrays, roles, weak6, output_dir, args):
    targets = {role: arrays[role]["targets"] for role in logits}
    calibration = fit_temperature(logits["calibration"], targets["calibration"], "calibration")
    probabilities = {role: _sigmoid(apply_temperature(value, calibration.temperature))
                     for role, value in logits.items()}
    targets5 = {role: value[:, COMMON_INDICES] for role, value in targets.items()}
    weak5 = {role: weak6[roles[role]][:, COMMON_INDICES] for role in logits}
    prob5 = {role: value[:, COMMON_INDICES] for role, value in probabilities.items()}
    selection = select_per_label_weak_recall_fpr_thresholds(
        targets5["threshold"], prob5["threshold"], weak5["threshold"],
        args.max_negative_fpr, "threshold_selection",
        grid=np.arange(0.01, 1.0, 0.01))
    result = _metrics(targets5["early"], prob5["early"], weak5["early"],
                      selection.thresholds,
                      arrays["early"]["targets"][:, 1].astype(bool))
    weak_auroc, per_label_auroc = _target_weak_auroc(
        targets5["early"], prob5["early"], weak5["early"])
    residual_early = residual["early"]
    result.update({
        "variant": variant,
        "target_weak_auroc": weak_auroc,
        "per_label_target_weak_auroc": per_label_auroc,
        "temperature": calibration.temperature,
        "thresholds": selection.thresholds,
        "residual_abs_mean": float(np.abs(residual_early).mean()),
        "residual_abs_max": float(np.abs(residual_early).max()),
        "per_label_residual_mean": {
            name: float(residual_early[:, i].mean()) for i, name in enumerate(ALL_NAMES)
        },
        "development_v2_used": True, "test_v2_used": False,
        "locked_test_used": False,
    })
    _write(output_dir / "calibration.json", calibration.__dict__)
    _write(output_dir / "thresholds.json", {
        "thresholds": selection.thresholds, "max_negative_fpr": args.max_negative_fpr,
        "source_role": "threshold_selection",
    })
    _write(output_dir / "metrics.json", result)
    np.savez_compressed(output_dir / "predictions.npz", logits=logits["early"],
                        probabilities=probabilities["early"],
                        residual=residual_early, targets=targets["early"],
                        sample_ids=roles["early"])
    return result


def _run_seed(args, config, labels6, weak6, seed, device):
    prior_root = args.h5_root / f"seed_{seed}"
    split = json.loads((prior_root / "split_receipt.json").read_text(encoding="utf-8"))
    roles = {name: np.asarray(values, dtype=np.int64) for name, values in split["roles"].items()}
    base_head, base_checkpoint, base_checkpoint_path = _load_frozen_c6(seed, args.h5_root, device)
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    backbone, backbone_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    arrays = {role: _extract_maps(
        backbone, base_head, base_checkpoint["feature_mean"], base_checkpoint["feature_std"],
        _data_path(config), ids, config["normalization"], device, args.feature_batch_size)
        for role, ids in roles.items()}
    del backbone, base_head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rng = np.random.default_rng(95000 + seed)
    order = rng.permutation(len(roles["train"]))
    cut = int(round(0.8 * len(order)))
    fit, validation = order[:cut], order[cut:]
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    _write(seed_root / "run_context.json", {
        "seed": seed, "roles": {key: value.tolist() for key, value in roles.items()},
        "split_sha256": split["split_sha256"], "manifest_sha256": split["manifest_sha256"],
        "backbone_sha256": _sha256(backbone_path),
        "base_head_sha256": _sha256(base_checkpoint_path),
        "development_v2_used": True, "test_v2_used": False, "locked_test_used": False,
    })
    rows = []
    for variant in VARIANTS:
        output_dir = seed_root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        if variant == "B0_FrozenC6":
            logits = {role: data["base_logits"] for role, data in arrays.items()
                      if role in ("early", "calibration", "threshold")}
            residual = {role: np.zeros_like(value, dtype=np.float32) for role, value in logits.items()}
            best_epoch, train_seed = 0, None
        else:
            model, best_epoch, train_seed, history = _train_variant(
                variant, arrays["train"], fit, validation, weak6[roles["train"]],
                seed, device, args)
            outputs = {role: _addon_logits(model, arrays[role], device, diagnostics=True)
                       for role in ("early", "calibration", "threshold")}
            logits = {role: value["logits"] for role, value in outputs.items()}
            residual = {role: value["residual"] for role, value in outputs.items()}
            torch.save({"model_state_dict": model.state_dict(), "variant": variant,
                        "best_epoch": best_epoch, "train_seed": train_seed,
                        "residual_cap": args.residual_cap, "topk": args.topk},
                       output_dir / "best_readout.pth")
            with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader(); writer.writerows(history)
        result = _evaluate(variant, logits, residual, arrays, roles, weak6, output_dir, args)
        result.update({"seed": seed, "best_epoch": best_epoch, "train_seed": train_seed,
                       "split_sha256": split["split_sha256"],
                       "manifest_sha256": split["manifest_sha256"]})
        _write(output_dir / "variant_summary.json", result)
        rows.append(result)
    del arrays
    return rows


def _aggregate(rows):
    metrics = ["macro_aupr", "macro_f1", "target_weak_recall", "target_weak_auroc",
               "target_negative_fpr", "saw_present_macro_f1", "saw_absent_macro_f1",
               "residual_abs_mean", "residual_abs_max"]
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {metric: _mean_ci([row[metric] for row in selected],
                                                120000 + vi * 100 + mi)
                              for mi, metric in enumerate(metrics)}
    comparisons = {}
    for candidate, baseline in (("L1_LocalBCE", "B0_FrozenC6"),
                                ("L2_LocalWeakRank", "B0_FrozenC6"),
                                ("L2_LocalWeakRank", "L1_LocalBCE")):
        paired = []
        for seed in sorted({row["seed"] for row in rows}):
            c = next(row for row in rows if row["seed"] == seed and row["variant"] == candidate)
            b = next(row for row in rows if row["seed"] == seed and row["variant"] == baseline)
            paired.append({"seed": seed, **{metric: c[metric] - b[metric] for metric in metrics}})
        comparisons[f"{candidate}_minus_{baseline}"] = {
            "paired": paired,
            "mean": {metric: float(np.mean([row[metric] for row in paired])) for metric in metrics},
        }
    delta = comparisons["L2_LocalWeakRank_minus_B0_FrozenC6"]
    mean = delta["mean"]
    candidate = aggregate["L2_LocalWeakRank"]
    gates = {
        "macro_aupr_noninferior": mean["macro_aupr"] >= -0.005,
        "macro_f1_gain": mean["macro_f1"] >= 0.010,
        "target_weak_recall_gain": mean["target_weak_recall"] >= 0.030,
        "target_weak_auroc_gain": mean["target_weak_auroc"] >= 0.015,
        "target_fpr_cap": candidate["target_negative_fpr"]["mean"] <= 0.20,
        "four_of_five_positive_weak_gain": sum(
            row["target_weak_recall"] > 0 for row in delta["paired"]) >= 4,
        "saw_present_noninferior": mean["saw_present_macro_f1"] >= -0.005,
        "residual_mean_bound": candidate["residual_abs_mean"]["mean"] <= 0.10,
        "residual_hard_bound": candidate["residual_abs_max"]["mean"] <= 0.350001,
    }
    return aggregate, comparisons, gates


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    labels6 = read_labels(data_path)
    values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        print(f"bounded-local development-v2 seed={seed}", flush=True)
        rows.extend(_run_seed(args, config, labels6, weak6, seed, device))
    aggregate, comparisons, gates = _aggregate(rows)
    payload = {
        "status": "completed", "experiment": "RouteA_BoundedLocalContrast_v2",
        "seeds": args.seeds, "data_path": str(data_path), "data_sha256": _sha256(data_path),
        "aggregate": aggregate, "comparisons": comparisons,
        "promotion_gates": gates, "promoted": bool(all(gates.values())),
        "next_action": ("unfreeze_temporal_backbone_at_0p1_lr" if all(gates.values())
                        else "move_local_expert_to_stage2_stage3"),
        "development_v2_used": True, "test_v2_used": False, "locked_test_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_bounded_local_contrast_v2_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--h5-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_h5_background_ablation_v2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_bounded_local_contrast_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--residual-cap", type=float, default=0.35)
    parser.add_argument("--rank-weight", type=float, default=0.50)
    parser.add_argument("--rank-margin", type=float, default=0.20)
    parser.add_argument("--residual-l2-weight", type=float, default=0.05)
    parser.add_argument("--mean-penalty-weight", type=float, default=0.20)
    parser.add_argument("--strong-anchor-weight", type=float, default=0.10)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

