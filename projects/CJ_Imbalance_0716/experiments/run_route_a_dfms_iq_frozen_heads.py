"""Frozen F2 head, weak-ranking, constrained-threshold, and SawFM expert study."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch  # Load PyTorch DLLs before sklearn/OpenMP on Windows.
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import (  # noqa: E402
    V73SignalDataset,
    read_labels,
    read_parameters,
)
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.thresholding import (  # noqa: E402
    apply_thresholds,
    select_per_label_f1_thresholds,
    select_per_label_weak_recall_fpr_thresholds,
)
from cjlt.losses.classification import DistributionBalancedLoss  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_structure_screen import CLASS_NAMES  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


SAW_INDEX = 1
F2_MODEL_NAME = "F2_DFMS_Full"
VARIANTS = {
    "J0_JointDB": {
        "train_head": False,
        "loss": "joint_db",
        "saw_expert": False,
    },
    "H1_FrozenDB": {
        "train_head": True,
        "loss": "db",
        "saw_expert": False,
    },
    "H2_PosBCE": {
        "train_head": True,
        "loss": "pos_bce",
        "saw_expert": False,
    },
    "H3_WeakRank": {
        "train_head": True,
        "loss": "weak_rank",
        "saw_expert": False,
    },
    "H4_WeakRankSawExpert": {
        "train_head": True,
        "loss": "weak_rank",
        "saw_expert": True,
    },
}
HEAD_METRICS = (
    "macro_aupr",
    "macro_f1",
    "hamming_loss",
    "weak_component_recall",
    "weak_vs_negative_auroc",
    "weak_vs_negative_aupr",
    "negative_fpr",
    "saw_aupr",
    "saw_weak_recall",
    "saw_weak_vs_negative_auroc",
    "saw_negative_fpr",
)


class FrozenF2Head(nn.Module):
    """Linear F2 head with an optional fixed-low-weight SawFM expert."""

    def __init__(self, feature_dim, num_labels, saw_expert, expert_weight):
        super().__init__()
        self.main = nn.Linear(int(feature_dim), int(num_labels))
        self.saw_expert = bool(saw_expert)
        self.expert_weight = float(expert_weight)
        self.expert = (
            nn.Linear(int(feature_dim), 1) if self.saw_expert else None
        )

    def forward(self, main_features, expert_features):
        logits = self.main(main_features)
        if self.expert is None:
            return logits
        saw_adjustment = self.expert_weight * self.expert(expert_features)
        one_hot = torch.zeros_like(logits)
        one_hot[:, SAW_INDEX : SAW_INDEX + 1] = saw_adjustment
        return logits + one_hot


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_power(values79):
    values = np.asarray(values79, dtype=np.float32)
    return np.stack(
        [values[:, get_parameter_specs(name)[0].column_index] for name in CLASS_NAMES],
        axis=1,
    )


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def _macro_aupr(targets, probabilities):
    return float(
        np.mean(
            [
                average_precision_score(targets[:, index], probabilities[:, index])
                for index in range(targets.shape[1])
            ]
        )
    )


def _binary_rank_metrics(targets, probabilities, weak_mask):
    positive = targets.astype(bool)
    weak_positive = positive & weak_mask
    negative = ~positive

    def score_pair(mask_positive, mask_negative, label_index=None):
        if label_index is None:
            positive_scores = probabilities[mask_positive]
            negative_scores = probabilities[mask_negative]
        else:
            positive_scores = probabilities[mask_positive, label_index]
            negative_scores = probabilities[mask_negative, label_index]
        binary_scores = np.concatenate([positive_scores, negative_scores])
        binary_targets = np.concatenate(
            [np.ones(positive_scores.size), np.zeros(negative_scores.size)]
        )
        if not positive_scores.size or not negative_scores.size:
            return float("nan"), float("nan")
        return (
            float(roc_auc_score(binary_targets, binary_scores)),
            float(average_precision_score(binary_targets, binary_scores)),
        )

    overall_auroc, overall_aupr = score_pair(weak_positive, negative)
    per_label = {}
    for index, name in enumerate(CLASS_NAMES):
        auroc, aupr = score_pair(
            weak_positive[:, index], negative[:, index], label_index=index
        )
        per_label[name] = {
            "weak_vs_negative_auroc": auroc,
            "weak_vs_negative_aupr": aupr,
        }
    return {
        "weak_vs_negative_auroc": overall_auroc,
        "weak_vs_negative_aupr": overall_aupr,
        "per_label": per_label,
    }


def _score_distribution(targets, probabilities, weak_mask):
    positive = targets.astype(bool)
    weak_positive = positive & weak_mask
    strong_positive = positive & ~weak_mask
    negative = ~positive

    def describe(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
        }

    return {
        "weak_positive": describe(probabilities[weak_positive]),
        "strong_positive": describe(probabilities[strong_positive]),
        "negative": describe(probabilities[negative]),
    }


def _extract_features(
    model, data_path, indices, normalization, device, batch_size
):
    dataset = V73SignalDataset(
        path=data_path,
        indices=np.asarray(indices, dtype=np.int64),
        class_mode="six",
        normalization=normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    main_features = []
    expert_features = []
    logits = []
    targets = []
    sample_ids = []
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    main, expert = model.forward_feature_branches(inputs)
                    original_logits = model.forward_classifier(main)
                main_features.append(main.detach().cpu().float().numpy())
                expert_features.append(expert.detach().cpu().float().numpy())
                logits.append(original_logits.detach().cpu().float().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(
                    np.asarray(batch["sample_index"], dtype=np.int64)
                )
    finally:
        dataset.close()
    return {
        "main": np.concatenate(main_features, axis=0),
        "expert": np.concatenate(expert_features, axis=0),
        "original_logits": np.concatenate(logits, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "sample_ids": np.concatenate(sample_ids, axis=0),
    }


def _standardize(values, mean, std):
    return ((values - mean) / std).astype(np.float32)


def _ranking_loss(logits, targets, weak_mask, margin):
    losses = []
    positive = targets > 0.5
    weak_positive = positive & weak_mask.bool()
    negative = ~positive
    for index in range(logits.shape[1]):
        weak_scores = logits[weak_positive[:, index], index]
        negative_scores = logits[negative[:, index], index]
        if not weak_scores.numel() or not negative_scores.numel():
            continue
        count = min(int(weak_scores.numel()), int(negative_scores.numel()))
        hard_negative = torch.topk(negative_scores, k=count).values
        pairwise = (
            float(margin)
            - weak_scores[:, None]
            + hard_negative[None, :]
        )
        losses.append(F.softplus(pairwise).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _head_logits(model, main, expert, device, batch_size=1024):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(main), int(batch_size)):
            stop = start + int(batch_size)
            main_tensor = torch.from_numpy(main[start:stop]).to(device)
            expert_tensor = torch.from_numpy(expert[start:stop]).to(device)
            outputs.append(
                model(main_tensor, expert_tensor).cpu().float().numpy()
            )
    return np.concatenate(outputs, axis=0)


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train_head(
    variant_name,
    specification,
    fit_main,
    fit_expert,
    fit_targets,
    fit_weak,
    validation_main,
    validation_expert,
    validation_targets,
    validation_weak,
    seed,
    device,
    args,
):
    train_seed = 71000 + int(seed) + list(VARIANTS).index(variant_name) * 1000
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    model = FrozenF2Head(
        feature_dim=fit_main.shape[1],
        num_labels=fit_targets.shape[1],
        saw_expert=specification["saw_expert"],
        expert_weight=args.saw_expert_weight,
    ).to(device)
    positives = fit_targets.sum(axis=0).astype(np.float64)
    negatives = len(fit_targets) - positives
    pos_weight = torch.as_tensor(
        negatives / np.maximum(positives, 1.0),
        dtype=torch.float32,
        device=device,
    )
    if specification["loss"] == "db":
        base_criterion = DistributionBalancedLoss(
            positive_counts=positives,
            sample_count=len(fit_targets),
        ).to(device)
    else:
        base_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dataset = TensorDataset(
        torch.from_numpy(fit_main),
        torch.from_numpy(fit_expert),
        torch.from_numpy(fit_targets.astype(np.float32)),
        torch.from_numpy(fit_weak.astype(np.uint8)),
    )
    generator = torch.Generator().manual_seed(train_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_state = None
    best_epoch = 0
    best_macro = -float("inf")
    best_weak_auroc = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_base = []
        epoch_rank = []
        epoch_total = []
        for main_batch, expert_batch, target_batch, weak_batch in loader:
            main_batch = main_batch.to(device)
            expert_batch = expert_batch.to(device)
            target_batch = target_batch.to(device)
            weak_batch = weak_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(main_batch, expert_batch)
            base_loss = base_criterion(logits, target_batch)
            if specification["loss"] == "weak_rank":
                rank_loss = _ranking_loss(
                    logits, target_batch, weak_batch, args.rank_margin
                )
            else:
                rank_loss = logits.sum() * 0.0
            loss = base_loss + args.rank_weight * rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            epoch_base.append(float(base_loss.detach().cpu()))
            epoch_rank.append(float(rank_loss.detach().cpu()))
            epoch_total.append(float(loss.detach().cpu()))
        validation_logits = _head_logits(
            model, validation_main, validation_expert, device
        )
        validation_probability = _sigmoid(validation_logits)
        validation_macro = _macro_aupr(
            validation_targets, validation_probability
        )
        validation_rank = _binary_rank_metrics(
            validation_targets, validation_probability, validation_weak
        )
        validation_weak_auroc = validation_rank["weak_vs_negative_auroc"]
        history.append(
            {
                "epoch": epoch,
                "train_base_loss": float(np.mean(epoch_base)),
                "train_rank_loss": float(np.mean(epoch_rank)),
                "train_total_loss": float(np.mean(epoch_total)),
                "validation_macro_aupr": validation_macro,
                "validation_weak_vs_negative_auroc": validation_weak_auroc,
            }
        )
        improved = validation_macro > best_macro + 1e-8
        tied_better = np.isclose(validation_macro, best_macro) and (
            validation_weak_auroc > best_weak_auroc + 1e-8
        )
        if improved or tied_better:
            best_macro = validation_macro
            best_weak_auroc = validation_weak_auroc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Frozen head training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, history, train_seed


def _threshold_diagnostics(targets, probabilities, weak_mask, thresholds):
    diagnostics = {}
    for index, name in enumerate(CLASS_NAMES):
        positives = targets[:, index].astype(bool)
        negatives = ~positives
        weak_positive = positives & weak_mask[:, index]
        prediction = probabilities[:, index] >= thresholds[index]
        diagnostics[name] = {
            "threshold": float(thresholds[index]),
            "weak_positive_count": int(weak_positive.sum()),
            "fallback_to_all_positive": not bool(weak_positive.any()),
            "weak_recall": (
                float(prediction[weak_positive].mean())
                if weak_positive.any()
                else float(prediction[positives].mean())
            ),
            "negative_fpr": (
                float(prediction[negatives].mean()) if negatives.any() else 0.0
            ),
        }
    return diagnostics


def _evaluate_variant(
    variant_name,
    logits_by_role,
    targets_by_role,
    weak_by_role,
    output_dir,
    args,
):
    calibration = fit_temperature(
        logits_by_role["calibration"],
        targets_by_role["calibration"],
        source_role="calibration",
    )
    probabilities = {
        role: _sigmoid(apply_temperature(logits, calibration.temperature))
        for role, logits in logits_by_role.items()
    }
    threshold_targets = targets_by_role["threshold"]
    threshold_probability = probabilities["threshold"]
    f1_selection = select_per_label_f1_thresholds(
        threshold_targets,
        threshold_probability,
        source_role="threshold_selection",
    )
    constrained_selection = select_per_label_weak_recall_fpr_thresholds(
        threshold_targets,
        threshold_probability,
        weak_component_mask=weak_by_role["threshold"],
        max_negative_fpr=args.max_negative_fpr,
        source_role="threshold_selection",
        grid=np.arange(0.01, 1.0, 0.01),
    )
    threshold_modes = {
        "fixed_0p5": [0.5] * len(CLASS_NAMES),
        "per_label_f1": f1_selection.thresholds,
        "weak_recall_fpr": constrained_selection.thresholds,
    }
    early_targets = targets_by_role["early"]
    early_probability = probabilities["early"]
    early_weak = weak_by_role["early"]
    metrics_by_mode = {}
    for mode, thresholds in threshold_modes.items():
        prediction = apply_thresholds(early_probability, thresholds)
        metrics = compute_multilabel_metrics(
            early_targets,
            prediction,
            probabilities=early_probability,
            class_names=CLASS_NAMES,
            tail_indices=[SAW_INDEX],
            weak_component_mask=early_weak,
        )
        negative = ~early_targets.astype(bool)
        weak_positive = early_targets.astype(bool) & early_weak
        saw_negative = early_targets[:, SAW_INDEX] == 0
        saw_weak = weak_positive[:, SAW_INDEX]
        metrics["negative_fpr"] = float(prediction[negative].mean())
        metrics["saw_negative_fpr"] = float(
            prediction[saw_negative, SAW_INDEX].mean()
        )
        metrics["saw_weak_recall"] = float(
            prediction[saw_weak, SAW_INDEX].mean()
        )
        metrics["threshold_mode"] = mode
        metrics["evaluation_role"] = "early_stop_development_diagnostic"
        metrics["locked_test_used"] = False
        metrics_by_mode[mode] = metrics
    rank_metrics = _binary_rank_metrics(
        early_targets, early_probability, early_weak
    )
    primary = metrics_by_mode["weak_recall_fpr"]
    row = {
        "variant": variant_name,
        "macro_aupr": float(primary["macro_aupr"]),
        "macro_f1": float(primary["macro_f1"]),
        "hamming_loss": float(primary["hamming_loss"]),
        "weak_component_recall": float(primary["weak_component_recall"]),
        "weak_vs_negative_auroc": rank_metrics["weak_vs_negative_auroc"],
        "weak_vs_negative_aupr": rank_metrics["weak_vs_negative_aupr"],
        "negative_fpr": float(primary["negative_fpr"]),
        "saw_aupr": float(primary["per_label"][SAW_INDEX]["aupr"]),
        "saw_weak_recall": float(primary["saw_weak_recall"]),
        "saw_weak_vs_negative_auroc": rank_metrics["per_label"]["SawFM"][
            "weak_vs_negative_auroc"
        ],
        "saw_negative_fpr": float(primary["saw_negative_fpr"]),
        "per_label_aupr": {
            item["label"]: float(item["aupr"]) for item in primary["per_label"]
        },
        "per_label_weak_rank": rank_metrics["per_label"],
        "score_distributions": _score_distribution(
            early_targets, early_probability, early_weak
        ),
        "thresholds": [float(value) for value in constrained_selection.thresholds],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(
        output_dir / "calibration.json",
        {
            "temperature": calibration.temperature,
            "loss_before": calibration.loss_before,
            "loss_after": calibration.loss_after,
            "source_role": "calibration",
        },
    )
    _write(
        output_dir / "thresholds.json",
        {
            "primary_mode": "weak_recall_fpr",
            "source_role": "threshold_selection",
            "max_negative_fpr": args.max_negative_fpr,
            "modes": threshold_modes,
            "threshold_role_diagnostics": _threshold_diagnostics(
                threshold_targets,
                threshold_probability,
                weak_by_role["threshold"],
                constrained_selection.thresholds,
            ),
            "early_role_diagnostics": _threshold_diagnostics(
                early_targets,
                early_probability,
                early_weak,
                constrained_selection.thresholds,
            ),
        },
    )
    _write(output_dir / "metrics_threshold_modes.json", metrics_by_mode)
    _write(output_dir / "metrics.json", primary)
    np.savez_compressed(
        str(output_dir / "predictions.npz"),
        logits=logits_by_role["early"],
        probabilities=early_probability,
        predictions=apply_thresholds(
            early_probability, constrained_selection.thresholds
        ),
        targets=early_targets,
    )
    return row


def _run_seed(args, config, labels, weak_mask, seed, device):
    source_dir = (
        args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    )
    receipt = _read(source_dir / "run_receipt.json")
    manifest = _read(source_dir / "longtail_manifest.json")
    if receipt.get("status") != "completed" or receipt.get("locked_test_used"):
        raise ValueError("Source F2 receipt is incomplete or used locked test")
    plan = build_run_plan(
        config,
        loss_name="db",
        seed=seed,
        max_train_samples=0,
        max_eval_samples=args.max_eval_samples,
    )
    if plan["split_sha256"] != receipt["split_sha256"]:
        raise ValueError("Split hash drifted for seed {}".format(seed))
    train_ids = np.asarray(manifest["selected_indices"], dtype=np.int64)
    role_ids = {
        "train": train_ids,
        "early": np.asarray(plan["early_stop_indices"], dtype=np.int64),
        "calibration": np.asarray(plan["calibration_indices"], dtype=np.int64),
        "threshold": np.asarray(plan["threshold_indices"], dtype=np.int64),
    }
    model, checkpoint_path = _load_backbone(
        source_dir, F2_MODEL_NAME, device
    )
    extracted = {}
    for role, indices in role_ids.items():
        extracted[role] = _extract_features(
            model,
            _data_path(config),
            indices,
            config["normalization"],
            device,
            args.feature_batch_size,
        )
        if not np.array_equal(extracted[role]["sample_ids"], indices):
            raise ValueError("Feature/sample alignment failed for {}".format(role))
        if not np.array_equal(extracted[role]["targets"], labels[indices]):
            raise ValueError("Target/sample alignment failed for {}".format(role))

    rng = np.random.default_rng(57000 + int(seed))
    order = rng.permutation(len(train_ids))
    split = int(round(len(order) * 0.8))
    fit_indices = order[:split]
    validation_indices = order[split:]
    main_mean = extracted["train"]["main"][fit_indices].mean(
        axis=0, keepdims=True
    )
    main_std = np.maximum(
        extracted["train"]["main"][fit_indices].std(axis=0, keepdims=True),
        1e-5,
    )
    expert_mean = extracted["train"]["expert"][fit_indices].mean(
        axis=0, keepdims=True
    )
    expert_std = np.maximum(
        extracted["train"]["expert"][fit_indices].std(axis=0, keepdims=True),
        1e-5,
    )
    normalized = {}
    for role in extracted:
        normalized[role] = {
            "main": _standardize(
                extracted[role]["main"], main_mean, main_std
            ),
            "expert": _standardize(
                extracted[role]["expert"], expert_mean, expert_std
            ),
        }
    targets_by_role = {
        role: extracted[role]["targets"]
        for role in ("early", "calibration", "threshold")
    }
    weak_by_role = {
        role: weak_mask[role_ids[role]]
        for role in ("early", "calibration", "threshold")
    }
    rows = []
    seed_root = args.output_root / "seed_{}".format(seed)
    for variant_name, specification in VARIANTS.items():
        print(
            "seed={} variant={}".format(seed, variant_name), flush=True
        )
        output_dir = seed_root / variant_name
        metrics_path = output_dir / "variant_summary.json"
        if metrics_path.exists() and not args.overwrite:
            rows.append(_read(metrics_path))
            continue
        if specification["train_head"]:
            head, best_epoch, history, train_seed = _train_head(
                variant_name,
                specification,
                normalized["train"]["main"][fit_indices],
                normalized["train"]["expert"][fit_indices],
                extracted["train"]["targets"][fit_indices],
                weak_mask[train_ids][fit_indices],
                normalized["train"]["main"][validation_indices],
                normalized["train"]["expert"][validation_indices],
                extracted["train"]["targets"][validation_indices],
                weak_mask[train_ids][validation_indices],
                seed,
                device,
                args,
            )
            logits_by_role = {
                role: _head_logits(
                    head,
                    normalized[role]["main"],
                    normalized[role]["expert"],
                    device,
                )
                for role in ("early", "calibration", "threshold")
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": head.state_dict(),
                    "variant": variant_name,
                    "best_epoch": best_epoch,
                    "train_seed": train_seed,
                    "main_mean": main_mean,
                    "main_std": main_std,
                    "expert_mean": expert_mean,
                    "expert_std": expert_std,
                },
                output_dir / "best_head.pth",
            )
            _write_history(output_dir / "history.csv", history)
        else:
            best_epoch = int(receipt["best_epoch"])
            train_seed = None
            logits_by_role = {
                role: extracted[role]["original_logits"]
                for role in ("early", "calibration", "threshold")
            }
        row = _evaluate_variant(
            variant_name,
            logits_by_role,
            targets_by_role,
            weak_by_role,
            output_dir,
            args,
        )
        row.update(
            {
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "train_seed": train_seed,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "split_sha256": receipt["split_sha256"],
                "manifest_sha256": receipt["manifest_sha256"],
                "train_samples": int(len(fit_indices)),
                "head_validation_samples": int(len(validation_indices)),
                "calibration_samples": int(len(role_ids["calibration"])),
                "threshold_samples": int(len(role_ids["threshold"])),
                "evaluation_samples": int(len(role_ids["early"])),
                "locked_test_used": False,
            }
        )
        _write(metrics_path, row)
        _write(
            output_dir / "run_receipt.json",
            {
                "status": "completed",
                "variant": variant_name,
                "seed": int(seed),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "split_sha256": row["split_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "locked_test_used": False,
            },
        )
        rows.append(row)
    return rows


def _aggregate(rows):
    aggregate = {}
    for variant_index, variant_name in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant_name]
        aggregate[variant_name] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                81000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(HEAD_METRICS)
        }
        aggregate[variant_name]["mean_thresholds"] = np.mean(
            [row["thresholds"] for row in selected], axis=0
        ).tolist()
    return aggregate


def _compare(rows, candidate, baseline, seed_offset):
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        candidate_row = next(
            row
            for row in rows
            if row["seed"] == seed and row["variant"] == candidate
        )
        baseline_row = next(
            row
            for row in rows
            if row["seed"] == seed and row["variant"] == baseline
        )
        delta = {
            metric: candidate_row[metric] - baseline_row[metric]
            for metric in HEAD_METRICS
        }
        paired.append({"seed": seed, "delta": delta})
    aggregate = {
        metric: _mean_ci(
            [row["delta"][metric] for row in paired],
            seed_offset + index,
        )
        for index, metric in enumerate(HEAD_METRICS)
    }
    aggregate["weak_auroc_positive_seeds"] = sum(
        row["delta"]["weak_vs_negative_auroc"] > 0 for row in paired
    )
    aggregate["weak_recall_positive_seeds"] = sum(
        row["delta"]["weak_component_recall"] > 0 for row in paired
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "per_seed": paired,
        "aggregate": aggregate,
    }


def _promotion_gate(comparison, aggregate):
    delta = comparison["aggregate"]
    h4 = aggregate["H4_WeakRankSawExpert"]
    checks = {
        "macro_noninferior": delta["macro_aupr"]["mean"] >= -0.005,
        "weak_auroc_gain": (
            delta["weak_vs_negative_auroc"]["mean"] >= 0.015
        ),
        "weak_recall_gain": (
            delta["weak_component_recall"]["mean"] >= 0.050
        ),
        "hamming_noninferior": delta["hamming_loss"]["mean"] <= 0.005,
        "negative_fpr_absolute": h4["negative_fpr"]["mean"] <= 0.25,
        "negative_fpr_delta": delta["negative_fpr"]["mean"] <= 0.02,
        "saw_weak_auroc_gain": (
            delta["saw_weak_vs_negative_auroc"]["mean"] >= 0.020
        ),
        "saw_aupr_noninferior": delta["saw_aupr"]["mean"] >= 0.0,
        "weak_auroc_positive_at_least_two": (
            delta["weak_auroc_positive_seeds"] >= 2
        ),
        "weak_recall_positive_at_least_two": (
            delta["weak_recall_positive_seeds"] >= 2
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _threshold_analysis(output_root, seeds):
    mode_rows = []
    constraint_violations = []
    early_drift = []
    for seed in seeds:
        for variant_name in VARIANTS:
            directory = (
                output_root / "seed_{}".format(seed) / variant_name
            )
            modes = _read(directory / "metrics_threshold_modes.json")
            constrained = modes["weak_recall_fpr"]
            f1 = modes["per_label_f1"]
            mode_rows.append(
                {
                    "seed": int(seed),
                    "variant": variant_name,
                    "delta_weak_component_recall": (
                        constrained["weak_component_recall"]
                        - f1["weak_component_recall"]
                    ),
                    "delta_negative_fpr": (
                        constrained["negative_fpr"] - f1["negative_fpr"]
                    ),
                    "delta_macro_f1": (
                        constrained["macro_f1"] - f1["macro_f1"]
                    ),
                    "delta_hamming_loss": (
                        constrained["hamming_loss"] - f1["hamming_loss"]
                    ),
                }
            )
            thresholds = _read(directory / "thresholds.json")
            for label_name in CLASS_NAMES:
                selected = thresholds["threshold_role_diagnostics"][label_name]
                evaluated = thresholds["early_role_diagnostics"][label_name]
                if selected["negative_fpr"] > 0.2000001:
                    constraint_violations.append(
                        {
                            "seed": int(seed),
                            "variant": variant_name,
                            "label": label_name,
                            "negative_fpr": selected["negative_fpr"],
                        }
                    )
                if evaluated["negative_fpr"] > 0.25:
                    early_drift.append(
                        {
                            "seed": int(seed),
                            "variant": variant_name,
                            "label": label_name,
                            "threshold_role_negative_fpr": selected[
                                "negative_fpr"
                            ],
                            "early_role_negative_fpr": evaluated[
                                "negative_fpr"
                            ],
                        }
                    )
    aggregate = {}
    for variant_index, variant_name in enumerate(VARIANTS):
        selected = [
            row for row in mode_rows if row["variant"] == variant_name
        ]
        aggregate[variant_name] = {
            key: _mean_ci(
                [row[key] for row in selected],
                83000 + variant_index * 100 + key_index,
            )
            for key_index, key in enumerate(
                (
                    "delta_weak_component_recall",
                    "delta_negative_fpr",
                    "delta_macro_f1",
                    "delta_hamming_loss",
                )
            )
        }
    return {
        "comparison": "weak_recall_fpr_minus_per_label_f1",
        "per_run": mode_rows,
        "aggregate": aggregate,
        "threshold_role_constraint_violations": constraint_violations,
        "early_role_fpr_over_0p25": early_drift,
    }


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    labels = select_labels(read_labels(data_path), "six")
    values79, _ = read_parameters(data_path)
    weak_mask = compute_weak_component_mask(
        labels, _component_power(values79), args.weak_threshold_db
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for seed in args.seeds:
        all_rows.extend(
            _run_seed(args, config, labels, weak_mask, seed, device)
        )
    aggregate = _aggregate(all_rows)
    comparisons = {
        "h1_vs_j0": _compare(all_rows, "H1_FrozenDB", "J0_JointDB", 82000),
        "h2_vs_h1": _compare(all_rows, "H2_PosBCE", "H1_FrozenDB", 82100),
        "h3_vs_h2": _compare(all_rows, "H3_WeakRank", "H2_PosBCE", 82200),
        "h4_vs_h3": _compare(
            all_rows, "H4_WeakRankSawExpert", "H3_WeakRank", 82300
        ),
        "h4_vs_h1": _compare(
            all_rows, "H4_WeakRankSawExpert", "H1_FrozenDB", 82400
        ),
        "h4_vs_j0": _compare(
            all_rows, "H4_WeakRankSawExpert", "J0_JointDB", 82500
        ),
    }
    gate = _promotion_gate(comparisons["h4_vs_h1"], aggregate)
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "device": str(device),
        "protocol": {
            "seeds": list(args.seeds),
            "variants": VARIANTS,
            "head_fit_fraction": 0.8,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "rank_weight": args.rank_weight,
            "rank_margin": args.rank_margin,
            "saw_expert_weight": args.saw_expert_weight,
            "max_negative_fpr": args.max_negative_fpr,
            "weak_threshold_db": args.weak_threshold_db,
            "primary_threshold_mode": "weak_recall_fpr",
        },
        "per_run": all_rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "threshold_analysis": _threshold_analysis(args.output_root, args.seeds),
        "promotion_gate": gate,
        "unfreeze_decision": (
            "run_last_stage_unfreeze"
            if gate["passed"]
            else "stop_before_unfreeze"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_dfms_iq_frozen_heads_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "tapc"
            / "route_a_dfms_iq_structure_screen"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "tapc"
            / "route_a_dfms_iq_frozen_heads"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--rank-margin", type=float, default=0.20)
    parser.add_argument("--saw-expert-weight", type=float, default=0.20)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
