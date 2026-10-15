"""Frozen F2 feature-sufficiency probes and hard-bounded weak residual screen."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import (  # noqa: E402
    CLASS_NAMES,
    HEAD_METRICS,
    _binary_rank_metrics,
    _component_power,
    _evaluate_variant,
    _sha256,
    _sigmoid,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_label_local_saw import (  # noqa: E402
    F2_MODEL_NAME,
    _extract_maps,
    _load_h2_head,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


REPRESENTATIONS = ("G_Global", "T_Temporal", "S2_Physics", "F_Fused")
RESIDUAL_VARIANTS = {
    "G_Global": "R1_GlobalBound",
    "T_Temporal": "R2_TemporalBound",
    "S2_Physics": "R3_Stage2Bound",
    "F_Fused": "R4_FusedBound",
}
SAW_INDEX = 1


def _topk_mean(values, topk=8):
    count = min(int(topk), int(values.shape[-1]))
    partitioned = np.partition(values, values.shape[-1] - count, axis=-1)
    return partitioned[..., -count:].mean(axis=-1)


def _statistics(values, topk=8):
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=-1)
    variance = np.maximum(np.square(values).mean(axis=-1) - np.square(mean), 1e-5)
    return np.concatenate(
        [mean, np.sqrt(variance), values.max(axis=-1), _topk_mean(values, topk)],
        axis=1,
    ).astype(np.float32)


def _temporal_summary(values, chunk_size=64):
    rows = []
    for start in range(0, len(values), int(chunk_size)):
        rows.append(_statistics(values[start : start + int(chunk_size)]))
    return np.concatenate(rows, axis=0)


def _previous(values):
    return np.concatenate([values[..., :1], values[..., :-1]], axis=-1)


def _stage2_summary(real, imag, chunk_size=32):
    rows = []
    for start in range(0, len(real), int(chunk_size)):
        stop = start + int(chunk_size)
        real_chunk = np.asarray(real[start:stop], dtype=np.float32)
        imag_chunk = np.asarray(imag[start:stop], dtype=np.float32)
        magnitude = np.sqrt(
            np.maximum(np.square(real_chunk) + np.square(imag_chunk), 1e-6)
        )
        normalized_real = real_chunk / magnitude
        normalized_imag = imag_chunk / magnitude
        previous_real = _previous(normalized_real)
        previous_imag = _previous(normalized_imag)
        phase_cos = normalized_real * previous_real + normalized_imag * previous_imag
        phase_sin = normalized_imag * previous_real - normalized_real * previous_imag
        descriptors = (
            magnitude,
            np.abs(magnitude - _previous(magnitude)),
            phase_cos,
            phase_sin,
            np.abs(phase_cos - _previous(phase_cos)),
            np.abs(phase_sin - _previous(phase_sin)),
        )
        rows.append(np.concatenate([_statistics(item) for item in descriptors], axis=1))
    return np.concatenate(rows, axis=0).astype(np.float32)


def _summarize_maps(arrays):
    global_features = np.asarray(arrays["global"], dtype=np.float32)
    temporal = _temporal_summary(arrays["temporal"])
    stage2 = _stage2_summary(arrays["stage2_real"], arrays["stage2_imag"])
    return {
        "G_Global": global_features,
        "T_Temporal": temporal,
        "S2_Physics": stage2,
        "F_Fused": np.concatenate([global_features, temporal, stage2], axis=1),
        "base_logits": np.asarray(arrays["base_logits"], dtype=np.float32),
        "targets": np.asarray(arrays["targets"], dtype=np.uint8),
        "sample_ids": np.asarray(arrays["sample_ids"], dtype=np.int64),
    }


def _standardization(features, fit_indices):
    mean = features[fit_indices].mean(axis=0, keepdims=True)
    std = np.maximum(features[fit_indices].std(axis=0, keepdims=True), 1e-5)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(features, mean, std):
    return ((features - mean) / std).astype(np.float32)


def _safe_rank(targets, scores):
    targets = np.asarray(targets, dtype=np.uint8)
    if targets.min() == targets.max():
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(targets, scores)),
        float(average_precision_score(targets, scores)),
    )


def _train_label_probe(
    fit_features,
    fit_targets,
    fit_weak,
    validation_features,
    validation_targets,
    validation_weak,
    label_index,
    train_seed,
    device,
    args,
):
    torch.manual_seed(int(train_seed))
    np.random.seed(int(train_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(train_seed))
    fit_positive = fit_targets[:, label_index].astype(bool)
    fit_mask = fit_weak[:, label_index] | ~fit_positive
    fit_binary = fit_weak[fit_mask, label_index].astype(np.float32)
    validation_positive = validation_targets[:, label_index].astype(bool)
    validation_mask = validation_weak[:, label_index] | ~validation_positive
    validation_binary = validation_weak[validation_mask, label_index].astype(np.uint8)
    positive_count = float(fit_binary.sum())
    negative_count = float(len(fit_binary) - positive_count)
    if positive_count < 2 or negative_count < 2:
        raise ValueError("Insufficient weak-positive/negative probe samples")
    model = nn.Linear(fit_features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    x_fit = torch.from_numpy(fit_features[fit_mask]).to(device)
    y_fit = torch.from_numpy(fit_binary[:, None]).to(device)
    x_validation = torch.from_numpy(validation_features[validation_mask]).to(device)
    best_state = None
    best_epoch = 0
    best_auroc = -float("inf")
    best_aupr = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_fit)
        loss = criterion(logits, y_fit)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_scores = model(x_validation).squeeze(1).cpu().numpy()
        validation_auroc, validation_aupr = _safe_rank(
            validation_binary, validation_scores
        )
        history.append(
            {
                "label": CLASS_NAMES[label_index],
                "epoch": int(epoch),
                "train_loss": float(loss.detach().cpu()),
                "validation_weak_auroc": validation_auroc,
                "validation_weak_aupr": validation_aupr,
            }
        )
        improved = validation_auroc > best_auroc + 1e-8
        tied_better = np.isclose(validation_auroc, best_auroc) and (
            validation_aupr > best_aupr + 1e-8
        )
        if improved or tied_better:
            best_auroc = validation_auroc
            best_aupr = validation_aupr
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("Probe training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, history, {
        "fit_weak_positive_count": int(positive_count),
        "fit_negative_count": int(negative_count),
        "validation_weak_positive_count": int(validation_binary.sum()),
        "validation_negative_count": int(len(validation_binary) - validation_binary.sum()),
        "best_validation_weak_auroc": float(best_auroc),
        "best_validation_weak_aupr": float(best_aupr),
    }


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _probe_scores(weights, bias, features):
    return features @ weights.T + bias[None, :]


def _bounded_logits(base_logits, raw_scores, center, scale, max_residual):
    standardized = (raw_scores - center[None, :]) / scale[None, :]
    gate = np.exp(-np.abs(base_logits))
    residual = float(max_residual) * np.tanh(standardized) * gate
    return (base_logits + residual).astype(np.float32), residual.astype(np.float32)


def _fit_representation(
    representation,
    normalized,
    targets,
    weak,
    fit_indices,
    validation_indices,
    seed,
    device,
    args,
    output_dir,
):
    weights = []
    biases = []
    best_epochs = []
    histories = []
    counts = {}
    for label_index, label_name in enumerate(CLASS_NAMES):
        train_seed = 121000 + int(seed) + REPRESENTATIONS.index(representation) * 1000 + label_index * 100
        model, best_epoch, history, label_counts = _train_label_probe(
            normalized[fit_indices],
            targets[fit_indices],
            weak[fit_indices],
            normalized[validation_indices],
            targets[validation_indices],
            weak[validation_indices],
            label_index,
            train_seed,
            device,
            args,
        )
        weights.append(model.weight.detach().cpu().numpy()[0])
        biases.append(float(model.bias.detach().cpu().numpy()[0]))
        best_epochs.append(int(best_epoch))
        histories.extend(history)
        counts[label_name] = label_counts
    weights = np.stack(weights, axis=0).astype(np.float32)
    biases = np.asarray(biases, dtype=np.float32)
    fit_raw = _probe_scores(weights, biases, normalized[fit_indices])
    center = fit_raw.mean(axis=0).astype(np.float32)
    score_scale = np.maximum(fit_raw.std(axis=0), 1e-5).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "oracle_probe.pth"
    torch.save(
        {
            "representation": representation,
            "weights": weights,
            "bias": biases,
            "raw_center": center,
            "raw_scale": score_scale,
            "best_epochs": best_epochs,
            "counts": counts,
        },
        checkpoint_path,
    )
    _write_history(output_dir / "history.csv", histories)
    return weights, biases, center, score_scale, best_epochs, counts, checkpoint_path


def _aggregate(rows, variants):
    result = {}
    for variant_index, variant in enumerate(variants):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                131000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(HEAD_METRICS)
        }
    return result


def _paired_comparison(rows, candidate, baseline, seed_offset):
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        candidate_row = next(
            row for row in rows if row["seed"] == seed and row["variant"] == candidate
        )
        baseline_row = next(
            row for row in rows if row["seed"] == seed and row["variant"] == baseline
        )
        paired.append(
            {
                "seed": int(seed),
                "delta": {
                    metric: candidate_row[metric] - baseline_row[metric]
                    for metric in HEAD_METRICS
                },
            }
        )
    aggregate = {
        metric: _mean_ci(
            [row["delta"][metric] for row in paired], seed_offset + index
        )
        for index, metric in enumerate(HEAD_METRICS)
    }
    aggregate["weak_auroc_positive_seeds"] = sum(
        row["delta"]["weak_vs_negative_auroc"] > 0 for row in paired
    )
    aggregate["weak_recall_positive_seeds"] = sum(
        row["delta"]["weak_component_recall"] > 0 for row in paired
    )
    return {"candidate": candidate, "baseline": baseline, "per_seed": paired, "aggregate": aggregate}


def _residual_gate(comparison, aggregate):
    delta = comparison["aggregate"]
    candidate = aggregate[comparison["candidate"]]
    checks = {
        "macro_noninferior": delta["macro_aupr"]["mean"] >= -0.003,
        "weak_auroc_gain": delta["weak_vs_negative_auroc"]["mean"] >= 0.010,
        "weak_recall_gain": delta["weak_component_recall"]["mean"] >= 0.020,
        "hamming_noninferior": delta["hamming_loss"]["mean"] <= 0.005,
        "negative_fpr_delta": delta["negative_fpr"]["mean"] <= 0.010,
        "negative_fpr_absolute": candidate["negative_fpr"]["mean"] <= 0.25,
        "saw_aupr_noninferior": delta["saw_aupr"]["mean"] >= 0.0,
        "weak_auroc_two_seeds": delta["weak_auroc_positive_seeds"] >= 2,
        "weak_recall_two_seeds": delta["weak_recall_positive_seeds"] >= 2,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _oracle_aggregate(rows):
    result = {}
    for representation_index, representation in enumerate(REPRESENTATIONS):
        selected = [row for row in rows if row["representation"] == representation]
        result[representation] = {
            "weak_vs_negative_auroc": _mean_ci(
                [row["weak_vs_negative_auroc"] for row in selected],
                141000 + representation_index * 100,
            ),
            "weak_vs_negative_aupr": _mean_ci(
                [row["weak_vs_negative_aupr"] for row in selected],
                141001 + representation_index * 100,
            ),
            "per_label": {
                name: {
                    "weak_vs_negative_auroc": _mean_ci(
                        [row["per_label"][name]["weak_vs_negative_auroc"] for row in selected],
                        142000 + representation_index * 100 + label_index,
                    ),
                    "weak_vs_negative_aupr": _mean_ci(
                        [row["per_label"][name]["weak_vs_negative_aupr"] for row in selected],
                        143000 + representation_index * 100 + label_index,
                    ),
                }
                for label_index, name in enumerate(CLASS_NAMES)
            },
        }
    return result


def _oracle_gates(oracle_rows, baseline_rows, oracle_aggregate):
    baseline_by_seed = {
        row["seed"]: row["weak_vs_negative_auroc"] for row in baseline_rows
    }
    gates = {}
    for representation in REPRESENTATIONS:
        selected = [row for row in oracle_rows if row["representation"] == representation]
        deltas = [row["weak_vs_negative_auroc"] - baseline_by_seed[row["seed"]] for row in selected]
        saw_deltas = [
            row["per_label"]["SawFM"]["weak_vs_negative_auroc"]
            - next(item for item in baseline_rows if item["seed"] == row["seed"])["per_label"]["SawFM"]["weak_vs_negative_auroc"]
            for row in selected
        ]
        checks = {
            "overall_gain_0p02": float(np.mean(deltas)) >= 0.020,
            "overall_gain_two_seeds": sum(value > 0 for value in deltas) >= 2,
        }
        if representation == "S2_Physics":
            checks.update(
                {
                    "saw_absolute_0p55": oracle_aggregate[representation]["per_label"]["SawFM"]["weak_vs_negative_auroc"]["mean"] >= 0.55,
                    "saw_gain_0p02": float(np.mean(saw_deltas)) >= 0.020,
                    "saw_gain_two_seeds": sum(value > 0 for value in saw_deltas) >= 2,
                }
            )
        gates[representation] = {
            "overall_delta_mean": float(np.mean(deltas)),
            "overall_positive_seeds": int(sum(value > 0 for value in deltas)),
            "checks": checks,
            "passed": all(checks.values()),
        }
    label_insufficiency = {}
    for label_name in CLASS_NAMES:
        best = max(
            oracle_aggregate[representation]["per_label"][label_name]["weak_vs_negative_auroc"]["mean"]
            for representation in REPRESENTATIONS
        )
        label_insufficiency[label_name] = {
            "best_oracle_weak_auroc": float(best),
            "feature_insufficient": bool(best < 0.55),
        }
    return gates, label_insufficiency


def _run_seed(args, config, labels, weak_mask, seed, device):
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    source_receipt = _read(source_dir / "run_receipt.json")
    manifest = _read(source_dir / "longtail_manifest.json")
    if source_receipt.get("status") != "completed" or source_receipt.get("locked_test_used"):
        raise ValueError("Source F2 receipt is incomplete or used locked test")
    plan = build_run_plan(
        config,
        loss_name="db",
        seed=seed,
        max_train_samples=0,
        max_eval_samples=args.max_eval_samples,
    )
    if plan["split_sha256"] != source_receipt["split_sha256"]:
        raise ValueError("Split hash drifted for seed {}".format(seed))
    role_ids = {
        "train": np.asarray(manifest["selected_indices"], dtype=np.int64),
        "early": np.asarray(plan["early_stop_indices"], dtype=np.int64),
        "calibration": np.asarray(plan["calibration_indices"], dtype=np.int64),
        "threshold": np.asarray(plan["threshold_indices"], dtype=np.int64),
    }
    backbone, f2_checkpoint_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    base_head, h2_checkpoint, h2_checkpoint_path, _ = _load_h2_head(
        seed, args.frozen_head_root, device
    )
    summarized = {}
    for role, ids in role_ids.items():
        print("seed={} extracting role={}".format(seed, role), flush=True)
        maps = _extract_maps(
            backbone,
            base_head,
            h2_checkpoint["main_mean"],
            h2_checkpoint["main_std"],
            _data_path(config),
            ids,
            config["normalization"],
            device,
            args.feature_batch_size,
        )
        summarized[role] = _summarize_maps(maps)
        if not np.array_equal(summarized[role]["sample_ids"], ids):
            raise ValueError("Feature/sample alignment failed for {}".format(role))
        if not np.array_equal(summarized[role]["targets"], labels[ids]):
            raise ValueError("Target/sample alignment failed for {}".format(role))
        del maps
    rng = np.random.default_rng(57000 + int(seed))
    order = rng.permutation(len(role_ids["train"]))
    split = int(round(len(order) * 0.8))
    fit_indices = order[:split]
    validation_indices = order[split:]
    train_targets = summarized["train"]["targets"]
    train_weak = weak_mask[role_ids["train"]]
    targets_by_role = {
        role: summarized[role]["targets"] for role in ("early", "calibration", "threshold")
    }
    weak_by_role = {
        role: weak_mask[role_ids[role]] for role in ("early", "calibration", "threshold")
    }
    logits_by_role = {
        role: summarized[role]["base_logits"] for role in ("early", "calibration", "threshold")
    }
    seed_root = args.output_root / "seed_{}".format(seed)
    control_dir = seed_root / "C0_H2"
    control = _evaluate_variant(
        "C0_H2", logits_by_role, targets_by_role, weak_by_role, control_dir, args
    )
    control.update(
        {
            "seed": int(seed),
            "split_sha256": source_receipt["split_sha256"],
            "manifest_sha256": source_receipt["manifest_sha256"],
            "f2_checkpoint_sha256": _sha256(f2_checkpoint_path),
            "h2_checkpoint_sha256": _sha256(h2_checkpoint_path),
            "locked_test_used": False,
        }
    )
    _write(control_dir / "variant_summary.json", control)
    _write(
        control_dir / "run_receipt.json",
        {
            "status": "completed",
            "variant": "C0_H2",
            "seed": int(seed),
            "split_sha256": control["split_sha256"],
            "manifest_sha256": control["manifest_sha256"],
            "f2_checkpoint_sha256": control["f2_checkpoint_sha256"],
            "h2_checkpoint_sha256": control["h2_checkpoint_sha256"],
            "locked_test_used": False,
        },
    )
    baseline_rank = _binary_rank_metrics(
        targets_by_role["early"],
        _sigmoid(logits_by_role["early"]),
        weak_by_role["early"],
    )
    baseline_rank.update({"seed": int(seed), "representation": "C0_H2"})
    residual_rows = [control]
    oracle_rows = []
    feature_statistics = {}
    for representation in REPRESENTATIONS:
        print("seed={} fitting representation={}".format(seed, representation), flush=True)
        mean, std = _standardization(summarized["train"][representation], fit_indices)
        normalized = {
            role: _standardize(summarized[role][representation], mean, std)
            for role in summarized
        }
        probe_dir = seed_root / "oracle_{}".format(representation)
        weights, biases, center, score_scale, best_epochs, counts, checkpoint_path = _fit_representation(
            representation,
            normalized["train"],
            train_targets,
            train_weak,
            fit_indices,
            validation_indices,
            seed,
            device,
            args,
            probe_dir,
        )
        raw_by_role = {
            role: _probe_scores(weights, biases, normalized[role])
            for role in ("early", "calibration", "threshold")
        }
        oracle_rank = _binary_rank_metrics(
            targets_by_role["early"],
            _sigmoid(raw_by_role["early"]),
            weak_by_role["early"],
        )
        oracle_row = {
            "seed": int(seed),
            "representation": representation,
            "feature_dim": int(normalized["train"].shape[1]),
            "weak_vs_negative_auroc": oracle_rank["weak_vs_negative_auroc"],
            "weak_vs_negative_aupr": oracle_rank["weak_vs_negative_aupr"],
            "per_label": oracle_rank["per_label"],
            "best_epochs": best_epochs,
            "counts": counts,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "locked_test_used": False,
        }
        oracle_rows.append(oracle_row)
        _write(probe_dir / "oracle_metrics.json", oracle_row)
        _write(
            probe_dir / "run_receipt.json",
            {
                "status": "completed",
                "variant": "oracle_{}".format(representation),
                "seed": int(seed),
                "representation": representation,
                "probe_checkpoint_sha256": oracle_row["checkpoint_sha256"],
                "split_sha256": source_receipt["split_sha256"],
                "manifest_sha256": source_receipt["manifest_sha256"],
                "f2_checkpoint_sha256": _sha256(f2_checkpoint_path),
                "h2_checkpoint_sha256": _sha256(h2_checkpoint_path),
                "locked_test_used": False,
            },
        )
        np.savez_compressed(
            str(probe_dir / "oracle_predictions.npz"),
            raw_scores=raw_by_role["early"],
            targets=targets_by_role["early"],
            weak_mask=weak_by_role["early"],
        )
        variant = RESIDUAL_VARIANTS[representation]
        residual_dir = seed_root / variant
        bounded_by_role = {}
        residual_by_role = {}
        for role in ("early", "calibration", "threshold"):
            bounded_by_role[role], residual_by_role[role] = _bounded_logits(
                logits_by_role[role], raw_by_role[role], center, score_scale, args.max_residual
            )
        row = _evaluate_variant(
            variant, bounded_by_role, targets_by_role, weak_by_role, residual_dir, args
        )
        residual_abs = np.abs(residual_by_role["early"])
        row.update(
            {
                "seed": int(seed),
                "representation": representation,
                "feature_dim": int(normalized["train"].shape[1]),
                "max_residual": float(args.max_residual),
                "observed_max_abs_residual": float(residual_abs.max()),
                "mean_abs_residual": float(residual_abs.mean()),
                "residual_mean_per_label": residual_by_role["early"].mean(axis=0).tolist(),
                "probe_checkpoint_sha256": _sha256(checkpoint_path),
                "split_sha256": source_receipt["split_sha256"],
                "manifest_sha256": source_receipt["manifest_sha256"],
                "f2_checkpoint_sha256": _sha256(f2_checkpoint_path),
                "h2_checkpoint_sha256": _sha256(h2_checkpoint_path),
                "locked_test_used": False,
            }
        )
        residual_rows.append(row)
        _write(residual_dir / "variant_summary.json", row)
        np.savez_compressed(
            str(residual_dir / "bounded_residual_predictions.npz"),
            logits=bounded_by_role["early"],
            residual=residual_by_role["early"],
            targets=targets_by_role["early"],
        )
        _write(
            residual_dir / "run_receipt.json",
            {
                "status": "completed",
                "variant": variant,
                "seed": int(seed),
                "representation": representation,
                "probe_checkpoint_sha256": row["probe_checkpoint_sha256"],
                "split_sha256": row["split_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "f2_checkpoint_sha256": row["f2_checkpoint_sha256"],
                "h2_checkpoint_sha256": row["h2_checkpoint_sha256"],
                "locked_test_used": False,
            },
        )
        feature_statistics[representation] = {
            "feature_dim": int(mean.shape[1]),
            "standardization_mean_abs": float(np.abs(mean).mean()),
            "standardization_std_median": float(np.median(std)),
            "raw_score_center": center.tolist(),
            "raw_score_scale": score_scale.tolist(),
        }
    _write(seed_root / "feature_statistics.json", feature_statistics)
    return residual_rows, oracle_rows, baseline_rank


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    labels = select_labels(read_labels(data_path), "six")
    values79, _ = read_parameters(data_path)
    weak_mask = compute_weak_component_mask(
        labels, _component_power(values79), args.weak_threshold_db
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_rows = []
    oracle_rows = []
    baseline_rows = []
    for seed in args.seeds:
        seed_residual, seed_oracle, seed_baseline = _run_seed(
            args, config, labels, weak_mask, seed, device
        )
        residual_rows.extend(seed_residual)
        oracle_rows.extend(seed_oracle)
        baseline_rows.append(seed_baseline)
    variants = ("C0_H2",) + tuple(RESIDUAL_VARIANTS.values())
    aggregate = _aggregate(residual_rows, variants)
    comparisons = {
        variant: _paired_comparison(
            residual_rows, variant, "C0_H2", 151000 + index * 100
        )
        for index, variant in enumerate(RESIDUAL_VARIANTS.values())
    }
    residual_gates = {
        variant: _residual_gate(comparisons[variant], aggregate)
        for variant in RESIDUAL_VARIANTS.values()
    }
    oracle_aggregate = _oracle_aggregate(oracle_rows)
    oracle_gates, label_insufficiency = _oracle_gates(
        oracle_rows, baseline_rows, oracle_aggregate
    )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "device": str(device),
        "protocol": {
            "seeds": list(args.seeds),
            "representations": list(REPRESENTATIONS),
            "base": "frozen F2 + frozen H2 PosBCE",
            "weak_threshold_db": float(args.weak_threshold_db),
            "max_residual": float(args.max_residual),
            "uncertainty_gate": "exp(-abs(H2_logit))",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
        },
        "oracle_per_run": oracle_rows,
        "oracle_baseline_per_run": baseline_rows,
        "oracle_aggregate": oracle_aggregate,
        "oracle_gates": oracle_gates,
        "label_feature_insufficiency": label_insufficiency,
        "residual_per_run": residual_rows,
        "residual_aggregate": aggregate,
        "residual_comparisons": comparisons,
        "residual_gates": residual_gates,
        "unfreeze_decision": (
            "eligible_for_last_stage_unfreeze"
            if any(item["passed"] for item in residual_gates.values())
            else "stop_before_unfreeze"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(
        args.output_root / "route_a_dfms_iq_feature_sufficiency_summary.json",
        payload,
    )
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
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen",
    )
    parser.add_argument(
        "--frozen-head-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_feature_sufficiency",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--max-residual", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
