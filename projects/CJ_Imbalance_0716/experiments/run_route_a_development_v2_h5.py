"""Development-v2 controlled five-label head ablation for Route A.

The consumed historical locked test is never opened.  A fresh development-v2
train file is split into optimization/early/calibration/threshold roles.  A
label-LT manifest is rebuilt inside optimization, then paired C6 and H5 linear
heads are trained on frozen historical F2 features.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import (  # noqa: E402
    build_label_longtail,
    compute_weak_component_mask,
)
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.split_builder import build_stratified_roles  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.thresholding import (  # noqa: E402
    apply_thresholds,
    select_per_label_weak_recall_fpr_thresholds,
)
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import (  # noqa: E402
    _extract_features,
    _load_backbone,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


CLASS_NAMES6 = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
COMMON_INDICES = np.asarray([0, 2, 3, 4, 5], dtype=np.int64)
COMMON_NAMES = tuple(CLASS_NAMES6[index] for index in COMMON_INDICES)
TARGET_COMMON_POSITIONS = np.asarray([2, 3, 4], dtype=np.int64)
TARGET_NAMES = tuple(COMMON_NAMES[index] for index in TARGET_COMMON_POSITIONS)
LABEL_LT_TARGET_COUNTS = np.asarray(
    [1600, 400, 1213, 919, 696, 528], dtype=np.int64
)
F2_MODEL_NAME = "F2_DFMS_Full"
VARIANTS = ("C6_AllCal", "C6_CommonCal", "H5_CommonCal")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError("Not JSON serializable: {!r}".format(type(value)))


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_history(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_sha256(*arrays):
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    return digest.hexdigest()


def _set_seed(seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sigmoid(values):
    logits = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(
        np.float32
    )


def _macro_aupr(targets, probabilities):
    truth = np.asarray(targets, dtype=np.uint8)
    scores = np.asarray(probabilities, dtype=np.float64)
    values = []
    for index in range(truth.shape[1]):
        if 0 < int(truth[:, index].sum()) < len(truth):
            values.append(average_precision_score(truth[:, index], scores[:, index]))
    return float(np.mean(values))


def _component_power(values79):
    values = np.asarray(values79, dtype=np.float32)
    return np.stack(
        [
            values[:, get_parameter_specs(name)[0].column_index]
            for name in CLASS_NAMES6
        ],
        axis=1,
    )


def _target_weak_rank(targets, probabilities, weak_mask):
    truth = np.asarray(targets, dtype=np.uint8)
    scores = np.asarray(probabilities, dtype=np.float64)
    weak = np.asarray(weak_mask, dtype=bool)
    positive_scores = []
    negative_scores = []
    for position in TARGET_COMMON_POSITIONS:
        weak_positive = (truth[:, position] == 1) & weak[:, position]
        negative = truth[:, position] == 0
        positive_scores.append(scores[weak_positive, position])
        negative_scores.append(scores[negative, position])
    positive_values = np.concatenate(positive_scores)
    negative_values = np.concatenate(negative_scores)
    if not len(positive_values) or not len(negative_values):
        return float("nan")
    labels = np.concatenate(
        [np.ones(len(positive_values)), np.zeros(len(negative_values))]
    )
    values = np.concatenate([positive_values, negative_values])
    return float(roc_auc_score(labels, values))


def _paired_heads(feature_dim, seed):
    _set_seed(seed)
    c6 = nn.Linear(int(feature_dim), 6)
    h5 = nn.Linear(int(feature_dim), 5)
    with torch.no_grad():
        h5.weight.copy_(c6.weight[torch.as_tensor(COMMON_INDICES)])
        h5.bias.copy_(c6.bias[torch.as_tensor(COMMON_INDICES)])
    return c6, h5


def _train_head(
    model,
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    validation_common_targets,
    validation_common_weak,
    common_output_indices,
    seed,
    device,
    args,
    output_dir,
):
    _set_seed(seed)
    model = model.to(device)
    positives = train_targets.sum(axis=0).astype(np.float64)
    negatives = len(train_targets) - positives
    pos_weight = torch.as_tensor(
        negatives / np.maximum(positives, 1.0),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32)),
        torch.from_numpy(train_targets.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_tensor = torch.from_numpy(
        validation_features.astype(np.float32)
    ).to(device)
    common_index_tensor = torch.as_tensor(
        common_output_indices, dtype=torch.long, device=device
    )
    best_state = None
    best_epoch = 0
    best_key = None
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.gradient_clip)
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_tensor).index_select(
                1, common_index_tensor
            )
            validation_probability = torch.sigmoid(validation_logits).cpu().numpy()
        macro = _macro_aupr(validation_common_targets, validation_probability)
        weak_auroc = _target_weak_rank(
            validation_common_targets,
            validation_probability,
            validation_common_weak,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_common_macro_aupr": float(macro),
                "validation_target_weak_auroc": float(weak_auroc),
            }
        )
        weak_key = -float("inf") if not np.isfinite(weak_auroc) else weak_auroc
        key = (float(macro), float(weak_key))
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = int(epoch)
            best_state = {
                key_name: value.detach().cpu().clone()
                for key_name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("head training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_common_macro_aupr": float(best_key[0]),
            "best_target_weak_auroc": float(best_key[1]),
            "pos_weight": pos_weight.detach().cpu(),
        },
        output_dir / "best_head.pth",
    )
    _write_history(output_dir / "history.csv", history)
    return model, {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "best_common_macro_aupr": float(best_key[0]),
        "best_target_weak_auroc": float(best_key[1]),
        "train_seconds": float(time.time() - started),
        "pos_weight": pos_weight.detach().cpu().numpy().tolist(),
    }


def _predict(model, features, device, output_indices=None, batch_size=2048):
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=int(batch_size), shuffle=False)
    chunks = []
    index_tensor = None
    if output_indices is not None:
        index_tensor = torch.as_tensor(output_indices, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            if index_tensor is not None:
                logits = logits.index_select(1, index_tensor)
            chunks.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _per_label_diagnostics(targets, predictions, probabilities, weak_mask):
    truth = np.asarray(targets, dtype=np.uint8)
    pred = np.asarray(predictions, dtype=np.uint8)
    scores = np.asarray(probabilities, dtype=np.float64)
    weak = np.asarray(weak_mask, dtype=bool)
    result = {}
    for position, name in enumerate(COMMON_NAMES):
        positive = truth[:, position] == 1
        negative = ~positive
        weak_positive = positive & weak[:, position]
        weak_target = weak_positive if weak_positive.any() else positive
        result[name] = {
            "aupr": float(average_precision_score(truth[:, position], scores[:, position])),
            "recall": float(pred[positive, position].mean()) if positive.any() else float("nan"),
            "weak_recall": (
                float(pred[weak_target, position].mean())
                if weak_target.any()
                else float("nan")
            ),
            "negative_fpr": (
                float(pred[negative, position].mean()) if negative.any() else float("nan")
            ),
            "positive_count": int(positive.sum()),
            "weak_positive_count": int(weak_positive.sum()),
            "negative_count": int(negative.sum()),
        }
    return result


def _evaluate_variant(
    variant,
    logits_by_role,
    common_targets_by_role,
    common_weak_by_role,
    saw_by_role,
    calibration_targets,
    output_dir,
    args,
):
    calibration = fit_temperature(
        logits_by_role["calibration"],
        calibration_targets,
        source_role="calibration",
    )
    probabilities = {
        role: _sigmoid(apply_temperature(logits, calibration.temperature))
        for role, logits in logits_by_role.items()
    }
    threshold_selection = select_per_label_weak_recall_fpr_thresholds(
        common_targets_by_role["threshold"],
        probabilities["threshold"],
        weak_component_mask=common_weak_by_role["threshold"],
        max_negative_fpr=float(args.max_negative_fpr),
        source_role="threshold_selection",
    )
    thresholds = threshold_selection.thresholds
    early_targets = common_targets_by_role["early"]
    early_probability = probabilities["early"]
    early_prediction = apply_thresholds(early_probability, thresholds)
    metrics = compute_multilabel_metrics(
        early_targets,
        early_prediction,
        probabilities=early_probability,
        class_names=COMMON_NAMES,
        weak_component_mask=common_weak_by_role["early"],
    )
    diagnostics = _per_label_diagnostics(
        early_targets,
        early_prediction,
        early_probability,
        common_weak_by_role["early"],
    )
    metrics["per_label_diagnostics"] = diagnostics
    metrics["target_mean_weak_recall"] = float(
        np.mean([diagnostics[name]["weak_recall"] for name in TARGET_NAMES])
    )
    metrics["target_mean_negative_fpr"] = float(
        np.mean([diagnostics[name]["negative_fpr"] for name in TARGET_NAMES])
    )
    metrics["target_weak_vs_negative_auroc"] = _target_weak_rank(
        early_targets,
        early_probability,
        common_weak_by_role["early"],
    )
    metrics["subgroups"] = {}
    saw_early = np.asarray(saw_by_role["early"], dtype=bool)
    for subgroup, mask in (
        ("sawfm_present", saw_early),
        ("sawfm_absent", ~saw_early),
    ):
        subgroup_metrics = compute_multilabel_metrics(
            early_targets[mask],
            early_prediction[mask],
            probabilities=early_probability[mask],
            class_names=COMMON_NAMES,
            weak_component_mask=common_weak_by_role["early"][mask],
        )
        metrics["subgroups"][subgroup] = subgroup_metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "calibration.json",
        {
            "variant": variant,
            "temperature": calibration.temperature,
            "loss_before": calibration.loss_before,
            "loss_after": calibration.loss_after,
            "source_role": "calibration",
        },
    )
    _write_json(
        output_dir / "thresholds.json",
        {
            "thresholds": thresholds,
            "source_role": "threshold_selection",
            "max_negative_fpr": float(args.max_negative_fpr),
        },
    )
    _write_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(
        str(output_dir / "predictions.npz"),
        sample_ids=np.asarray(saw_by_role["early_sample_ids"], dtype=np.int64),
        logits=logits_by_role["early"],
        probabilities=early_probability,
        predictions=early_prediction,
        targets=early_targets,
        weak_mask=common_weak_by_role["early"].astype(np.uint8),
        sawfm_present=saw_early.astype(np.uint8),
    )
    return {
        "variant": variant,
        "temperature": float(calibration.temperature),
        "thresholds": [float(value) for value in thresholds],
        "metrics": metrics,
    }


def _role_views(extracted, lengths):
    result = {}
    start = 0
    for role, length in lengths:
        end = start + int(length)
        result[role] = {
            key: np.asarray(value[start:end]) for key, value in extracted.items()
        }
        start = end
    if start != len(extracted["features"]):
        raise AssertionError("feature role slicing did not consume all rows")
    return result


def _run_seed(args, config, labels6, weak6, seed, device, data_sha256):
    seed_dir = Path(args.output_root) / "seed_{}".format(seed)
    seed_summary_path = seed_dir / "seed_summary.json"
    if args.resume and seed_summary_path.is_file():
        return _read_json(seed_summary_path)

    eligible = np.arange(len(labels6), dtype=np.int64)
    split = build_stratified_roles(labels6, eligible, seed=int(seed))
    manifest = build_label_longtail(
        labels6,
        split.optimization,
        target_label_counts=LABEL_LT_TARGET_COUNTS,
        seed=int(seed),
        tolerance=0.0,
        source_split_sha256=split.sha256,
        target_sample_count=2400,
        sample_count_tolerance=0.0,
    )
    selected = manifest.selected_indices.copy()
    rng = np.random.RandomState(int(seed) + 31000)
    order = np.arange(len(selected), dtype=np.int64)
    rng.shuffle(order)
    fit_count = int(round(len(order) * 0.8))
    fit_ids = selected[order[:fit_count]]
    validation_ids = selected[order[fit_count:]]
    early_ids = split.early_stop.copy()
    calibration_ids = split.calibration.copy()
    threshold_ids = split.threshold_selection.copy()

    if args.smoke:
        fit_ids = fit_ids[: int(args.smoke_train_samples)]
        validation_ids = validation_ids[: int(args.smoke_eval_samples)]
        early_ids = early_ids[: int(args.smoke_eval_samples)]
        calibration_ids = calibration_ids[: int(args.smoke_eval_samples)]
        threshold_ids = threshold_ids[: int(args.smoke_eval_samples)]

    all_ids = np.concatenate(
        [fit_ids, validation_ids, early_ids, calibration_ids, threshold_ids]
    )
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError("H5 data roles overlap")

    f2_run_dir = (
        Path(args.f2_root) / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    )
    backbone, checkpoint_path = _load_backbone(
        f2_run_dir, F2_MODEL_NAME, device
    )
    features, extracted_targets, sample_ids = _extract_features(
        backbone,
        _data_path(config),
        all_ids,
        str(config["normalization"]),
        device,
        int(args.feature_batch_size),
    )
    if not np.array_equal(sample_ids, all_ids):
        raise AssertionError("feature extraction changed sample order")
    extracted = {
        "features": features,
        "targets6": extracted_targets,
        "sample_ids": sample_ids,
    }
    role_lengths = [
        ("fit", len(fit_ids)),
        ("validation", len(validation_ids)),
        ("early", len(early_ids)),
        ("calibration", len(calibration_ids)),
        ("threshold", len(threshold_ids)),
    ]
    roles = _role_views(extracted, role_lengths)
    for role in roles:
        ids = roles[role]["sample_ids"]
        roles[role]["common_targets"] = roles[role]["targets6"][:, COMMON_INDICES]
        roles[role]["common_weak"] = weak6[ids][:, COMMON_INDICES]
        roles[role]["sawfm_present"] = roles[role]["targets6"][:, 1].astype(bool)

    c6, h5 = _paired_heads(features.shape[1], int(seed) + 50000)
    c6, c6_train = _train_head(
        c6,
        roles["fit"]["features"],
        roles["fit"]["targets6"],
        roles["validation"]["features"],
        roles["validation"]["targets6"],
        roles["validation"]["common_targets"],
        roles["validation"]["common_weak"],
        COMMON_INDICES,
        int(seed) + 51000,
        device,
        args,
        seed_dir / "C6_head",
    )
    h5, h5_train = _train_head(
        h5,
        roles["fit"]["features"],
        roles["fit"]["common_targets"],
        roles["validation"]["features"],
        roles["validation"]["common_targets"],
        roles["validation"]["common_targets"],
        roles["validation"]["common_weak"],
        np.arange(5, dtype=np.int64),
        int(seed) + 51000,
        device,
        args,
        seed_dir / "H5_head",
    )

    c6_logits6 = {
        role: _predict(c6, roles[role]["features"], device)
        for role in ("early", "calibration", "threshold")
    }
    c6_common_logits = {
        role: values[:, COMMON_INDICES] for role, values in c6_logits6.items()
    }
    h5_logits = {
        role: _predict(h5, roles[role]["features"], device)
        for role in ("early", "calibration", "threshold")
    }
    common_targets_by_role = {
        role: roles[role]["common_targets"]
        for role in ("early", "calibration", "threshold")
    }
    common_weak_by_role = {
        role: roles[role]["common_weak"]
        for role in ("early", "calibration", "threshold")
    }
    saw_by_role = {
        role: roles[role]["sawfm_present"]
        for role in ("early", "calibration", "threshold")
    }
    saw_by_role["early_sample_ids"] = roles["early"]["sample_ids"]

    variants = []
    # C6-AllCal fits its temperature on all six logits/targets, then evaluates
    # only the common-five logits.  It cannot go through _evaluate_variant,
    # whose calibration contract is intentionally common-five only.
    all_calibration = fit_temperature(
        c6_logits6["calibration"],
        roles["calibration"]["targets6"],
        source_role="calibration",
    )
    allcal_probabilities = {
        role: _sigmoid(apply_temperature(values, all_calibration.temperature))
        for role, values in c6_common_logits.items()
    }
    allcal_threshold = select_per_label_weak_recall_fpr_thresholds(
        common_targets_by_role["threshold"],
        allcal_probabilities["threshold"],
        weak_component_mask=common_weak_by_role["threshold"],
        max_negative_fpr=float(args.max_negative_fpr),
        source_role="threshold_selection",
    )
    early_prediction = apply_thresholds(
        allcal_probabilities["early"], allcal_threshold.thresholds
    )
    allcal_metrics = compute_multilabel_metrics(
        common_targets_by_role["early"],
        early_prediction,
        probabilities=allcal_probabilities["early"],
        class_names=COMMON_NAMES,
        weak_component_mask=common_weak_by_role["early"],
    )
    diagnostics = _per_label_diagnostics(
        common_targets_by_role["early"],
        early_prediction,
        allcal_probabilities["early"],
        common_weak_by_role["early"],
    )
    allcal_metrics["per_label_diagnostics"] = diagnostics
    allcal_metrics["target_mean_weak_recall"] = float(
        np.mean([diagnostics[name]["weak_recall"] for name in TARGET_NAMES])
    )
    allcal_metrics["target_mean_negative_fpr"] = float(
        np.mean([diagnostics[name]["negative_fpr"] for name in TARGET_NAMES])
    )
    allcal_metrics["target_weak_vs_negative_auroc"] = _target_weak_rank(
        common_targets_by_role["early"],
        allcal_probabilities["early"],
        common_weak_by_role["early"],
    )
    allcal_metrics["subgroups"] = {}
    saw_early = saw_by_role["early"]
    for subgroup, mask in (("sawfm_present", saw_early), ("sawfm_absent", ~saw_early)):
        allcal_metrics["subgroups"][subgroup] = compute_multilabel_metrics(
            common_targets_by_role["early"][mask],
            early_prediction[mask],
            probabilities=allcal_probabilities["early"][mask],
            class_names=COMMON_NAMES,
            weak_component_mask=common_weak_by_role["early"][mask],
        )
    c6_all_dir = seed_dir / "C6_AllCal"
    _write_json(
        c6_all_dir / "calibration.json",
        {
            "variant": "C6_AllCal",
            "temperature": all_calibration.temperature,
            "loss_before": all_calibration.loss_before,
            "loss_after": all_calibration.loss_after,
            "source_role": "calibration",
            "calibration_labels": list(CLASS_NAMES6),
        },
    )
    _write_json(
        c6_all_dir / "thresholds.json",
        {
            "thresholds": allcal_threshold.thresholds,
            "source_role": "threshold_selection",
            "max_negative_fpr": float(args.max_negative_fpr),
        },
    )
    _write_json(c6_all_dir / "metrics.json", allcal_metrics)
    np.savez_compressed(
        str(c6_all_dir / "predictions.npz"),
        sample_ids=roles["early"]["sample_ids"],
        logits=c6_common_logits["early"],
        probabilities=allcal_probabilities["early"],
        predictions=early_prediction,
        targets=common_targets_by_role["early"],
        weak_mask=common_weak_by_role["early"].astype(np.uint8),
        sawfm_present=saw_early.astype(np.uint8),
    )
    variants.append({
        "variant": "C6_AllCal",
        "temperature": float(all_calibration.temperature),
        "thresholds": [float(value) for value in allcal_threshold.thresholds],
        "metrics": allcal_metrics,
    })
    variants.append(
        _evaluate_variant(
            "C6_CommonCal",
            c6_common_logits,
            common_targets_by_role,
            common_weak_by_role,
            saw_by_role,
            common_targets_by_role["calibration"],
            seed_dir / "C6_CommonCal",
            args,
        )
    )
    variants.append(
        _evaluate_variant(
            "H5_CommonCal",
            h5_logits,
            common_targets_by_role,
            common_weak_by_role,
            saw_by_role,
            common_targets_by_role["calibration"],
            seed_dir / "H5_CommonCal",
            args,
        )
    )

    _write_json(seed_dir / "longtail_manifest.json", manifest.to_dict())
    np.savez_compressed(
        str(seed_dir / "role_indices.npz"),
        manifest=selected,
        fit=fit_ids,
        validation=validation_ids,
        early=early_ids,
        calibration=calibration_ids,
        threshold=threshold_ids,
    )
    seed_summary = {
        "seed": int(seed),
        "status": "completed",
        "smoke": bool(args.smoke),
        "data_sha256": data_sha256,
        "split_sha256": split.sha256,
        "manifest_sha256": manifest.sha256,
        "role_indices_sha256": _indices_sha256(
            fit_ids, validation_ids, early_ids, calibration_ids, threshold_ids
        ),
        "f2_checkpoint": str(checkpoint_path),
        "f2_checkpoint_sha256": _sha256(checkpoint_path),
        "head_training": {"C6": c6_train, "H5": h5_train},
        "variants": variants,
        "role_counts": {
            "manifest": int(len(selected)),
            "fit": int(len(fit_ids)),
            "validation": int(len(validation_ids)),
            "early": int(len(early_ids)),
            "calibration": int(len(calibration_ids)),
            "threshold": int(len(threshold_ids)),
        },
        "locked_test_used": False,
        "test_v2_used": False,
    }
    _write_json(seed_summary_path, seed_summary)
    return seed_summary


def _metric(row, path):
    value = row
    for key in path:
        value = value[key]
    return float(value)


def _variant_map(seed_row):
    return {row["variant"]: row for row in seed_row["variants"]}


def _aggregate_variants(seed_rows):
    metric_paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "hamming_loss": ("metrics", "hamming_loss"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "target_weak_vs_negative_auroc": (
            "metrics",
            "target_weak_vs_negative_auroc",
        ),
        "sawfm_present_macro_f1": (
            "metrics",
            "subgroups",
            "sawfm_present",
            "macro_f1",
        ),
        "sawfm_absent_macro_f1": (
            "metrics",
            "subgroups",
            "sawfm_absent",
            "macro_f1",
        ),
    }
    aggregate = {}
    for variant in VARIANTS:
        rows = [_variant_map(seed_row)[variant] for seed_row in seed_rows]
        aggregate[variant] = {
            name: _mean_ci(
                [_metric(row, path) for row in rows], seed=20260731 + offset
            )
            for offset, (name, path) in enumerate(metric_paths.items())
        }
        aggregate[variant]["per_label_aupr"] = {}
        for label_offset, label_name in enumerate(COMMON_NAMES):
            aggregate[variant]["per_label_aupr"][label_name] = _mean_ci(
                [
                    float(
                        row["metrics"]["per_label_diagnostics"][label_name][
                            "aupr"
                        ]
                    )
                    for row in rows
                ],
                seed=20260800 + label_offset,
            )
    return aggregate


def _comparison(seed_rows, candidate, baseline):
    metric_paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "sawfm_present_macro_f1": (
            "metrics",
            "subgroups",
            "sawfm_present",
            "macro_f1",
        ),
        "sawfm_absent_macro_f1": (
            "metrics",
            "subgroups",
            "sawfm_absent",
            "macro_f1",
        ),
    }
    per_seed = []
    for seed_row in seed_rows:
        variants = _variant_map(seed_row)
        candidate_row = variants[candidate]
        baseline_row = variants[baseline]
        delta = {
            name: _metric(candidate_row, path) - _metric(baseline_row, path)
            for name, path in metric_paths.items()
        }
        delta["per_label_aupr"] = {
            label_name: float(
                candidate_row["metrics"]["per_label_diagnostics"][label_name][
                    "aupr"
                ]
                - baseline_row["metrics"]["per_label_diagnostics"][label_name][
                    "aupr"
                ]
            )
            for label_name in COMMON_NAMES
        }
        per_seed.append({"seed": seed_row["seed"], "delta": delta})
    aggregate = {
        name: _mean_ci(
            [row["delta"][name] for row in per_seed], seed=20260900 + offset
        )
        for offset, name in enumerate(metric_paths)
    }
    aggregate["per_label_aupr"] = {
        label_name: _mean_ci(
            [row["delta"]["per_label_aupr"][label_name] for row in per_seed],
            seed=20261000 + offset,
        )
        for offset, label_name in enumerate(COMMON_NAMES)
    }
    aggregate["positive_target_weak_recall_seeds"] = int(
        sum(row["delta"]["target_mean_weak_recall"] > 0.0 for row in per_seed)
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def _promotion_gate(h5_comparison, variant_aggregate):
    delta = h5_comparison["aggregate"]
    h5 = variant_aggregate["H5_CommonCal"]
    checks = {
        "macro_aupr_noninferior_minus_0p005": delta["macro_aupr"]["mean"]
        >= -0.005,
        "macro_f1_gain_0p010": delta["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_0p030": delta[
            "target_mean_weak_recall"
        ]["mean"]
        >= 0.030,
        "target_label_aupr_noninferior_minus_0p010": all(
            delta["per_label_aupr"][name]["mean"] >= -0.010
            for name in TARGET_NAMES
        ),
        "target_mean_fpr_0p20": h5["target_mean_negative_fpr"]["mean"]
        <= 0.20,
        "positive_target_weak_recall_at_least_four": delta[
            "positive_target_weak_recall_seeds"
        ]
        >= 4,
        "sawfm_present_macro_f1_positive": delta[
            "sawfm_present_macro_f1"
        ]["mean"]
        > 0.0,
        "sawfm_absent_macro_f1_noninferior_minus_0p005": delta[
            "sawfm_absent_macro_f1"
        ]["mean"]
        >= -0.005,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _validate_config(config):
    if bool(config.get("allow_locked_test", False)):
        raise ValueError("development-v2 H5 requires allow_locked_test=false")
    if str(config.get("evidence_role")) != "development_v2":
        raise ValueError("config evidence_role must be development_v2")
    if not bool(config.get("test_v2_sealed", False)):
        raise ValueError("test_v2 must be declared sealed")


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha256 = _sha256(data_path)
    labels6 = read_labels(data_path)
    values79, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(
        labels6, _component_power(values79), float(args.weak_threshold_db)
    )
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    effective_epochs = int(args.smoke_epochs) if args.smoke else int(args.epochs)
    args.epochs = effective_epochs
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_rows = [
        _run_seed(
            args,
            config,
            labels6,
            weak6,
            int(seed),
            device,
            data_sha256,
        )
        for seed in args.seeds
    ]
    variant_aggregate = _aggregate_variants(seed_rows)
    calibration_comparison = _comparison(
        seed_rows, "C6_CommonCal", "C6_AllCal"
    )
    h5_comparison = _comparison(seed_rows, "H5_CommonCal", "C6_CommonCal")
    promotion = (
        {"checks": {}, "passed": False, "reason": "smoke_only"}
        if args.smoke
        else _promotion_gate(h5_comparison, variant_aggregate)
    )
    summary = {
        "status": "completed",
        "experiment": "route_a_development_v2_h5",
        "smoke": bool(args.smoke),
        "device": str(device),
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "class_names6": list(CLASS_NAMES6),
            "common_labels": list(COMMON_NAMES),
            "target_labels": list(TARGET_NAMES),
            "label_lt_target_counts_code_order": LABEL_LT_TARGET_COUNTS.tolist(),
            "label_lt_target_sample_count": 2400,
            "max_negative_fpr": float(args.max_negative_fpr),
            "weak_threshold_db": float(args.weak_threshold_db),
            "paired_initialization": True,
            "locked_test_used": False,
            "test_v2_used": False,
        },
        "data": {
            "development_v2_path": str(data_path),
            "development_v2_sha256": data_sha256,
            "sample_count": int(len(labels6)),
            "test_v2_path_declared_sealed": str(config.get("test_v2_path")),
        },
        "per_seed": seed_rows,
        "aggregate": variant_aggregate,
        "comparisons": {
            "calibration_coupling": calibration_comparison,
            "h5_head_effect": h5_comparison,
        },
        "promotion_gate": promotion,
        "next_decision": (
            "preregister_last_stage_h5_unfreeze_with_head_evidence"
            if promotion.get("passed")
            else (
                "smoke_complete_run_formal_five_seed"
                if args.smoke
                else "frozen_head_saw_output_not_sufficient_preregister_representation_gradient_test"
            )
        ),
        "locked_test_used": False,
        "test_v2_used": False,
    }
    summary_path = output_root / "route_a_development_v2_h5_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "student_a"
        / "six_class_development_v2.yaml",
    )
    parser.add_argument(
        "--f2-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v2_h5",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=256)
    parser.add_argument("--smoke-eval-samples", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
