"""Development-v3 unified RawSaw plus VTSE-VBF experiment."""

from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.engine.thresholding import select_per_label_weak_recall_fpr_thresholds  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _component_power,
    _load_signal_array,
)
from experiments.run_route_a_development_v3_label_selective_projection import (  # noqa: E402
    _load_model_from_state,
)
from experiments.run_route_a_stage23_identifiability_v2 import (  # noqa: E402
    TARGET_INDICES,
    TARGET_NAMES,
    _train_probe,
    temporal_statistics,
)
from experiments.run_route_a_visibility_balanced_fusion_v2 import (  # noqa: E402
    VisibilityBalancedFusion,
    _score_normalization,
    _train_fusion,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
SAW_INDEX = 1
VARIANTS = (
    "C0_F2_C6",
    "C1_RawSaw",
    "C2_VTSE_VBF",
    "C3_Unified_RawSaw_VTSE_VBF",
    "C4_SameCapacity_NoVisibility",
)
EXPECTED_V3_SHA256 = "b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47"


class GenericBoundedFusion(nn.Module):
    """Same six trainable scalars as VBF, without a visibility gate."""

    def __init__(self, residual_cap=0.50):
        super().__init__()
        self.residual_cap = float(residual_cap)
        self.expert_weight_raw = nn.Parameter(torch.full((3,), 0.54132485))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, base_logits, normalized_expert, return_residual=False):
        target = base_logits[:, TARGET_INDICES]
        raw = F.softplus(self.expert_weight_raw)[None, :] * normalized_expert + self.bias[None, :]
        target_residual = self.residual_cap * torch.tanh(raw)
        residual = torch.zeros_like(base_logits)
        residual[:, TARGET_INDICES] = target_residual
        logits = base_logits + residual
        return (logits, residual) if return_residual else logits


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config):
    if config.get("evidence_role") != "development_v3":
        raise ValueError("development-v3 config required")
    if config.get("allow_locked_test"):
        raise ValueError("locked test access is forbidden")
    if not config.get("test_v3_sealed") or config.get("test_v3_access_authorized"):
        raise ValueError("test-v3 must be sealed and unauthorized")
    if config.get("test_v3_used"):
        raise ValueError("test-v3 has already been marked used")


def _load_roles(source_root, seed, data_path, normalization, io_batch_size):
    indices_path = source_root / f"seed_{seed}" / "role_indices.npz"
    saved = np.load(indices_path)
    names = ("fit", "validation", "early", "calibration", "threshold")
    ids = {name: saved[name].astype(np.int64) for name in names}
    all_ids = np.concatenate([ids[name] for name in names])
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError("development-v3 roles overlap")
    extracted = _load_signal_array(data_path, all_ids, normalization, io_batch_size)
    result, start = {}, 0
    for name in names:
        end = start + len(ids[name])
        result[name] = {
            "signals": extracted["signals"][start:end],
            "targets6": extracted["targets6"][start:end],
            "sample_ids": extracted["sample_ids"][start:end],
        }
        start = end
    return result, indices_path


def _extract_backbone(model, roles, device, batch_size, topk):
    output = {}
    model.eval()
    for role, values in roles.items():
        signals = values["signals"]
        base_logits, temporal_stats = [], []
        with torch.no_grad():
            for start in range(0, len(signals), int(batch_size)):
                batch = torch.from_numpy(signals[start : start + int(batch_size)]).to(device)
                maps = model.forward_structure_maps(batch)
                logits = model.forward_classifier(maps["global"])
                stats = temporal_statistics(maps["temporal"].float(), topk)
                base_logits.append(logits.cpu().numpy().astype(np.float32))
                temporal_stats.append(stats.cpu().numpy().astype(np.float32))
        output[role] = {
            "base_logits": np.concatenate(base_logits),
            "temporal_stats": np.concatenate(temporal_stats),
            "targets6": values["targets6"],
            "sample_ids": values["sample_ids"],
        }
    return output


def _raw_forward(model, signals, indices, device):
    batch = torch.from_numpy(signals[indices]).to(device)
    return model(batch[:, 0:1], batch[:, 1:2])


def _raw_scores(model, signals, device, batch_size):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), len(signals)))
            values.append(_raw_forward(model, signals, indices, device).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _weak_binary_rank(targets, weak, scores):
    weak_positive = (targets == 1) & weak
    negative = targets == 0
    y = np.concatenate([np.ones(weak_positive.sum()), np.zeros(negative.sum())])
    p = np.concatenate([scores[weak_positive], scores[negative]])
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _train_raw(roles, weak6, seed, device, args):
    train_seed = 881000 + int(seed)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    model = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=args.raw_hidden_channels,
        topk=args.raw_topk, dropout=args.raw_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.raw_learning_rate, weight_decay=args.weight_decay
    )
    fit_targets = roles["fit"]["targets6"][:, SAW_INDEX].astype(np.float32)
    positives = max(float(fit_targets.sum()), 1.0)
    pos_weight = torch.tensor(
        (len(fit_targets) - positives) / positives, dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(train_seed)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.raw_epochs
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        order = rng.permutation(len(fit_targets))
        for start in range(0, len(order), int(args.raw_batch_size)):
            ids = order[start : start + int(args.raw_batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = _raw_forward(model, roles["fit"]["signals"], ids, device)
            target = torch.from_numpy(fit_targets[ids]).to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = _raw_scores(model, roles["validation"]["signals"], device, args.raw_batch_size)
        target = roles["validation"]["targets6"][:, SAW_INDEX]
        weak = weak6[roles["validation"]["sample_ids"], SAW_INDEX]
        aupr = float(average_precision_score(target, score))
        weak_auroc = _weak_binary_rank(target, weak, score)
        key = (aupr, -float("inf") if not np.isfinite(weak_auroc) else weak_auroc)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_sawfm_aupr": aupr,
                        "validation_sawfm_weak_auroc": weak_auroc})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.raw_patience):
                break
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "train_seed": train_seed,
                            "validation_sawfm_aupr": best_key[0],
                            "validation_sawfm_weak_auroc": best_key[1]}


def _train_all_positive_probe(fit_x, validation_x, fit_targets6, validation_targets6,
                              seed, device, args):
    train_seed = 891000 + int(seed)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    model = nn.Linear(fit_x.shape[1], 3).to(device)
    y_fit = fit_targets6[:, TARGET_INDICES].astype(np.float32)
    y_validation = validation_targets6[:, TARGET_INDICES].astype(np.uint8)
    positives = y_fit.sum(axis=0)
    negatives = len(y_fit) - positives
    pos_weight = torch.as_tensor(negatives / np.maximum(positives, 1.0), device=device)
    dataset = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(y_fit))
    loader = DataLoader(dataset, batch_size=args.probe_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_learning_rate,
                                  weight_decay=args.weight_decay)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state, best_value, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.probe_epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device)); y = y.to(device)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): score = model(validation_tensor).cpu().numpy()
        value = float(np.mean([
            roc_auc_score(y_validation[:, i], score[:, i]) for i in range(3)
        ]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_target_macro_auroc": value})
        if value > best_value + 1e-8:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.probe_patience):
                break
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "train_seed": train_seed,
                            "validation_target_macro_auroc": best_value}


def _probe_scores(model, x, device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float32)


def _standardize(fit, roles):
    mean = fit.mean(axis=0).astype(np.float32)
    std = fit.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return {role: ((value - mean) / std).astype(np.float32) for role, value in roles.items()}, mean, std


def _train_generic_fusion(base, expert, targets, fit, validation, seed, device, args):
    train_seed = 901000 + int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = GenericBoundedFusion(args.residual_cap).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.fusion_learning_rate,
                                  weight_decay=args.weight_decay)
    base_t = torch.from_numpy(base).to(device)
    expert_t = torch.from_numpy(expert).to(device)
    target_t = torch.from_numpy(targets.astype(np.float32)).to(device)
    fit_t = torch.as_tensor(fit, dtype=torch.long, device=device)
    validation_t = torch.as_tensor(validation, dtype=torch.long, device=device)
    best_state, best_value, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.fusion_epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, residual = model(base_t[fit_t], expert_t[fit_t], return_residual=True)
        loss = F.binary_cross_entropy_with_logits(
            logits[:, TARGET_INDICES], target_t[fit_t][:, TARGET_INDICES]
        ) + args.residual_l2_weight * residual.pow(2).mean() + args.bias_l2_weight * model.bias.pow(2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step(); model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(base_t[validation_t], expert_t[validation_t])).cpu().numpy()
        value = float(np.mean([
            average_precision_score(targets[validation, index], score[:, index])
            for index in TARGET_INDICES
        ]))
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()),
                        "validation_target_macro_aupr": value})
        if value > best_value + 1e-8:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.fusion_patience): break
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "train_seed": train_seed,
                            "validation_target_macro_aupr": best_value}


def _apply_fusion(model, base, expert, device):
    model.eval()
    with torch.no_grad():
        logits, residual = model(torch.from_numpy(base).to(device),
                                 torch.from_numpy(expert).to(device), return_residual=True)
    return logits.cpu().numpy().astype(np.float32), residual.cpu().numpy().astype(np.float32)


def _fit_platt(logits, targets):
    models = []
    for index in range(logits.shape[1]):
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
        model.fit(logits[:, index:index + 1], targets[:, index])
        models.append(model)
    return models


def _apply_platt(models, logits):
    return np.column_stack([
        model.predict_proba(logits[:, index:index + 1])[:, 1]
        for index, model in enumerate(models)
    ])


def _platt_payload(models):
    return [{"coefficient": float(model.coef_[0, 0]),
             "intercept": float(model.intercept_[0])} for model in models]


def _per_label_metrics(targets, probability, prediction, weak):
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        positive = targets[:, index] == 1
        negative = ~positive
        weak_positive = positive & weak[:, index]
        strong_positive = positive & ~weak[:, index]
        result[name] = {
            "aupr": float(average_precision_score(targets[:, index], probability[:, index])),
            "recall": float(prediction[positive, index].mean()),
            "weak_recall": float(prediction[weak_positive, index].mean()) if weak_positive.any() else float("nan"),
            "strong_recall": float(prediction[strong_positive, index].mean()) if strong_positive.any() else float("nan"),
            "negative_fpr": float(prediction[negative, index].mean()),
            "weak_vs_negative_auroc": _weak_binary_rank(
                targets[:, index], weak[:, index], probability[:, index]
            ),
            "positive_count": int(positive.sum()),
            "weak_positive_count": int(weak_positive.sum()),
            "negative_count": int(negative.sum()),
        }
    return result


def _evaluate(variant, logits, role_data, weak6, output_dir, args):
    calibrator = _fit_platt(logits["calibration"], role_data["calibration"]["targets6"])
    probabilities = {role: _apply_platt(calibrator, value) for role, value in logits.items()}
    threshold_weak = weak6[role_data["threshold"]["sample_ids"]]
    selection = select_per_label_weak_recall_fpr_thresholds(
        role_data["threshold"]["targets6"], probabilities["threshold"],
        weak_component_mask=threshold_weak,
        max_negative_fpr=args.max_negative_fpr,
        source_role="threshold_selection",
    )
    thresholds = np.asarray(selection.thresholds)
    targets = role_data["early"]["targets6"]
    weak = weak6[role_data["early"]["sample_ids"]]
    probability = probabilities["early"]
    prediction = probability >= thresholds[None, :]
    per_label = _per_label_metrics(targets, probability, prediction, weak)
    target = [per_label[name] for name in TARGET_NAMES]
    saw_present = targets[:, SAW_INDEX].astype(bool)
    subgroup = {}
    for name, mask in (("sawfm_present", saw_present), ("sawfm_absent", ~saw_present)):
        subgroup[name] = {
            "macro_f1": float(f1_score(targets[mask], prediction[mask], average="macro", zero_division=0)),
            "macro_aupr": float(average_precision_score(targets[mask], probability[mask], average="macro")),
        }
    metrics = {
        "macro_aupr": float(average_precision_score(targets, probability, average="macro")),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, prediction)),
        "per_label": per_label,
        "target_mean_weak_recall": float(np.mean([row["weak_recall"] for row in target])),
        "target_mean_strong_recall": float(np.mean([row["strong_recall"] for row in target])),
        "target_mean_negative_fpr": float(np.mean([row["negative_fpr"] for row in target])),
        "target_weak_vs_negative_auroc": float(np.mean([row["weak_vs_negative_auroc"] for row in target])),
        "subgroups": subgroup,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "calibration.json", {"source_role": "calibration",
                                                     "method": "per_label_platt",
                                                     "parameters": _platt_payload(calibrator)})
    _write_json(output_dir / "thresholds.json", {"source_role": "threshold_selection",
                                                   "max_negative_fpr": args.max_negative_fpr,
                                                   "thresholds": thresholds.tolist()})
    _write_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(output_dir / "predictions.npz",
                        sample_ids=role_data["early"]["sample_ids"], targets=targets,
                        weak_mask=weak.astype(np.uint8), logits=logits["early"],
                        probabilities=probability, predictions=prediction.astype(np.uint8))
    return {"variant": variant, "metrics": metrics,
            "thresholds": thresholds.tolist(), "calibration": _platt_payload(calibrator)}


def _variant_logits(base_by_role, raw_by_role, visibility_by_role, generic_by_role):
    variants = {}
    for variant in VARIANTS:
        values = {}
        for role, base in base_by_role.items():
            logits = base.copy()
            if variant in {"C1_RawSaw", "C3_Unified_RawSaw_VTSE_VBF", "C4_SameCapacity_NoVisibility"}:
                logits[:, SAW_INDEX] = raw_by_role[role]
            if variant in {"C2_VTSE_VBF", "C3_Unified_RawSaw_VTSE_VBF"}:
                logits[:, TARGET_INDICES] = visibility_by_role[role][:, TARGET_INDICES]
            if variant == "C4_SameCapacity_NoVisibility":
                logits[:, TARGET_INDICES] = generic_by_role[role][:, TARGET_INDICES]
            values[role] = logits
        variants[variant] = values
    return variants


def _run_seed(args, config, labels6, weak6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    roles, role_indices_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    c6_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    checkpoint = torch.load(c6_path, map_location="cpu", weights_only=False)
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    model = _load_model_from_state(f2_dir, device, checkpoint["model_state_dict"])
    base_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    extracted = _extract_backbone(model, roles, device, args.feature_batch_size, args.topk)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()

    raw, raw_history, raw_info = _train_raw(roles, weak6, seed, device, args)
    raw_scores = {role: _raw_scores(raw, values["signals"], device, args.raw_batch_size)
                  for role, values in roles.items()}

    stats = {role: value["temporal_stats"] for role, value in extracted.items()}
    standardized, stats_mean, stats_std = _standardize(stats["fit"], stats)
    probe_args = copy.copy(args)
    probe_args.batch_size = int(args.probe_batch_size)
    probe_args.learning_rate = float(args.probe_learning_rate)
    probe_args.epochs = int(args.smoke_epochs if args.smoke else args.probe_epochs)
    probe_args.patience = int(args.probe_patience)
    vtse, vtse_epoch, vtse_seed, vtse_history, vtse_counts = _train_probe(
        standardized["fit"], standardized["validation"],
        roles["fit"]["targets6"], roles["validation"]["targets6"],
        weak6[roles["fit"]["sample_ids"]], weak6[roles["validation"]["sample_ids"]],
        seed, "T_FinalTemporal", device, probe_args,
    )
    generic_probe, generic_history, generic_info = _train_all_positive_probe(
        standardized["fit"], standardized["validation"],
        roles["fit"]["targets6"], roles["validation"]["targets6"],
        seed, device, args,
    )
    vtse_scores = {role: _probe_scores(vtse, value, device) for role, value in standardized.items()}
    generic_scores = {role: _probe_scores(generic_probe, value, device) for role, value in standardized.items()}

    combined_roles = ("fit", "validation")
    base_fitval = np.concatenate([extracted[r]["base_logits"] for r in combined_roles])
    vtse_fitval = np.concatenate([vtse_scores[r] for r in combined_roles])
    targets_fitval = np.concatenate([roles[r]["targets6"] for r in combined_roles])
    ids_fitval = np.concatenate([roles[r]["sample_ids"] for r in combined_roles])
    fit_positions = np.arange(len(roles["fit"]["sample_ids"]), dtype=np.int64)
    validation_positions = np.arange(len(fit_positions), len(ids_fitval), dtype=np.int64)
    center, scale, orientation, normalization_diagnostics = _score_normalization(
        vtse_fitval[fit_positions], targets_fitval[fit_positions], weak6[ids_fitval][fit_positions]
    )
    fusion_arrays = {"base_logits": base_fitval, "expert_scores": vtse_fitval,
                     "targets": targets_fitval}
    fusion_args = copy.copy(args)
    fusion_args.learning_rate = float(args.fusion_learning_rate)
    fusion_args.epochs = int(args.smoke_epochs if args.smoke else args.fusion_epochs)
    fusion_args.patience = int(args.fusion_patience)
    fusion, fusion_epoch, fusion_seed, fusion_history = _train_fusion(
        fusion_arrays, weak6, ids_fitval, fit_positions, validation_positions,
        center, scale, orientation, seed, device, fusion_args,
    )
    vtse_normalized = {
        role: ((scores * orientation[None, :] - center[None, :]) / scale[None, :]).astype(np.float32)
        for role, scores in vtse_scores.items()
    }
    visibility_logits = {
        role: _apply_fusion(fusion, extracted[role]["base_logits"], value, device)[0]
        for role, value in vtse_normalized.items()
    }
    generic_fitval = np.concatenate([generic_scores[r] for r in combined_roles])
    generic_center = generic_fitval[fit_positions].mean(axis=0).astype(np.float32)
    generic_scale = np.maximum(generic_fitval[fit_positions].std(axis=0), 1e-6).astype(np.float32)
    generic_normalized = {
        role: ((scores - generic_center[None, :]) / generic_scale[None, :]).astype(np.float32)
        for role, scores in generic_scores.items()
    }
    generic_fusion, generic_fusion_history, generic_fusion_info = _train_generic_fusion(
        base_fitval, np.concatenate([generic_normalized[r] for r in combined_roles]),
        targets_fitval, fit_positions, validation_positions, seed, device, args,
    )
    generic_logits = {
        role: _apply_fusion(generic_fusion, extracted[role]["base_logits"], value, device)[0]
        for role, value in generic_normalized.items()
    }
    base_logits = {role: value["base_logits"] for role, value in extracted.items()}
    logits = _variant_logits(base_logits, raw_scores, visibility_logits, generic_logits)

    seed_root.mkdir(parents=True, exist_ok=True)
    checkpoints = seed_root / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    torch.save({"model_state_dict": raw.state_dict(), **raw_info}, checkpoints / "rawsaw.pth")
    torch.save({"model_state_dict": vtse.state_dict(), "mean": stats_mean, "std": stats_std,
                "best_epoch": vtse_epoch, "train_seed": vtse_seed}, checkpoints / "vtse.pth")
    torch.save({"model_state_dict": fusion.state_dict(), "center": center, "scale": scale,
                "orientation": orientation, "best_epoch": fusion_epoch,
                "train_seed": fusion_seed, "residual_cap": args.residual_cap,
                "uncertainty_power": args.uncertainty_power}, checkpoints / "vbf.pth")
    torch.save({"model_state_dict": generic_probe.state_dict(), "mean": stats_mean, "std": stats_std,
                **generic_info}, checkpoints / "capacity_probe.pth")
    torch.save({"model_state_dict": generic_fusion.state_dict(),
                "center": generic_center, "scale": generic_scale,
                **generic_fusion_info}, checkpoints / "capacity_fusion.pth")
    _write_history(seed_root / "rawsaw_history.csv", raw_history)
    _write_history(seed_root / "vtse_history.csv", vtse_history)
    _write_history(seed_root / "vbf_history.csv", fusion_history)
    _write_history(seed_root / "capacity_probe_history.csv", generic_history)
    _write_history(seed_root / "capacity_fusion_history.csv", generic_fusion_history)

    results = [_evaluate(variant, logits[variant], roles, weak6,
                         seed_root / variant, args) for variant in VARIANTS]
    parameter_counts = {
        "base_model": base_parameter_count,
        "rawsaw": int(sum(p.numel() for p in raw.parameters())),
        "vtse": int(sum(p.numel() for p in vtse.parameters())),
        "vbf": int(sum(p.numel() for p in fusion.parameters())),
        "capacity_probe": int(sum(p.numel() for p in generic_probe.parameters())),
        "capacity_fusion": int(sum(p.numel() for p in generic_fusion.parameters())),
    }
    parameter_counts["c3_addition"] = parameter_counts["rawsaw"] + parameter_counts["vtse"] + parameter_counts["vbf"]
    parameter_counts["c4_addition"] = parameter_counts["rawsaw"] + parameter_counts["capacity_probe"] + parameter_counts["capacity_fusion"]
    summary = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
        "data_sha256": data_sha, "role_indices_path": str(role_indices_path.resolve()),
        "role_indices_sha256": _sha256(role_indices_path),
        "c6_checkpoint_sha256": _sha256(c6_path),
        "parameter_counts": parameter_counts,
        "same_capacity_exact": parameter_counts["c3_addition"] == parameter_counts["c4_addition"],
        "training": {"rawsaw": raw_info, "vtse": {"best_epoch": vtse_epoch,
                     "train_seed": vtse_seed, "counts": vtse_counts},
                     "vbf": {"best_epoch": fusion_epoch, "train_seed": fusion_seed,
                     "normalization": normalization_diagnostics},
                     "capacity_probe": generic_info, "capacity_fusion": generic_fusion_info},
        "variants": results, "identity_audit_passed": True,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, summary)
    return summary


def _variant_map(seed_row):
    return {row["variant"]: row for row in seed_row["variants"]}


def _metric(row, path):
    value = row
    for key in path: value = value[key]
    return float(value)


def _aggregate(seed_rows):
    paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "hamming_loss": ("metrics", "hamming_loss"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_strong_recall": ("metrics", "target_mean_strong_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "target_weak_vs_negative_auroc": ("metrics", "target_weak_vs_negative_auroc"),
        "sawfm_aupr": ("metrics", "per_label", "SawFM", "aupr"),
        "sawfm_weak_recall": ("metrics", "per_label", "SawFM", "weak_recall"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        rows = [_variant_map(seed)[variant] for seed in seed_rows]
        aggregate[variant] = {name: _mean_ci([_metric(row, path) for row in rows],
                                               20263100 + vi * 100 + i)
                              for i, (name, path) in enumerate(paths.items())}
        aggregate[variant]["per_label_aupr"] = {
            name: _mean_ci([row["metrics"]["per_label"][name]["aupr"] for row in rows],
                           20263600 + vi * 10 + i)
            for i, name in enumerate(CLASS_NAMES)
        }
    return aggregate


def _compare(seed_rows, candidate, baseline):
    metrics = ("macro_aupr", "macro_f1", "target_mean_weak_recall",
               "target_mean_strong_recall", "target_mean_negative_fpr",
               "target_weak_vs_negative_auroc", "sawfm_aupr", "sawfm_weak_recall",
               "sawfm_present_macro_f1", "sawfm_absent_macro_f1")
    paired = []
    for seed_row in seed_rows:
        c, b = _variant_map(seed_row)[candidate]["metrics"], _variant_map(seed_row)[baseline]["metrics"]
        flat = {
            "macro_aupr": c["macro_aupr"] - b["macro_aupr"],
            "macro_f1": c["macro_f1"] - b["macro_f1"],
            "target_mean_weak_recall": c["target_mean_weak_recall"] - b["target_mean_weak_recall"],
            "target_mean_strong_recall": c["target_mean_strong_recall"] - b["target_mean_strong_recall"],
            "target_mean_negative_fpr": c["target_mean_negative_fpr"] - b["target_mean_negative_fpr"],
            "target_weak_vs_negative_auroc": c["target_weak_vs_negative_auroc"] - b["target_weak_vs_negative_auroc"],
            "sawfm_aupr": c["per_label"]["SawFM"]["aupr"] - b["per_label"]["SawFM"]["aupr"],
            "sawfm_weak_recall": c["per_label"]["SawFM"]["weak_recall"] - b["per_label"]["SawFM"]["weak_recall"],
            "sawfm_present_macro_f1": c["subgroups"]["sawfm_present"]["macro_f1"] - b["subgroups"]["sawfm_present"]["macro_f1"],
            "sawfm_absent_macro_f1": c["subgroups"]["sawfm_absent"]["macro_f1"] - b["subgroups"]["sawfm_absent"]["macro_f1"],
        }
        paired.append({"seed": seed_row["seed"], **flat})
    aggregate = {name: _mean_ci([row[name] for row in paired], 20264000 + i)
                 for i, name in enumerate(metrics)}
    aggregate["positive_target_weak_recall_seeds"] = int(sum(
        row["target_mean_weak_recall"] > 0 for row in paired
    ))
    return {"candidate": candidate, "baseline": baseline,
            "per_seed": paired, "aggregate": aggregate}


def _gates(aggregate, comparisons, seed_rows):
    c3 = aggregate["C3_Unified_RawSaw_VTSE_VBF"]
    vs_c0 = comparisons["C3_vs_C0"]["aggregate"]
    vs_c1 = comparisons["C3_vs_C1"]["aggregate"]
    vs_c2 = comparisons["C3_vs_C2"]["aggregate"]
    vs_c4 = comparisons["C3_vs_C4"]["aggregate"]
    per_target_noninferior = all(
        c3["per_label_aupr"][name]["mean"] >= aggregate["C0_F2_C6"]["per_label_aupr"][name]["mean"] - 0.010
        for name in TARGET_NAMES
    )
    checks = {
        "macro_aupr_gain_0p005_vs_c0": vs_c0["macro_aupr"]["mean"] >= 0.005,
        "macro_f1_gain_0p010_vs_c0": vs_c0["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_0p030_vs_c0": vs_c0["target_mean_weak_recall"]["mean"] >= 0.030,
        "target_weak_auroc_gain_0p015_vs_c0": vs_c0["target_weak_vs_negative_auroc"]["mean"] >= 0.015,
        "target_fpr_cap_0p20": c3["target_mean_negative_fpr"]["mean"] <= 0.20,
        "target_strong_recall_noninferior": vs_c0["target_mean_strong_recall"]["mean"] >= -0.010,
        "four_of_five_positive_target_weak_gain": vs_c0["positive_target_weak_recall_seeds"] >= 4,
        "target_label_aupr_noninferior": per_target_noninferior,
        "sawfm_aupr_preserved_vs_c1": vs_c1["sawfm_aupr"]["mean"] >= -1e-8,
        "sawfm_weak_recall_preserved_vs_c1": vs_c1["sawfm_weak_recall"]["mean"] >= -1e-8,
        "target_weak_recall_preserved_vs_c2": vs_c2["target_mean_weak_recall"]["mean"] >= -1e-8,
        "target_weak_auroc_preserved_vs_c2": vs_c2["target_weak_vs_negative_auroc"]["mean"] >= -1e-8,
        "better_macro_aupr_than_same_capacity": vs_c4["macro_aupr"]["mean"] > 0.0,
        "better_target_weak_recall_than_same_capacity": vs_c4["target_mean_weak_recall"]["mean"] > 0.0,
        "sawfm_present_noninferior": vs_c0["sawfm_present_macro_f1"]["mean"] >= -0.005,
        "sawfm_absent_noninferior": vs_c0["sawfm_absent_macro_f1"]["mean"] >= -0.005,
        "same_capacity_exact": all(row["same_capacity_exact"] for row in seed_rows),
        "identity_and_test_boundary_safe": all(
            row["identity_audit_passed"] and not row["locked_test_used"]
            and not row["test_v2_used"] and not row["test_v3_used"] for row in seed_rows
        ),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA256 mismatch")
    sealed = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if sealed.get("status") != "sealed" or sealed["test_v3"].get("access_authorized"):
        raise ValueError("test-v3 seal is missing or access is authorized")
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameter_values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_rows = [_run_seed(args, config, labels6, weak6, seed, device, data_sha)
                 for seed in args.seeds]
    aggregate = _aggregate(seed_rows)
    comparisons = {
        "C3_vs_C0": _compare(seed_rows, "C3_Unified_RawSaw_VTSE_VBF", "C0_F2_C6"),
        "C3_vs_C1": _compare(seed_rows, "C3_Unified_RawSaw_VTSE_VBF", "C1_RawSaw"),
        "C3_vs_C2": _compare(seed_rows, "C3_Unified_RawSaw_VTSE_VBF", "C2_VTSE_VBF"),
        "C3_vs_C4": _compare(seed_rows, "C3_Unified_RawSaw_VTSE_VBF", "C4_SameCapacity_NoVisibility"),
    }
    gates = {"checks": {}, "passed": False} if args.smoke else _gates(aggregate, comparisons, seed_rows)
    payload = {
        "status": "completed", "experiment": "RouteA_Unified_RawSaw_VTSE_VBF_v3",
        "smoke": bool(args.smoke), "seeds": list(args.seeds),
        "data_path": str(data_path), "data_sha256": data_sha,
        "test_v3_manifest": str(args.test_v3_manifest.resolve()),
        "aggregate": aggregate, "comparisons": comparisons,
        "promotion_gates": gates, "promoted": bool(gates["passed"]),
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_unified_rawsaw_vtse_vbf_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--raw-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--raw-epochs", type=int, default=30)
    parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument("--fusion-epochs", type=int, default=200)
    parser.add_argument("--raw-patience", type=int, default=5)
    parser.add_argument("--probe-patience", type=int, default=8)
    parser.add_argument("--fusion-patience", type=int, default=20)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--residual-cap", type=float, default=0.50)
    parser.add_argument("--uncertainty-power", type=float, default=0.5)
    parser.add_argument("--residual-l2-weight", type=float, default=0.02)
    parser.add_argument("--bias-l2-weight", type=float, default=0.05)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
