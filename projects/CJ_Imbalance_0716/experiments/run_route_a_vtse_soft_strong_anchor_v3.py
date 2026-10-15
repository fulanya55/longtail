"""Fixed soft-strong-anchor VTSE validation on development-v3."""

from __future__ import annotations

import argparse
import copy
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

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _component_power,
)
from experiments.run_route_a_development_v3_label_selective_projection import (  # noqa: E402
    _load_model_from_state,
)
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES, TARGET_NAMES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _apply_fusion,
    _evaluate,
    _extract_backbone,
    _load_roles,
    _probe_scores,
    _raw_scores,
    _standardize,
    _validate_config,
    _write_history,
)
from experiments.run_route_a_visibility_balanced_fusion_v2 import (  # noqa: E402
    _score_normalization,
    _train_fusion,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


CANDIDATE = "S1_SoftStrongAnchor_Unified"
HARD_VARIANT = "C3_Unified_RawSaw_VTSE_VBF"
ALL_POSITIVE_VARIANT = "C4_SameCapacity_NoVisibility"
BASE_VARIANT = "C0_F2_C6"


def _continuous_visibility_severity(labels6, component_power6, center_db=-10.0,
                                    temperature_db=2.5):
    """Return a fixed [0, 1] visibility severity for each positive component.

    Single-component positives and negatives receive zero severity.  For a
    component in a compound sample, gap is its power minus the strongest other
    positive component.  The preregistered sigmoid is centred at -10 dB.
    """
    labels = np.asarray(labels6).astype(bool)
    power = np.asarray(component_power6, dtype=np.float64)
    if labels.shape != power.shape:
        raise ValueError("labels and component powers must have the same shape")
    if float(temperature_db) <= 0:
        raise ValueError("temperature_db must be positive")
    severity = np.zeros(labels.shape, dtype=np.float32)
    for row in range(len(labels)):
        positive = np.flatnonzero(labels[row])
        if len(positive) < 2:
            continue
        for index in positive:
            other = positive[positive != index]
            gap = power[row, index] - np.nanmax(power[row, other])
            if np.isfinite(gap):
                z = (float(center_db) - float(gap)) / float(temperature_db)
                severity[row, index] = np.float32(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))
    return severity


def _continuous_visibility_loss(logits, targets3, severity3, pos_weight,
                                maximum_multiplier=2.0):
    """All-positive BCE with fixed continuous upweighting of low-visibility positives."""
    element = F.binary_cross_entropy_with_logits(
        logits, targets3, pos_weight=pos_weight, reduction="none"
    )
    positive_multiplier = 1.0 + (float(maximum_multiplier) - 1.0) * severity3
    sample_weight = torch.where(targets3 > 0.5, positive_multiplier, torch.ones_like(targets3))
    return (element * sample_weight).mean()


def _soft_anchor_loss(logits, targets6, weak6, strong_weight):
    losses = []
    for local, index in enumerate(TARGET_INDICES):
        target = targets6[:, index]
        weak = (target > 0.5) & weak6[:, index].bool()
        strong = (target > 0.5) & ~weak6[:, index].bool()
        negative = target < 0.5
        weak_loss = F.binary_cross_entropy_with_logits(logits[weak, local], target[weak])
        negative_loss = F.binary_cross_entropy_with_logits(logits[negative, local], target[negative])
        strong_loss = F.binary_cross_entropy_with_logits(logits[strong, local], target[strong])
        losses.append((weak_loss + negative_loss + float(strong_weight) * strong_loss)
                      / (2.0 + float(strong_weight)))
    return torch.stack(losses).mean()


def _validation_key(logits, targets6, weak6):
    score = logits.detach().cpu().numpy()
    targets = targets6.detach().cpu().numpy().astype(np.uint8)
    weak = weak6.detach().cpu().numpy().astype(bool)
    weak_auroc, aupr = [], []
    for local, index in enumerate(TARGET_INDICES):
        weak_positive = (targets[:, index] == 1) & weak[:, index]
        negative = targets[:, index] == 0
        y = np.concatenate([np.ones(weak_positive.sum()), np.zeros(negative.sum())])
        p = np.concatenate([score[weak_positive, local], score[negative, local]])
        weak_auroc.append(roc_auc_score(y, p))
        aupr.append(average_precision_score(targets[:, index], score[:, local]))
    return float(np.mean(weak_auroc)), float(np.mean(aupr))


def _train_soft_probe(fit_x, validation_x, fit_targets6, validation_targets6,
                      fit_weak6, validation_weak6, seed, device, args):
    train_seed = 911000 + int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = nn.Linear(fit_x.shape[1], 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_learning_rate,
                                  weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(fit_x),
                            torch.from_numpy(fit_targets6.astype(np.float32)),
                            torch.from_numpy(fit_weak6.astype(np.uint8)))
    loader = DataLoader(dataset, batch_size=args.probe_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    validation_x = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_targets6.astype(np.float32)).to(device)
    validation_weak = torch.from_numpy(validation_weak6.astype(np.uint8)).to(device)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.probe_epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); losses = []
        for x, y, weak in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = _soft_anchor_loss(logits, y.to(device), weak.to(device), args.strong_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): validation_logits = model(validation_x)
        key = _validation_key(validation_logits, validation_targets, validation_weak)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_weak_auroc": key[0], "validation_target_aupr": key[1]})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.probe_patience): break
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "train_seed": train_seed,
                            "validation_weak_auroc": best_key[0],
                            "validation_target_aupr": best_key[1],
                            "strong_weight": float(args.strong_weight)}


def _train_continuous_probe(fit_x, validation_x, fit_targets6, validation_targets6,
                            fit_severity6, validation_weak6, seed, device, args):
    train_seed = 921000 + int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = nn.Linear(fit_x.shape[1], 3).to(device)
    y_fit = fit_targets6[:, TARGET_INDICES].astype(np.float32)
    severity_fit = fit_severity6[:, TARGET_INDICES].astype(np.float32)
    positives = y_fit.sum(axis=0)
    negatives = len(y_fit) - positives
    pos_weight = torch.as_tensor(
        negatives / np.maximum(positives, 1.0), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_learning_rate,
                                  weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(y_fit),
                            torch.from_numpy(severity_fit))
    loader = DataLoader(dataset, batch_size=args.probe_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    validation_x = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_targets6.astype(np.float32)).to(device)
    validation_weak = torch.from_numpy(validation_weak6.astype(np.uint8)).to(device)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.probe_epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); losses = []
        for x, y, severity in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device)); y = y.to(device); severity = severity.to(device)
            loss = _continuous_visibility_loss(
                logits, y, severity, pos_weight, args.maximum_multiplier
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): validation_logits = model(validation_x)
        key = _validation_key(validation_logits, validation_targets, validation_weak)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_weak_auroc": key[0], "validation_target_aupr": key[1]})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.probe_patience): break
    model.load_state_dict(best_state)
    return model, history, {
        "best_epoch": best_epoch, "train_seed": train_seed,
        "validation_weak_auroc": best_key[0], "validation_target_aupr": best_key[1],
        "visibility_center_db": float(args.visibility_center_db),
        "visibility_temperature_db": float(args.visibility_temperature_db),
        "maximum_multiplier": float(args.maximum_multiplier),
        "positive_weight": pos_weight.detach().cpu().numpy().tolist(),
    }


def _read_prior(seed, variant, prior_root):
    path = prior_root / f"seed_{seed}" / variant / "metrics.json"
    return {"variant": variant, "metrics": json.loads(path.read_text(encoding="utf-8"))}


def _run_seed(args, config, labels6, weak6, severity6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    roles, role_indices_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    c6_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(c6_path, map_location="cpu", weights_only=False)["model_state_dict"]
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    base_model = _load_model_from_state(f2_dir, device, state)
    extracted = _extract_backbone(base_model, roles, device, args.feature_batch_size, args.topk)
    del base_model
    if device.type == "cuda": torch.cuda.empty_cache()
    stats = {role: value["temporal_stats"] for role, value in extracted.items()}
    standardized, stats_mean, stats_std = _standardize(stats["fit"], stats)
    if args.training_mode == "continuous":
        probe, probe_history, probe_info = _train_continuous_probe(
            standardized["fit"], standardized["validation"],
            roles["fit"]["targets6"], roles["validation"]["targets6"],
            severity6[roles["fit"]["sample_ids"]], weak6[roles["validation"]["sample_ids"]],
            seed, device, args,
        )
    else:
        probe, probe_history, probe_info = _train_soft_probe(
            standardized["fit"], standardized["validation"],
            roles["fit"]["targets6"], roles["validation"]["targets6"],
            weak6[roles["fit"]["sample_ids"]], weak6[roles["validation"]["sample_ids"]],
            seed, device, args,
        )
    expert_scores = {role: _probe_scores(probe, value, device)
                     for role, value in standardized.items()}
    fitval_roles = ("fit", "validation")
    base_fitval = np.concatenate([extracted[r]["base_logits"] for r in fitval_roles])
    expert_fitval = np.concatenate([expert_scores[r] for r in fitval_roles])
    targets_fitval = np.concatenate([roles[r]["targets6"] for r in fitval_roles])
    ids_fitval = np.concatenate([roles[r]["sample_ids"] for r in fitval_roles])
    fit = np.arange(len(roles["fit"]["sample_ids"]), dtype=np.int64)
    validation = np.arange(len(fit), len(ids_fitval), dtype=np.int64)
    center, scale, orientation, diagnostics = _score_normalization(
        expert_fitval[fit], targets_fitval[fit], weak6[ids_fitval][fit]
    )
    fusion_args = copy.copy(args)
    fusion_args.learning_rate = args.fusion_learning_rate
    fusion_args.epochs = args.smoke_epochs if args.smoke else args.fusion_epochs
    fusion_args.patience = args.fusion_patience
    fusion, fusion_epoch, fusion_seed, fusion_history = _train_fusion(
        {"base_logits": base_fitval, "expert_scores": expert_fitval, "targets": targets_fitval},
        weak6, ids_fitval, fit, validation, center, scale, orientation,
        seed, device, fusion_args,
    )
    normalized = {role: ((score * orientation[None, :] - center[None, :]) / scale[None, :]).astype(np.float32)
                  for role, score in expert_scores.items()}
    target_logits = {role: _apply_fusion(fusion, extracted[role]["base_logits"], value, device)[0]
                     for role, value in normalized.items()}
    raw_path = args.prior_root / f"seed_{seed}" / "checkpoints" / "rawsaw.pth"
    raw_state = torch.load(raw_path, map_location="cpu", weights_only=False)
    raw = RawStage1SawFMFrontend(mode="raw", hidden_channels=args.raw_hidden_channels,
                                 topk=args.raw_topk, dropout=args.raw_dropout).to(device)
    raw.load_state_dict(raw_state["model_state_dict"], strict=True)
    raw_scores = {role: _raw_scores(raw, values["signals"], device, args.raw_batch_size)
                  for role, values in roles.items()}
    candidate_logits = {}
    for role in roles:
        values = target_logits[role].copy()
        values[:, 1] = raw_scores[role]
        candidate_logits[role] = values
    seed_root.mkdir(parents=True, exist_ok=True)
    result = _evaluate(CANDIDATE, candidate_logits, roles, weak6, seed_root / CANDIDATE, args)
    prefix = "continuous_visibility" if args.training_mode == "continuous" else "soft_anchor"
    torch.save({"model_state_dict": probe.state_dict(), "mean": stats_mean, "std": stats_std,
                **probe_info}, seed_root / f"{prefix}_vtse.pth")
    torch.save({"model_state_dict": fusion.state_dict(), "center": center, "scale": scale,
                "orientation": orientation, "best_epoch": fusion_epoch,
                "train_seed": fusion_seed}, seed_root / f"{prefix}_vbf.pth")
    _write_history(seed_root / f"{prefix}_vtse_history.csv", probe_history)
    _write_history(seed_root / f"{prefix}_vbf_history.csv", fusion_history)
    variants = [
        _read_prior(seed, BASE_VARIANT, args.prior_root),
        _read_prior(seed, HARD_VARIANT, args.prior_root),
        _read_prior(seed, ALL_POSITIVE_VARIANT, args.prior_root),
        result,
    ]
    summary = {"seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
               "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
               "probe": probe_info, "fusion": {"best_epoch": fusion_epoch,
               "train_seed": fusion_seed, "normalization": diagnostics},
               "variants": variants, "parameter_count": int(sum(p.numel() for p in probe.parameters())
               + sum(p.numel() for p in fusion.parameters()) + sum(p.numel() for p in raw.parameters())),
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, summary)
    return summary


def _vmap(seed_row):
    return {row["variant"]: row["metrics"] for row in seed_row["variants"]}


def _flat(metrics):
    return {"macro_aupr": metrics["macro_aupr"], "macro_f1": metrics["macro_f1"],
            "target_weak_recall": metrics["target_mean_weak_recall"],
            "target_weak_auroc": metrics["target_weak_vs_negative_auroc"],
            "target_strong_recall": metrics["target_mean_strong_recall"],
            "target_fpr": metrics["target_mean_negative_fpr"],
            "sawfm_aupr": metrics["per_label"]["SawFM"]["aupr"],
            "sawfm_weak_recall": metrics["per_label"]["SawFM"]["weak_recall"]}


def _aggregate(seed_rows):
    variants = (BASE_VARIANT, HARD_VARIANT, ALL_POSITIVE_VARIANT, CANDIDATE)
    result = {}
    for vi, variant in enumerate(variants):
        rows = [_flat(_vmap(seed)[variant]) for seed in seed_rows]
        result[variant] = {name: _mean_ci([row[name] for row in rows], 20265100 + vi * 100 + i)
                           for i, name in enumerate(rows[0])}
    return result


def _compare(seed_rows, baseline):
    paired = []
    for seed in seed_rows:
        candidate = _flat(_vmap(seed)[CANDIDATE]); base = _flat(_vmap(seed)[baseline])
        paired.append({"seed": seed["seed"], **{name: candidate[name] - base[name] for name in candidate}})
    aggregate = {name: _mean_ci([row[name] for row in paired], 20265600 + i)
                 for i, name in enumerate(_flat(_vmap(seed_rows[0])[CANDIDATE]))}
    aggregate["positive_weak_recall_seeds"] = int(sum(row["target_weak_recall"] > 0 for row in paired))
    return {"baseline": baseline, "candidate": CANDIDATE, "per_seed": paired, "aggregate": aggregate}


def run(args):
    global CANDIDATE
    if args.training_mode == "continuous":
        CANDIDATE = "S2_ContinuousVisibility_Unified"
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    sealed = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if sealed["test_v3"]["access_authorized"] or sealed["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    component_power6 = _component_power(values)
    weak6 = compute_weak_component_mask(labels6, component_power6, args.weak_threshold_db)
    severity6 = _continuous_visibility_severity(
        labels6, component_power6, args.visibility_center_db, args.visibility_temperature_db
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_rows = [_run_seed(args, config, labels6, weak6, severity6, seed, device, data_sha)
                 for seed in args.seeds]
    aggregate = _aggregate(seed_rows)
    comparisons = {"vs_c0": _compare(seed_rows, BASE_VARIANT),
                   "vs_hard": _compare(seed_rows, HARD_VARIANT),
                   "vs_all_positive": _compare(seed_rows, ALL_POSITIVE_VARIANT)}
    if args.smoke:
        gates = {"checks": {}, "passed": False}
    else:
        c = aggregate[CANDIDATE]
        c0 = comparisons["vs_c0"]["aggregate"]
        hard = comparisons["vs_hard"]["aggregate"]
        all_positive = comparisons["vs_all_positive"]["aggregate"]
        checks = {
            "better_weak_recall_than_hard": hard["target_weak_recall"]["mean"] > 0,
            "better_weak_auroc_than_hard": hard["target_weak_auroc"]["mean"] > 0,
            "weak_recall_noninferior_to_all_positive": all_positive["target_weak_recall"]["mean"] >= 0,
            "weak_auroc_noninferior_to_all_positive": all_positive["target_weak_auroc"]["mean"] >= 0,
            "strong_recall_noninferior_to_all_positive": all_positive["target_strong_recall"]["mean"] >= -0.010,
            "target_fpr_cap": c["target_fpr"]["mean"] <= 0.20,
            "target_weak_gain_0p030_vs_c0": c0["target_weak_recall"]["mean"] >= 0.030,
            "target_auroc_gain_0p015_vs_c0": c0["target_weak_auroc"]["mean"] >= 0.015,
            "four_of_five_positive_vs_c0": c0["positive_weak_recall_seeds"] >= 4,
            "sawfm_preserved": abs(comparisons["vs_hard"]["aggregate"]["sawfm_aupr"]["mean"]) <= 1e-8,
            "identity_and_test_safe": all(not row["locked_test_used"] and not row["test_v2_used"]
                                           and not row["test_v3_used"] for row in seed_rows),
        }
        gates = {"checks": checks, "passed": bool(all(checks.values()))}
    experiment = ("RouteA_VTSE_ContinuousVisibility_v3" if args.training_mode == "continuous"
                  else "RouteA_VTSE_SoftStrongAnchor_v3")
    payload = {"status": "completed", "experiment": experiment,
               "smoke": bool(args.smoke), "seeds": list(args.seeds), "strong_weight": args.strong_weight,
               "data_sha256": data_sha, "aggregate": aggregate, "comparisons": comparisons,
               "promotion_gates": gates, "promoted": bool(gates["passed"]),
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_name = ("route_a_vtse_continuous_visibility_v3_summary.json"
                    if args.training_mode == "continuous"
                    else "route_a_vtse_soft_strong_anchor_v3_summary.json")
    _write_json(args.output_root / summary_name, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--prior-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_vtse_soft_strong_anchor_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--training-mode", choices=("soft", "continuous"), default="soft")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64); parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--raw-hidden-channels", type=int, default=16); parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3); parser.add_argument("--fusion-learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument("--fusion-epochs", type=int, default=200); parser.add_argument("--probe-patience", type=int, default=8)
    parser.add_argument("--fusion-patience", type=int, default=20); parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--strong-weight", type=float, default=0.20); parser.add_argument("--residual-cap", type=float, default=0.50)
    parser.add_argument("--visibility-center-db", type=float, default=-10.0)
    parser.add_argument("--visibility-temperature-db", type=float, default=2.5)
    parser.add_argument("--maximum-multiplier", type=float, default=2.0)
    parser.add_argument("--uncertainty-power", type=float, default=0.5); parser.add_argument("--residual-l2-weight", type=float, default=0.02)
    parser.add_argument("--bias-l2-weight", type=float, default=0.05); parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
