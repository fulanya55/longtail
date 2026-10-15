"""Aligned-vs-shuffled predicted-visibility fusion on development-v3."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_presence_visibility_aux_v3 import (  # noqa: E402
    AUX_VARIANT, BASE_VARIANT, LINEAR_VARIANT, PresenceVisibilityHead, _head_scores,
)
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256, _apply_fusion, _evaluate, _extract_backbone, _load_roles,
    _raw_scores, _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


SHUFFLED_VARIANT = "E0_ShuffledPredictedVisibility"
ALIGNED_VARIANT = "E1_AlignedPredictedVisibility"
VARIANTS = (BASE_VARIANT, LINEAR_VARIANT, AUX_VARIANT, SHUFFLED_VARIANT, ALIGNED_VARIANT)


class PredictedVisibilityConditionedFusion(nn.Module):
    """Twelve-scalar bounded residual controlled by predicted visibility only."""

    def __init__(self, residual_cap=0.50):
        super().__init__()
        self.residual_cap = float(residual_cap)
        self.presence_weight_raw = nn.Parameter(torch.full((3,), 0.54132485))
        self.severity_weight = nn.Parameter(torch.zeros(3))
        self.interaction_weight = nn.Parameter(torch.zeros(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, base_logits, presence, severity, return_residual=False):
        raw = (F.softplus(self.presence_weight_raw)[None, :] * presence
               + self.severity_weight[None, :] * severity
               + self.interaction_weight[None, :] * presence * severity
               + self.bias[None, :])
        target_residual = self.residual_cap * torch.tanh(raw)
        residual = torch.zeros_like(base_logits)
        residual[:, TARGET_INDICES] = target_residual
        logits = base_logits + residual
        return (logits, residual) if return_residual else logits


def _predicted_visibility_severity(predicted_gap_normalized, center_db=-10.0,
                                   temperature_db=2.5, gap_clip_db=40.0):
    """Map predicted normalized relative-power gap to fixed weak-visibility severity."""
    center = float(center_db) / float(gap_clip_db)
    temperature = float(temperature_db) / float(gap_clip_db)
    if temperature <= 0:
        raise ValueError("temperature_db must be positive")
    z = (center - np.asarray(predicted_gap_normalized, dtype=np.float64)) / temperature
    return (1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))).astype(np.float32)


def _standardize_triplet(fit_presence, fit_severity, presence_by_role, severity_by_role):
    p_center = fit_presence.mean(axis=0).astype(np.float32)
    p_scale = np.maximum(fit_presence.std(axis=0), 1e-6).astype(np.float32)
    s_center = fit_severity.mean(axis=0).astype(np.float32)
    s_scale = np.maximum(fit_severity.std(axis=0), 1e-6).astype(np.float32)
    p = {role: ((value - p_center[None, :]) / p_scale[None, :]).astype(np.float32)
         for role, value in presence_by_role.items()}
    s = {role: ((value - s_center[None, :]) / s_scale[None, :]).astype(np.float32)
         for role, value in severity_by_role.items()}
    return p, s, {"presence_center": p_center.tolist(), "presence_scale": p_scale.tolist(),
                  "severity_center": s_center.tolist(), "severity_scale": s_scale.tolist()}


def _shuffle_by_role(values_by_role, seed):
    shuffled, receipt = {}, {}
    for offset, (role, values) in enumerate(values_by_role.items()):
        rng = np.random.default_rng(951000 + int(seed) * 10 + offset)
        permutation = rng.permutation(len(values)).astype(np.int64)
        shuffled[role] = values[permutation].copy()
        digest = hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest()
        receipt[role] = {"length": int(len(permutation)), "permutation_sha256": digest,
                         "fixed_seed": int(951000 + int(seed) * 10 + offset)}
    return shuffled, receipt


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _train_fusion(base, presence, severity, targets, fit, validation, seed, device, args):
    train_seed = 961000 + int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = PredictedVisibilityConditionedFusion(args.residual_cap).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.fusion_learning_rate,
                                  weight_decay=args.weight_decay)
    base_t = torch.from_numpy(base).to(device)
    presence_t = torch.from_numpy(presence).to(device)
    severity_t = torch.from_numpy(severity).to(device)
    target_t = torch.from_numpy(targets.astype(np.float32)).to(device)
    fit_t = torch.as_tensor(fit, dtype=torch.long, device=device)
    validation_t = torch.as_tensor(validation, dtype=torch.long, device=device)
    best_state, best_value, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.fusion_epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, residual = model(base_t[fit_t], presence_t[fit_t], severity_t[fit_t],
                                 return_residual=True)
        loss = (F.binary_cross_entropy_with_logits(
                    logits[:, TARGET_INDICES], target_t[fit_t][:, TARGET_INDICES])
                + args.residual_l2_weight * residual.pow(2).mean()
                + args.bias_l2_weight * model.bias.pow(2).mean())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step(); model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(base_t[validation_t], presence_t[validation_t],
                                        severity_t[validation_t])).cpu().numpy()
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


def _apply_conditioned(model, base, presence, severity, device):
    model.eval()
    with torch.no_grad():
        logits, residual = model(torch.from_numpy(base).to(device),
                                 torch.from_numpy(presence).to(device),
                                 torch.from_numpy(severity).to(device),
                                 return_residual=True)
    return logits.cpu().numpy().astype(np.float32), residual.cpu().numpy().astype(np.float32)


def _read_prior(seed, variant, root):
    path = root / f"seed_{seed}" / variant / "metrics.json"
    return {"variant": variant, "metrics": json.loads(path.read_text(encoding="utf-8"))}


def _run_seed(args, config, labels6, weak6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(args.c6_root, seed, _data_path(config),
                                           config["normalization"], args.io_batch_size)
    state = torch.load(args.c6_root/f"seed_{seed}"/"C6_Joint"/"best_model.pth",
                       map_location="cpu", weights_only=False)["model_state_dict"]
    base_model = _load_model_from_state(args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}",
                                        device, state)
    extracted = _extract_backbone(base_model, roles, device, args.feature_batch_size, args.topk)
    del base_model
    if device.type == "cuda": torch.cuda.empty_cache()

    d2_checkpoint_path = args.d2_root/f"seed_{seed}"/f"{AUX_VARIANT}.pth"
    d2 = torch.load(d2_checkpoint_path, map_location="cpu", weights_only=False)
    head = PresenceVisibilityHead(extracted["fit"]["temporal_stats"].shape[1], args.adapter_rank).to(device)
    head.load_state_dict(d2["head_state_dict"], strict=True)
    standardized = {
        role: ((row["temporal_stats"] - d2["mean"][None, :])
               / np.maximum(d2["std"][None, :], 1e-6)).astype(np.float32)
        for role, row in extracted.items()
    }
    presence_raw, severity_raw = {}, {}
    for role, values in standardized.items():
        presence, predicted_gap = _head_scores(head, values, device)
        presence_raw[role] = presence
        severity_raw[role] = _predicted_visibility_severity(
            predicted_gap, args.visibility_center_db, args.visibility_temperature_db,
            args.gap_clip_db,
        )
    presence, severity_aligned, normalization = _standardize_triplet(
        presence_raw["fit"], severity_raw["fit"], presence_raw, severity_raw
    )
    severity_shuffled, shuffle_receipt = _shuffle_by_role(severity_aligned, seed)

    combined_roles = ("fit", "validation")
    base_fitval = np.concatenate([extracted[role]["base_logits"] for role in combined_roles])
    presence_fitval = np.concatenate([presence[role] for role in combined_roles])
    target_fitval = np.concatenate([roles[role]["targets6"] for role in combined_roles])
    fit = np.arange(len(roles["fit"]["sample_ids"]), dtype=np.int64)
    validation = np.arange(len(fit), len(target_fitval), dtype=np.int64)
    variants = {SHUFFLED_VARIANT: severity_shuffled, ALIGNED_VARIANT: severity_aligned}
    fused_logits, histories, fusion_info, residual_diagnostics = {}, {}, {}, {}
    for variant, severity in variants.items():
        severity_fitval = np.concatenate([severity[role] for role in combined_roles])
        fusion, history, info = _train_fusion(
            base_fitval, presence_fitval, severity_fitval, target_fitval,
            fit, validation, seed, device, args,
        )
        histories[variant] = history; fusion_info[variant] = info
        fused_logits[variant], residuals = {}, {}
        for role in roles:
            fused_logits[variant][role], residuals[role] = _apply_conditioned(
                fusion, extracted[role]["base_logits"], presence[role], severity[role], device
            )
        residual_diagnostics[variant] = {
            role: {"mean_abs_target_residual": float(np.abs(value[:, TARGET_INDICES]).mean()),
                   "max_abs_target_residual": float(np.abs(value[:, TARGET_INDICES]).max())}
            for role, value in residuals.items()
        }
        torch.save({"model_state_dict": fusion.state_dict(), "normalization": normalization,
                    "variant": variant, **info}, seed_root/f"{variant}.pth")
        _write_history(seed_root/f"{variant}_history.csv", history)

    raw_state = torch.load(args.unified_root/f"seed_{seed}"/"checkpoints"/"rawsaw.pth",
                           map_location="cpu", weights_only=False)
    raw = RawStage1SawFMFrontend(mode="raw", hidden_channels=args.raw_hidden_channels,
                                 topk=args.raw_topk, dropout=args.raw_dropout).to(device)
    raw.load_state_dict(raw_state["model_state_dict"], strict=True)
    raw_scores = {role: _raw_scores(raw, row["signals"], device, args.raw_batch_size)
                  for role, row in roles.items()}
    results = [_read_prior(seed, BASE_VARIANT, args.unified_root),
               _read_prior(seed, LINEAR_VARIANT, args.unified_root),
               _read_prior(seed, AUX_VARIANT, args.d2_root)]
    for variant in (SHUFFLED_VARIANT, ALIGNED_VARIANT):
        logits = {}
        for role, values in fused_logits[variant].items():
            combined = values.copy(); combined[:, 1] = raw_scores[role]; logits[role] = combined
        results.append(_evaluate(variant, logits, roles, weak6, seed_root/variant, args))
    parameter_count = int(sum(p.numel() for p in PredictedVisibilityConditionedFusion(
        args.residual_cap).parameters()))
    payload = {"seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
               "data_sha256": data_sha, "d2_checkpoint_sha256": _sha256(d2_checkpoint_path),
               "role_indices_sha256": _sha256(role_indices_path), "variants": results,
               "fusion_info": fusion_info, "normalization": normalization,
               "shuffle_receipt": shuffle_receipt, "residual_diagnostics": residual_diagnostics,
               "fusion_parameter_count": parameter_count, "same_capacity": True,
               "inference_uses_true_visibility": False, "locked_test_used": False,
               "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, payload); return payload


def _vmap(row): return {item["variant"]: item["metrics"] for item in row["variants"]}


def _flat(metrics):
    return {"macro_aupr": metrics["macro_aupr"], "macro_f1": metrics["macro_f1"],
            "weak_recall": metrics["target_mean_weak_recall"],
            "weak_auroc": metrics["target_weak_vs_negative_auroc"],
            "strong_recall": metrics["target_mean_strong_recall"],
            "fpr": metrics["target_mean_negative_fpr"],
            "isrj_fpr": metrics["per_label"]["ISRJ"]["negative_fpr"],
            "sawfm_aupr": metrics["per_label"]["SawFM"]["aupr"],
            "sawfm_weak_recall": metrics["per_label"]["SawFM"]["weak_recall"]}


def _aggregate(rows):
    result = {}
    for vi, variant in enumerate(VARIANTS):
        values = [_flat(_vmap(row)[variant]) for row in rows]
        result[variant] = {name: _mean_ci([value[name] for value in values], 20269100+vi*100+i)
                           for i, name in enumerate(values[0])}
    return result


def _compare(rows, baseline):
    paired = []
    for row in rows:
        candidate = _flat(_vmap(row)[ALIGNED_VARIANT]); base = _flat(_vmap(row)[baseline])
        paired.append({"seed": row["seed"], **{name: candidate[name]-base[name] for name in candidate}})
    aggregate = {name: _mean_ci([row[name] for row in paired], 20269700+i)
                 for i, name in enumerate(_flat(_vmap(rows[0])[ALIGNED_VARIANT]))}
    aggregate["positive_weak_recall_seeds"] = int(sum(row["weak_recall"] > 0 for row in paired))
    return {"baseline": baseline, "candidate": ALIGNED_VARIANT,
            "per_seed": paired, "aggregate": aggregate}


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, labels6, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = _aggregate(rows)
    comparisons = {"vs_shuffled": _compare(rows, SHUFFLED_VARIANT),
                   "vs_d2": _compare(rows, AUX_VARIANT),
                   "vs_c4": _compare(rows, LINEAR_VARIANT),
                   "vs_c0": _compare(rows, BASE_VARIANT)}
    if args.smoke: gate = {"checks": {}, "passed": False}
    else:
        aligned = aggregate[ALIGNED_VARIANT]; shuffled = comparisons["vs_shuffled"]["aggregate"]
        c0 = comparisons["vs_c0"]["aggregate"]
        checks = {
            "weak_recall_gain_0p010_vs_shuffled": shuffled["weak_recall"]["mean"] >= .010,
            "weak_auroc_gain_0p010_vs_shuffled": shuffled["weak_auroc"]["mean"] >= .010,
            "macro_aupr_noninferior_vs_shuffled": shuffled["macro_aupr"]["mean"] >= -.003,
            "strong_recall_noninferior_vs_shuffled": shuffled["strong_recall"]["mean"] >= -.010,
            "target_fpr_cap": aligned["fpr"]["mean"] <= .20,
            "weak_recall_gain_0p030_vs_c0": c0["weak_recall"]["mean"] >= .030,
            "weak_auroc_gain_0p015_vs_c0": c0["weak_auroc"]["mean"] >= .015,
            "four_of_five_positive_vs_c0": c0["positive_weak_recall_seeds"] >= 4,
            "sawfm_preserved": abs(comparisons["vs_d2"]["aggregate"]["sawfm_aupr"]["mean"]) <= 1e-8,
            "same_capacity_no_truth_and_test_safe": all(
                row["same_capacity"] and not row["inference_uses_true_visibility"]
                and not row["test_v3_used"] for row in rows),
        }
        gate = {"checks": checks, "passed": bool(all(checks.values()))}
    payload = {"status": "completed", "experiment": "RouteA_PredictedVisibilityConditionedFusion_v3",
               "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "comparisons": comparisons, "promotion_gate": gate,
               "promoted": bool(gate["passed"]), "locked_test_used": False,
               "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root/"route_a_predicted_visibility_conditioned_fusion_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_dfms_iq_structure_screen")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--d2-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_presence_visibility_aux_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_predicted_visibility_conditioned_fusion_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64); parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16); parser.add_argument("--raw-dropout", type=float, default=.1)
    parser.add_argument("--adapter-rank", type=int, default=32); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-2); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fusion-epochs", type=int, default=200); parser.add_argument("--fusion-patience", type=int, default=20)
    parser.add_argument("--gradient-clip", type=float, default=5.0); parser.add_argument("--residual-cap", type=float, default=.50)
    parser.add_argument("--residual-l2-weight", type=float, default=.02); parser.add_argument("--bias-l2-weight", type=float, default=.05)
    parser.add_argument("--max-negative-fpr", type=float, default=.20); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--visibility-center-db", type=float, default=-10.0)
    parser.add_argument("--visibility-temperature-db", type=float, default=2.5)
    parser.add_argument("--gap-clip-db", type=float, default=40.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())

