"""Train six-scalar visibility-balanced fusion on frozen C6 and VTSE scores."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _component_power, _load_backbone  # noqa: E402
from experiments.run_route_a_h5_background_ablation_v2 import _sha256, _write  # noqa: E402
from experiments.run_route_a_bounded_local_contrast_v2 import _load_frozen_c6  # noqa: E402
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES  # noqa: E402
from experiments.run_route_a_visibility_temporal_expert_v2 import (  # noqa: E402
    F2_MODEL_NAME, VARIANTS, _evaluate, _extract, _load_probe, _score_normalization,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


class VisibilityBalancedFusion(nn.Module):
    def __init__(self, residual_cap=0.50, uncertainty_power=0.5):
        super().__init__()
        self.residual_cap = float(residual_cap)
        self.uncertainty_power = float(uncertainty_power)
        self.expert_weight_raw = nn.Parameter(torch.full((3,), 0.54132485))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, base_logits, normalized_expert, return_residual=False):
        target_base = base_logits[:, TARGET_INDICES]
        probability = torch.sigmoid(target_base)
        uncertainty = torch.pow(4.0 * probability * (1.0 - probability),
                                self.uncertainty_power)
        raw = F.softplus(self.expert_weight_raw)[None, :] * normalized_expert + self.bias[None, :]
        target_residual = self.residual_cap * torch.tanh(raw) * uncertainty
        residual = torch.zeros_like(base_logits)
        residual[:, TARGET_INDICES] = target_residual
        logits = base_logits + residual
        return (logits, residual) if return_residual else logits


def _visibility_balanced_loss(logits, targets6, weak6, residual, bias,
                              residual_weight, bias_weight):
    losses = []
    for local, index in enumerate(TARGET_INDICES):
        target = targets6[:, index]
        weak = (target > 0.5) & weak6[:, index].bool()
        strong = (target > 0.5) & ~weak6[:, index].bool()
        negative = target < 0.5
        strata = []
        for mask in (weak, strong, negative):
            if mask.any():
                strata.append(F.binary_cross_entropy_with_logits(
                    logits[mask, index], target[mask]))
        losses.append(torch.stack(strata).mean())
    return (torch.stack(losses).mean()
            + float(residual_weight) * residual.pow(2).mean()
            + float(bias_weight) * bias.pow(2).mean())


def _validation_key(logits, targets6, weak6):
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    targets = targets6.detach().cpu().numpy().astype(np.uint8)
    weak = weak6.detach().cpu().numpy().astype(bool)
    aupr, weak_auroc = [], []
    for index in TARGET_INDICES:
        aupr.append(average_precision_score(targets[:, index], probabilities[:, index]))
        weak_positive = (targets[:, index] == 1) & weak[:, index]
        negative = targets[:, index] == 0
        y = np.concatenate([np.ones(weak_positive.sum()), np.zeros(negative.sum())])
        p = np.concatenate([probabilities[weak_positive, index], probabilities[negative, index]])
        weak_auroc.append(roc_auc_score(y, p))
    return float(np.mean(aupr)), float(np.mean(weak_auroc))


def _train_fusion(arrays, weak6, optimization_ids, fit, validation,
                  center, scale, orientation, seed, device, args):
    train_seed = 161000 + seed
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = VisibilityBalancedFusion(args.residual_cap, args.uncertainty_power).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    base = torch.from_numpy(arrays["base_logits"]).to(device)
    score = torch.from_numpy(
        ((arrays["expert_scores"] * orientation[None, :] - center[None, :]) /
         scale[None, :]).astype(np.float32)).to(device)
    targets = torch.from_numpy(arrays["targets"].astype(np.float32)).to(device)
    weak = torch.from_numpy(weak6[optimization_ids].astype(np.uint8)).to(device)
    fit_tensor = torch.as_tensor(fit, dtype=torch.long, device=device)
    validation_tensor = torch.as_tensor(validation, dtype=torch.long, device=device)
    best_state, best_epoch, best_key, stale, history = None, 0, (-float("inf"), -float("inf")), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, residual = model(base[fit_tensor], score[fit_tensor], return_residual=True)
        loss = _visibility_balanced_loss(logits, targets[fit_tensor], weak[fit_tensor],
                                         residual, model.bias, args.residual_l2_weight,
                                         args.bias_l2_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step(); model.eval()
        with torch.no_grad(): validation_logits = model(base[validation_tensor], score[validation_tensor])
        key = _validation_key(validation_logits, targets[validation_tensor], weak[validation_tensor])
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()),
                        "validation_target_macro_aupr": key[0],
                        "validation_target_weak_auroc": key[1],
                        "expert_weights": F.softplus(model.expert_weight_raw).detach().cpu().tolist(),
                        "bias": model.bias.detach().cpu().tolist()})
        improved = key[0] > best_key[0] + 1e-8 or (
            np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8)
        if improved:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("fusion did not produce checkpoint")
    model.load_state_dict(best_state); return model, best_epoch, train_seed, history


def _apply(model, arrays, center, scale, orientation, device):
    base = torch.from_numpy(arrays["base_logits"]).to(device)
    score = torch.from_numpy(
        ((arrays["expert_scores"] * orientation[None, :] - center[None, :]) /
         scale[None, :]).astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad(): logits, residual = model(base, score, return_residual=True)
    return logits.cpu().numpy(), residual.cpu().numpy()


def _run_seed(args, config, labels6, weak6, seed, device):
    plan = build_run_plan(config, "visibility_balanced_fusion", seed, 0, args.max_eval_samples)
    optimization = np.asarray(plan["train_indices"], dtype=np.int64)
    rng = np.random.default_rng(133000 + seed); order = rng.permutation(len(optimization))
    cut = int(round(0.8 * len(order))); fit, validation = order[:cut], order[cut:]
    role_ids = {"optimization": optimization,
                "early": np.asarray(plan["early_stop_indices"], dtype=np.int64),
                "calibration": np.asarray(plan["calibration_indices"], dtype=np.int64),
                "threshold": np.asarray(plan["threshold_indices"], dtype=np.int64)}
    base_head, base_checkpoint, base_path = _load_frozen_c6(seed, args.h5_root, device)
    probe, probe_checkpoint, probe_path = _load_probe(seed, args.stage23_root, device)
    backbone_dir = args.screen_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    backbone, backbone_path = _load_backbone(backbone_dir, F2_MODEL_NAME, device)
    arrays = {role: _extract(backbone, base_head, probe,
                             base_checkpoint["feature_mean"], base_checkpoint["feature_std"],
                             probe_checkpoint["mean"], probe_checkpoint["std"],
                             _data_path(config), ids, config["normalization"], device,
                             args.feature_batch_size, args.topk)
              for role, ids in role_ids.items()}
    del backbone, base_head, probe
    center, scale, orientation, diagnostics = _score_normalization(
        arrays["optimization"]["expert_scores"][fit],
        arrays["optimization"]["targets"][fit], weak6[optimization][fit])
    model, best_epoch, train_seed, history = _train_fusion(
        arrays["optimization"], weak6, optimization, fit, validation,
        center, scale, orientation, seed, device, args)
    seed_root = args.output_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "center": center, "scale": scale,
                "orientation": orientation, "best_epoch": best_epoch,
                "train_seed": train_seed, "residual_cap": args.residual_cap,
                "uncertainty_power": args.uncertainty_power}, seed_root / "best_fusion.pth")
    with (seed_root / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    _write(seed_root / "score_normalization.json", {"center": center, "scale": scale,
                                                      "orientation": orientation,
                                                      "diagnostics": diagnostics,
                                                      "source_role": "optimization_fit"})
    rows = []
    for variant in VARIANTS:
        if variant == "B0_FrozenC6":
            logits = {role: arrays[role]["base_logits"] for role in ("early", "calibration", "threshold")}
            residual = {role: np.zeros_like(value) for role, value in logits.items()}
        else:
            converted = {role: _apply(model, arrays[role], center, scale, orientation, device)
                         for role in ("early", "calibration", "threshold")}
            logits = {role: value[0] for role, value in converted.items()}
            residual = {role: value[1] for role, value in converted.items()}
        out = seed_root / variant; out.mkdir(parents=True, exist_ok=True)
        result = _evaluate(variant, logits, residual, arrays, role_ids, weak6, out, args)
        result.update({"seed": seed, "best_epoch": best_epoch if variant != "B0_FrozenC6" else 0,
                       "train_seed": train_seed if variant != "B0_FrozenC6" else None,
                       "split_sha256": plan["split_sha256"]})
        _write(out / "variant_summary.json", result); rows.append(result)
    _write(seed_root / "run_context.json", {"seed": seed, "split_sha256": plan["split_sha256"],
                                              "backbone_sha256": _sha256(backbone_path),
                                              "base_head_sha256": _sha256(base_path),
                                              "probe_sha256": _sha256(probe_path),
                                              "development_v2_used": True,
                                              "test_v2_used": False, "locked_test_used": False})
    return rows


def _aggregate(rows):
    metrics = ["macro_aupr", "macro_f1", "target_weak_recall", "target_weak_auroc",
               "target_negative_fpr", "target_strong_recall", "saw_present_macro_f1",
               "saw_absent_macro_f1", "residual_abs_mean", "residual_abs_max"]
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {metric: _mean_ci([row[metric] for row in selected],
                                                171000 + vi * 100 + mi)
                              for mi, metric in enumerate(metrics)}
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        c = next(row for row in rows if row["seed"] == seed and row["variant"] == "E1_VTSE_Bounded")
        b = next(row for row in rows if row["seed"] == seed and row["variant"] == "B0_FrozenC6")
        paired.append({"seed": seed, **{metric: c[metric] - b[metric] for metric in metrics}})
    mean = {metric: float(np.mean([row[metric] for row in paired])) for metric in metrics}
    candidate = aggregate["E1_VTSE_Bounded"]
    gates = {"macro_aupr_noninferior": mean["macro_aupr"] >= -0.005,
             "macro_f1_gain": mean["macro_f1"] >= 0.010,
             "target_weak_recall_gain": mean["target_weak_recall"] >= 0.030,
             "target_weak_auroc_gain": mean["target_weak_auroc"] >= 0.015,
             "target_fpr_cap": candidate["target_negative_fpr"]["mean"] <= 0.20,
             "strong_recall_noninferior": mean["target_strong_recall"] >= -0.010,
             "four_of_five_positive_weak_gain": sum(row["target_weak_recall"] > 0 for row in paired) >= 4,
             "saw_present_noninferior": mean["saw_present_macro_f1"] >= -0.005,
             "residual_hard_bound": candidate["residual_abs_max"]["mean"] <= 0.500001}
    return aggregate, paired, mean, gates


def run(args):
    config = load_config(args.config); data_path = _data_path(config)
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for seed in args.seeds:
        print(f"visibility-balanced-fusion development-v2 seed={seed}", flush=True)
        rows.extend(_run_seed(args, config, labels6, weak6, seed, device))
    aggregate, paired, mean, gates = _aggregate(rows)
    payload = {"status": "completed", "experiment": "RouteA_VisibilityBalancedFusion_v2",
               "seeds": args.seeds, "data_path": str(data_path), "data_sha256": _sha256(data_path),
               "aggregate": aggregate, "paired_candidate_minus_c6": paired,
               "mean_deltas": mean, "promotion_gates": gates,
               "promoted": bool(all(gates.values())),
               "next_action": ("freeze_candidate_then_run_test_v2_once" if all(gates.values())
                               else "retain_visibility_innovation_and_revisit_threshold_utility"),
               "development_v2_used": True, "test_v2_used": False, "locked_test_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_visibility_balanced_fusion_v2_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--h5-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_h5_background_ablation_v2")
    parser.add_argument("--stage23-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stage23_identifiability_v2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--max-eval-samples", type=int, default=1200)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--residual-cap", type=float, default=0.50)
    parser.add_argument("--uncertainty-power", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--residual-l2-weight", type=float, default=0.02)
    parser.add_argument("--bias-l2-weight", type=float, default=0.05)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())

