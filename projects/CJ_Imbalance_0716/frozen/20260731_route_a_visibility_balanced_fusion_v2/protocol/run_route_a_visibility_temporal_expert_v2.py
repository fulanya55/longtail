"""Visibility Temporal Statistics Expert (VTSE) bounded reconnection."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.thresholding import select_per_label_weak_recall_fpr_thresholds  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _component_power, _load_backbone  # noqa: E402
from experiments.run_route_a_h5_background_ablation_v2 import (  # noqa: E402
    ALL_NAMES, COMMON_INDICES, COMMON_NAMES, TARGET_LOCAL,
    _metrics, _sha256, _sigmoid, _write,
)
from experiments.run_route_a_bounded_local_contrast_v2 import (  # noqa: E402
    _load_frozen_c6, _target_weak_auroc,
)
from experiments.run_route_a_stage23_identifiability_v2 import (  # noqa: E402
    TARGET_INDICES, TARGET_NAMES, temporal_statistics,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


F2_MODEL_NAME = "F2_DFMS_Full"
VARIANTS = ("B0_FrozenC6", "E1_VTSE_Bounded")


def _load_probe(seed, root, device):
    path = root / f"seed_{seed}" / "T_FinalTemporal" / "best_probe.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = nn.Linear(512, 3)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, checkpoint, path


def _extract(backbone, base_head, probe, base_mean, base_std, probe_mean,
             probe_std, data_path, indices, normalization, device, batch_size, topk):
    dataset = V73SignalDataset(data_path, indices, "six", normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    base_logits, expert_scores, targets, sample_ids = [], [], [], []
    base_mean = torch.as_tensor(base_mean, dtype=torch.float32, device=device)
    base_std = torch.as_tensor(base_std, dtype=torch.float32, device=device)
    probe_mean = torch.as_tensor(probe_mean, dtype=torch.float32, device=device)
    probe_std = torch.as_tensor(probe_std, dtype=torch.float32, device=device)
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type,
                                        enabled=device.type == "cuda"):
                    maps = backbone.forward_structure_maps(inputs)
                global_feature = maps["global"].float()
                summary = temporal_statistics(maps["temporal"].float(), topk)
                base_logits.append(base_head((global_feature - base_mean) / base_std).cpu().numpy())
                expert_scores.append(probe((summary - probe_mean) / probe_std).cpu().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {"base_logits": np.concatenate(base_logits).astype(np.float32),
            "expert_scores": np.concatenate(expert_scores).astype(np.float32),
            "targets": np.concatenate(targets), "sample_ids": np.concatenate(sample_ids)}


def _score_normalization(scores, targets6, weak6):
    center, scale, orientation, diagnostics = [], [], [], {}
    for local, (index, name) in enumerate(zip(TARGET_INDICES, TARGET_NAMES)):
        weak_positive = (targets6[:, index] == 1) & weak6[:, index]
        negative = targets6[:, index] == 0
        weak_values, negative_values = scores[weak_positive, local], scores[negative, local]
        direction = 1.0 if np.median(weak_values) >= np.median(negative_values) else -1.0
        weak_values, negative_values = direction * weak_values, direction * negative_values
        midpoint = 0.5 * (np.median(weak_values) + np.median(negative_values))
        pooled = np.concatenate([weak_values, negative_values])
        robust = (np.percentile(pooled, 75) - np.percentile(pooled, 25)) / 1.349
        separation = 0.5 * abs(np.median(weak_values) - np.median(negative_values))
        denominator = max(float(robust), float(separation), 1e-3)
        center.append(float(midpoint)); scale.append(denominator); orientation.append(direction)
        diagnostics[name] = {"weak_count": int(weak_positive.sum()),
                             "negative_count": int(negative.sum()),
                             "weak_median": float(np.median(weak_values)),
                             "negative_median": float(np.median(negative_values)),
                             "center": float(midpoint), "scale": denominator,
                             "orientation": direction}
    return np.asarray(center), np.asarray(scale), np.asarray(orientation), diagnostics


def _reconnect(base_logits, expert_scores, center, scale, orientation, cap,
               uncertainty_power=1.0):
    logits = np.asarray(base_logits, dtype=np.float32).copy()
    oriented = expert_scores * orientation[None, :]
    normalized = (oriented - center[None, :]) / scale[None, :]
    target_base = logits[:, TARGET_INDICES]
    probability = _sigmoid(target_base)
    uncertainty = np.power(
        4.0 * probability * (1.0 - probability),
        float(uncertainty_power),
    )
    residual = float(cap) * np.tanh(normalized) * uncertainty
    logits[:, TARGET_INDICES] += residual.astype(np.float32)
    full_residual = np.zeros_like(logits)
    full_residual[:, TARGET_INDICES] = residual
    return logits, full_residual


def _strong_recall(targets5, probabilities5, weak5, thresholds):
    prediction = probabilities5 >= np.asarray(thresholds)[None, :]
    values, per_label = [], {}
    for index in TARGET_LOCAL:
        strong = (targets5[:, index] == 1) & ~weak5[:, index]
        value = float(prediction[strong, index].mean()) if strong.any() else float("nan")
        per_label[COMMON_NAMES[index]] = value; values.append(value)
    return float(np.nanmean(values)), per_label


def _evaluate(variant, logits, residual, arrays, role_ids, weak6, output_dir, args):
    targets = {role: arrays[role]["targets"] for role in logits}
    calibration = fit_temperature(logits["calibration"], targets["calibration"], "calibration")
    probability = {role: _sigmoid(apply_temperature(value, calibration.temperature))
                   for role, value in logits.items()}
    targets5 = {role: value[:, COMMON_INDICES] for role, value in targets.items()}
    weak5 = {role: weak6[role_ids[role]][:, COMMON_INDICES] for role in logits}
    prob5 = {role: value[:, COMMON_INDICES] for role, value in probability.items()}
    selection = select_per_label_weak_recall_fpr_thresholds(
        targets5["threshold"], prob5["threshold"], weak5["threshold"],
        args.max_negative_fpr, "threshold_selection", grid=np.arange(0.01, 1.0, 0.01))
    result = _metrics(targets5["early"], prob5["early"], weak5["early"],
                      selection.thresholds, arrays["early"]["targets"][:, 1].astype(bool))
    weak_auroc, per_weak_auroc = _target_weak_auroc(
        targets5["early"], prob5["early"], weak5["early"])
    strong_recall, per_strong = _strong_recall(
        targets5["early"], prob5["early"], weak5["early"], selection.thresholds)
    result.update({"variant": variant, "target_weak_auroc": weak_auroc,
                   "per_label_target_weak_auroc": per_weak_auroc,
                   "target_strong_recall": strong_recall,
                   "per_label_target_strong_recall": per_strong,
                   "temperature": calibration.temperature, "thresholds": selection.thresholds,
                   "residual_abs_mean": float(np.abs(residual["early"]).mean()),
                   "residual_abs_max": float(np.abs(residual["early"]).max()),
                   "development_v2_used": True, "test_v2_used": False,
                   "locked_test_used": False})
    _write(output_dir / "metrics.json", result)
    _write(output_dir / "calibration.json", calibration.__dict__)
    _write(output_dir / "thresholds.json", {"thresholds": selection.thresholds,
                                              "max_negative_fpr": args.max_negative_fpr,
                                              "source_role": "threshold_selection"})
    np.savez_compressed(output_dir / "predictions.npz", logits=logits["early"],
                        probabilities=probability["early"], residual=residual["early"],
                        targets=targets["early"], sample_ids=role_ids["early"])
    return result


def _run_seed(args, config, labels6, weak6, seed, device):
    plan = build_run_plan(config, "vtse_reconnection", seed, 0, args.max_eval_samples)
    optimization = np.asarray(plan["train_indices"], dtype=np.int64)
    rng = np.random.default_rng(133000 + seed); order = rng.permutation(len(optimization))
    cut = int(round(0.8 * len(order))); scorefit = optimization[order[cut:]]
    role_ids = {"scorefit": scorefit,
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
    center, scale, orientation, score_diagnostics = _score_normalization(
        arrays["scorefit"]["expert_scores"], arrays["scorefit"]["targets"], weak6[scorefit])
    seed_root = args.output_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True)
    _write(seed_root / "score_normalization.json", {"center": center, "scale": scale,
                                                      "orientation": orientation,
                                                      "diagnostics": score_diagnostics,
                                                      "source_role": "optimization_internal_validation"})
    rows = []
    for variant in VARIANTS:
        if variant == "B0_FrozenC6":
            logits = {role: arrays[role]["base_logits"] for role in ("early", "calibration", "threshold")}
            residual = {role: np.zeros_like(value) for role, value in logits.items()}
        else:
            converted = {role: _reconnect(arrays[role]["base_logits"],
                                           arrays[role]["expert_scores"], center, scale,
                                           orientation, args.residual_cap,
                                           args.uncertainty_power)
                         for role in ("early", "calibration", "threshold")}
            logits = {role: value[0] for role, value in converted.items()}
            residual = {role: value[1] for role, value in converted.items()}
        out = seed_root / variant; out.mkdir(parents=True, exist_ok=True)
        result = _evaluate(variant, logits, residual, arrays, role_ids, weak6, out, args)
        result.update({"seed": seed, "split_sha256": plan["split_sha256"]})
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
                                                151000 + vi * 100 + mi)
                              for mi, metric in enumerate(metrics)}
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        candidate = next(row for row in rows if row["seed"] == seed and row["variant"] == "E1_VTSE_Bounded")
        baseline = next(row for row in rows if row["seed"] == seed and row["variant"] == "B0_FrozenC6")
        paired.append({"seed": seed, **{metric: candidate[metric] - baseline[metric] for metric in metrics}})
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
             "residual_hard_bound": candidate["residual_abs_max"]["mean"] <= 0.350001}
    return aggregate, paired, mean, gates


def run(args):
    config = load_config(args.config); data_path = _data_path(config)
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        print(f"VTSE development-v2 seed={seed}", flush=True)
        rows.extend(_run_seed(args, config, labels6, weak6, seed, device))
    aggregate, paired, mean, gates = _aggregate(rows)
    payload = {"status": "completed", "experiment": "RouteA_VTSE_Bounded_v2",
               "seeds": args.seeds, "data_path": str(data_path), "data_sha256": _sha256(data_path),
               "uncertainty_power": args.uncertainty_power,
               "aggregate": aggregate, "paired_vtse_minus_c6": paired, "mean_deltas": mean,
               "promotion_gates": gates, "promoted": bool(all(gates.values())),
               "next_action": ("freeze_complete_candidate_then_run_test_v2_once" if all(gates.values())
                               else "redesign_reconnection_keep_visibility_expert"),
               "development_v2_used": True, "test_v2_used": False, "locked_test_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_visibility_temporal_expert_v2_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--h5-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_h5_background_ablation_v2")
    parser.add_argument("--stage23-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stage23_identifiability_v2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_visibility_temporal_expert_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--max-eval-samples", type=int, default=1200)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--residual-cap", type=float, default=0.35)
    parser.add_argument("--uncertainty-power", type=float, default=1.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
