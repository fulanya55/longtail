"""Frozen F2 stage2/stage3 weak-evidence identifiability on development-v2."""

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

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _component_power, _load_backbone  # noqa: E402
from experiments.run_route_a_h5_background_ablation_v2 import _sha256, _write  # noqa: E402
from experiments.run_route_a_bounded_local_contrast_v2 import _load_frozen_c6  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


TARGET_INDICES = np.asarray([3, 4, 5], dtype=np.int64)
TARGET_NAMES = ["MFT", "VFT", "ISRJ"]
REPRESENTATIONS = ("G_Global", "T_FinalTemporal", "S2_Physics", "S3_Physics", "S23_Multiscale")
F2_MODEL_NAME = "F2_DFMS_Full"


def _previous(values):
    return torch.cat([values[..., :1], values[..., :-1]], dim=-1)


def temporal_statistics(values, topk=8):
    count = min(int(topk), int(values.shape[-1]))
    mean = values.mean(dim=-1)
    variance = torch.clamp(values.pow(2).mean(dim=-1) - mean.pow(2), min=1e-5)
    maximum = values.max(dim=-1).values
    top_mean = torch.topk(values, k=count, dim=-1).values.mean(dim=-1)
    return torch.cat([mean, torch.sqrt(variance), maximum, top_mean], dim=1)


def complex_physics_statistics(real, imag, topk=8):
    magnitude = torch.sqrt(torch.clamp(real.pow(2) + imag.pow(2), min=1e-6))
    normalized_real, normalized_imag = real / magnitude, imag / magnitude
    previous_real, previous_imag = _previous(normalized_real), _previous(normalized_imag)
    phase_cos = normalized_real * previous_real + normalized_imag * previous_imag
    phase_sin = normalized_imag * previous_real - normalized_real * previous_imag
    descriptors = (
        magnitude,
        torch.abs(magnitude - _previous(magnitude)),
        phase_cos,
        phase_sin,
        torch.abs(phase_cos - _previous(phase_cos)),
        torch.abs(phase_sin - _previous(phase_sin)),
    )
    return torch.cat([temporal_statistics(item, topk) for item in descriptors], dim=1)


def _extract(backbone, base_head, mean, std, data_path, indices, normalization,
             device, batch_size, topk):
    dataset = V73SignalDataset(data_path, indices, "six", normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    rows = {name: [] for name in REPRESENTATIONS}
    base_logits, targets, sample_ids = [], [], []
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
                temporal = temporal_statistics(maps["temporal"].float(), topk)
                stage2 = complex_physics_statistics(
                    maps["stage2_real"].float(), maps["stage2_imag"].float(), topk)
                stage3 = complex_physics_statistics(
                    maps["stage3_real"].float(), maps["stage3_imag"].float(), topk)
                rows["G_Global"].append(global_feature.cpu().numpy())
                rows["T_FinalTemporal"].append(temporal.cpu().numpy())
                rows["S2_Physics"].append(stage2.cpu().numpy())
                rows["S3_Physics"].append(stage3.cpu().numpy())
                rows["S23_Multiscale"].append(torch.cat([stage2, stage3], dim=1).cpu().numpy())
                base_logits.append(base_head((global_feature - mean_tensor) / std_tensor).cpu().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {
        "features": {name: np.concatenate(values).astype(np.float32)
                     for name, values in rows.items()},
        "base_logits": np.concatenate(base_logits).astype(np.float32),
        "targets": np.concatenate(targets),
        "sample_ids": np.concatenate(sample_ids),
    }


def _safe_metrics(targets, scores):
    auroc, aupr = [], []
    for index in range(targets.shape[1]):
        auroc.append(float(roc_auc_score(targets[:, index], scores[:, index])))
        aupr.append(float(average_precision_score(targets[:, index], scores[:, index])))
    return {
        "macro_auroc": float(np.mean(auroc)), "macro_aupr": float(np.mean(aupr)),
        "per_label": {name: {"auroc": auroc[i], "aupr": aupr[i]}
                      for i, name in enumerate(TARGET_NAMES)},
    }


def _masked_view(targets6, weak6):
    positive = targets6[:, TARGET_INDICES].astype(bool)
    weak = weak6[:, TARGET_INDICES].astype(bool)
    mask = weak | ~positive
    target = weak.astype(np.float32)
    return target, mask


def _evaluate_masked(target, mask, scores):
    per_target, per_score = [], []
    for index in range(3):
        per_target.append(target[mask[:, index], index])
        per_score.append(scores[mask[:, index], index])
    metrics = {"per_label": {}}
    aurocs, auprs = [], []
    for index, name in enumerate(TARGET_NAMES):
        auroc = float(roc_auc_score(per_target[index], per_score[index]))
        aupr = float(average_precision_score(per_target[index], per_score[index]))
        metrics["per_label"][name] = {"auroc": auroc, "aupr": aupr,
                                      "weak_count": int(per_target[index].sum()),
                                      "negative_count": int(len(per_target[index]) - per_target[index].sum())}
        aurocs.append(auroc); auprs.append(aupr)
    metrics["macro_auroc"] = float(np.mean(aurocs))
    metrics["macro_aupr"] = float(np.mean(auprs))
    return metrics


def _train_probe(fit_x, validation_x, fit_targets6, validation_targets6,
                 fit_weak6, validation_weak6, seed, representation, device, args):
    train_seed = 131000 + seed + REPRESENTATIONS.index(representation) * 1000
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    fit_target, fit_mask = _masked_view(fit_targets6, fit_weak6)
    validation_target, validation_mask = _masked_view(validation_targets6, validation_weak6)
    positives = (fit_target * fit_mask).sum(axis=0)
    negatives = fit_mask.sum(axis=0) - positives
    pos_weight = torch.as_tensor(negatives / np.maximum(positives, 1.0),
                                 dtype=torch.float32, device=device)
    model = nn.Linear(fit_x.shape[1], 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_target),
                            torch.from_numpy(fit_mask.astype(np.uint8)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state, best_epoch, best_value, stale, history = None, 0, -float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight,
                                                      reduction="none")
            loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): validation_score = model(validation_tensor).cpu().numpy()
        metrics = _evaluate_masked(validation_target, validation_mask, validation_score)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_macro_auroc": metrics["macro_auroc"],
                        "validation_macro_aupr": metrics["macro_aupr"]})
        if metrics["macro_auroc"] > best_value + 1e-8:
            best_value, best_epoch = metrics["macro_auroc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("probe did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, train_seed, history, {
        name: {"fit_weak": int(positives[i]), "fit_negative": int(negatives[i])}
        for i, name in enumerate(TARGET_NAMES)}


def _run_seed(args, config, labels6, weak6, seed, device):
    plan = build_run_plan(config, "stage23_identifiability", seed, 0, args.max_eval_samples)
    optimization = np.asarray(plan["train_indices"], dtype=np.int64)
    early = np.asarray(plan["early_stop_indices"], dtype=np.int64)
    base_head, base_checkpoint, base_path = _load_frozen_c6(seed, args.h5_root, device)
    backbone_dir = args.screen_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    backbone, backbone_path = _load_backbone(backbone_dir, F2_MODEL_NAME, device)
    extracted = {
        "optimization": _extract(backbone, base_head, base_checkpoint["feature_mean"],
                                  base_checkpoint["feature_std"], _data_path(config),
                                  optimization, config["normalization"], device,
                                  args.feature_batch_size, args.topk),
        "early": _extract(backbone, base_head, base_checkpoint["feature_mean"],
                           base_checkpoint["feature_std"], _data_path(config), early,
                           config["normalization"], device, args.feature_batch_size, args.topk),
    }
    del backbone, base_head
    if device.type == "cuda": torch.cuda.empty_cache()
    rng = np.random.default_rng(133000 + seed)
    order = rng.permutation(len(optimization)); cut = int(round(0.8 * len(order)))
    fit, validation = order[:cut], order[cut:]
    seed_root = args.output_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True)
    base_target, base_mask = _masked_view(extracted["early"]["targets"], weak6[early])
    base_metrics = _evaluate_masked(base_target, base_mask,
                                    extracted["early"]["base_logits"][:, TARGET_INDICES])
    base_metrics.update({"seed": seed, "representation": "B0_FrozenC6",
                         "validation_macro_auroc": None, "validation_early_gap": None})
    _write(seed_root / "B0_FrozenC6" / "metrics.json", base_metrics)
    rows = [base_metrics]
    for representation in REPRESENTATIONS:
        train_features = extracted["optimization"]["features"][representation]
        early_features = extracted["early"]["features"][representation]
        mean = train_features[fit].mean(axis=0, keepdims=True)
        std = np.maximum(train_features[fit].std(axis=0, keepdims=True), 1e-5)
        normalized = ((train_features - mean) / std).astype(np.float32)
        normalized_early = ((early_features - mean) / std).astype(np.float32)
        model, best_epoch, train_seed, history, counts = _train_probe(
            normalized[fit], normalized[validation],
            extracted["optimization"]["targets"][fit],
            extracted["optimization"]["targets"][validation],
            weak6[optimization][fit], weak6[optimization][validation], seed,
            representation, device, args)
        model.eval()
        with torch.no_grad():
            validation_scores = model(torch.from_numpy(normalized[validation]).to(device)).cpu().numpy()
            early_scores = model(torch.from_numpy(normalized_early).to(device)).cpu().numpy()
        validation_target, validation_mask = _masked_view(
            extracted["optimization"]["targets"][validation], weak6[optimization][validation])
        early_target, early_mask = _masked_view(extracted["early"]["targets"], weak6[early])
        validation_metrics = _evaluate_masked(validation_target, validation_mask, validation_scores)
        metrics = _evaluate_masked(early_target, early_mask, early_scores)
        metrics.update({"seed": seed, "representation": representation,
                        "feature_dim": int(normalized.shape[1]), "best_epoch": best_epoch,
                        "train_seed": train_seed,
                        "validation_macro_auroc": validation_metrics["macro_auroc"],
                        "validation_macro_aupr": validation_metrics["macro_aupr"],
                        "validation_early_gap": validation_metrics["macro_auroc"] - metrics["macro_auroc"],
                        "counts": counts, "development_v2_used": True,
                        "test_v2_used": False, "locked_test_used": False})
        out = seed_root / representation; out.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "mean": mean, "std": std,
                    "representation": representation, "best_epoch": best_epoch,
                    "train_seed": train_seed}, out / "best_probe.pth")
        with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        _write(out / "metrics.json", metrics); rows.append(metrics)
    _write(seed_root / "run_context.json", {
        "seed": seed, "split_sha256": plan["split_sha256"],
        "optimization_samples": len(optimization), "early_samples": len(early),
        "backbone_sha256": _sha256(backbone_path), "base_head_sha256": _sha256(base_path),
        "development_v2_used": True, "test_v2_used": False, "locked_test_used": False})
    return rows


def _aggregate(rows):
    aggregate = {}
    for ri, representation in enumerate(("B0_FrozenC6",) + REPRESENTATIONS):
        selected = [row for row in rows if row["representation"] == representation]
        aggregate[representation] = {
            "macro_auroc": _mean_ci([row["macro_auroc"] for row in selected], 140000 + ri * 10),
            "macro_aupr": _mean_ci([row["macro_aupr"] for row in selected], 140001 + ri * 10),
            "per_label_auroc": {name: _mean_ci([row["per_label"][name]["auroc"] for row in selected],
                                                 141000 + ri * 100 + i)
                                for i, name in enumerate(TARGET_NAMES)},
            "validation_early_gap": (None if representation == "B0_FrozenC6" else
                                      _mean_ci([row["validation_early_gap"] for row in selected],
                                               140002 + ri * 10)),
        }
    candidates = list(REPRESENTATIONS)
    best = max(candidates, key=lambda name: aggregate[name]["macro_auroc"]["mean"])
    base = aggregate["B0_FrozenC6"]["macro_auroc"]["mean"]
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        candidate = next(row for row in rows if row["seed"] == seed and row["representation"] == best)
        baseline = next(row for row in rows if row["seed"] == seed and row["representation"] == "B0_FrozenC6")
        paired.append({"seed": seed, "macro_auroc": candidate["macro_auroc"] - baseline["macro_auroc"]})
    label_positive = sum(
        aggregate[best]["per_label_auroc"][name]["mean"] >=
        aggregate["B0_FrozenC6"]["per_label_auroc"][name]["mean"]
        for name in TARGET_NAMES)
    gates = {
        "macro_auroc_gain": aggregate[best]["macro_auroc"]["mean"] - base >= 0.020,
        "two_of_three_labels_positive": label_positive >= 2,
        "four_of_five_seeds_positive": sum(row["macro_auroc"] > 0 for row in paired) >= 4,
        "generalization_gap": aggregate[best]["validation_early_gap"]["mean"] <= 0.15,
        "absolute_auroc": aggregate[best]["macro_auroc"]["mean"] >= 0.63,
    }
    return aggregate, best, paired, gates


def run(args):
    config = load_config(args.config); data_path = _data_path(config)
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        print(f"stage23-identifiability development-v2 seed={seed}", flush=True)
        rows.extend(_run_seed(args, config, labels6, weak6, seed, device))
    aggregate, best, paired, gates = _aggregate(rows)
    payload = {"status": "completed", "experiment": "RouteA_Stage23Identifiability_v2",
               "seeds": args.seeds, "data_path": str(data_path), "data_sha256": _sha256(data_path),
               "aggregate": aggregate, "best_representation": best,
               "paired_best_minus_c6": paired, "promotion_gates": gates,
               "promoted": bool(all(gates.values())),
               "next_action": (f"build_low_rank_expert_from_{best}" if all(gates.values())
                               else "change_backbone_training_with_identity_paired_visibility_objective"),
               "development_v2_used": True, "test_v2_used": False, "locked_test_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_stage23_identifiability_v2_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--h5-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_h5_background_ablation_v2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stage23_identifiability_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--max-eval-samples", type=int, default=1200)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())

