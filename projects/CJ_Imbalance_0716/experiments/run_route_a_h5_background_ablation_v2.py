"""Frozen-F2 C6 versus H5 SawFM-background ablation on development-v2 only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import (  # noqa: E402
    build_label_longtail,
    compute_weak_component_mask,
)
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.thresholding import select_per_label_weak_recall_fpr_thresholds  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import (  # noqa: E402
    _component_power,
    _extract_features,
    _load_backbone,
    _standardize,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


ALL_NAMES = ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"]
COMMON_INDICES = np.asarray([0, 2, 3, 4, 5], dtype=np.int64)
COMMON_NAMES = [ALL_NAMES[index] for index in COMMON_INDICES]
TARGET_LOCAL = np.asarray([2, 3, 4], dtype=np.int64)  # MFT, VFT, ISRJ in H5 space.
TARGET_COUNTS = [1600, 400, 1213, 919, 696, 528]
VARIANTS = {"C6_PosBCE": 6, "H5_SawBackground": 5}
F2_MODEL_NAME = "F2_DFMS_Full"


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True,
                  default=lambda value: value.tolist() if isinstance(value, np.ndarray)
                  else float(value) if isinstance(value, np.floating)
                  else int(value) if isinstance(value, np.integer)
                  else str(value))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(logits):
    values = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def _predict(model, features, device, batch_size=2048):
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            tensor = torch.from_numpy(features[start:start + batch_size]).to(device)
            rows.append(model(tensor).cpu().float().numpy())
    return np.concatenate(rows, axis=0)


def _rank_score(targets, probabilities, weak):
    scores = []
    for index in TARGET_LOCAL:
        positive = (targets[:, index] == 1) & weak[:, index]
        negative = targets[:, index] == 0
        if positive.any() and negative.any():
            y = np.concatenate([np.ones(positive.sum()), np.zeros(negative.sum())])
            p = np.concatenate([probabilities[positive, index], probabilities[negative, index]])
            scores.append(roc_auc_score(y, p))
    return float(np.mean(scores)) if scores else float("nan")


def _train_head(name, output_dim, fit_x, fit_y6, fit_weak6, val_x, val_y6,
                val_weak6, seed, device, args):
    selected = np.arange(6) if output_dim == 6 else COMMON_INDICES
    fit_y = fit_y6[:, selected].astype(np.float32)
    val_y = val_y6[:, selected].astype(np.uint8)
    val_weak = val_weak6[:, selected]
    train_seed = 93000 + int(seed) + (0 if output_dim == 6 else 1000)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    model = nn.Linear(fit_x.shape[1], output_dim).to(device)
    positives = fit_y.sum(axis=0)
    pos_weight = torch.as_tensor(
        (len(fit_y) - positives) / np.maximum(positives, 1.0),
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    best_state = None
    best_epoch = 0
    best_key = (-float("inf"), -float("inf"))
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_logits = _predict(model, val_x, device)
        val_prob = _sigmoid(val_logits)
        val_common = val_y[:, COMMON_INDICES] if output_dim == 6 else val_y
        prob_common = val_prob[:, COMMON_INDICES] if output_dim == 6 else val_prob
        weak_common = val_weak[:, COMMON_INDICES] if output_dim == 6 else val_weak
        macro_aupr = float(np.mean([
            average_precision_score(val_common[:, i], prob_common[:, i])
            for i in range(5)
        ]))
        rank = _rank_score(val_common, prob_common, weak_common)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_macro_aupr_5": macro_aupr,
                        "validation_target_weak_auroc": rank})
        key = (macro_aupr, rank)
        if key > (best_key[0] + 1e-8, best_key[1]) or (
                np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8):
            best_key = key
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("head training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, train_seed, history


def _metrics(targets, probabilities, weak, thresholds, saw_present):
    predictions = probabilities >= np.asarray(thresholds)[None, :]
    per_label = {}
    for index, name in enumerate(COMMON_NAMES):
        positive = targets[:, index] == 1
        negative = ~positive
        weak_positive = positive & weak[:, index]
        per_label[name] = {
            "aupr": float(average_precision_score(targets[:, index], probabilities[:, index])),
            "f1": float(f1_score(targets[:, index], predictions[:, index], zero_division=0)),
            "weak_recall": float(predictions[weak_positive, index].mean()) if weak_positive.any() else float("nan"),
            "negative_fpr": float(predictions[negative, index].mean()) if negative.any() else 0.0,
            "weak_count": int(weak_positive.sum()),
        }
    def subset_f1(mask):
        if not mask.any():
            return float("nan")
        return float(np.mean([
            f1_score(targets[mask, i], predictions[mask, i], zero_division=0)
            for i in range(5)
        ]))
    return {
        "macro_aupr": float(np.mean([per_label[name]["aupr"] for name in COMMON_NAMES])),
        "macro_f1": float(np.mean([per_label[name]["f1"] for name in COMMON_NAMES])),
        "target_weak_recall": float(np.nanmean([per_label[COMMON_NAMES[i]]["weak_recall"] for i in TARGET_LOCAL])),
        "target_negative_fpr": float(np.mean([per_label[COMMON_NAMES[i]]["negative_fpr"] for i in TARGET_LOCAL])),
        "saw_present_macro_f1": subset_f1(saw_present),
        "saw_absent_macro_f1": subset_f1(~saw_present),
        "per_label": per_label,
    }


def _run_seed(args, config, labels6, weak6, seed, device):
    plan = build_run_plan(config, "h5_background", seed, 0, args.max_eval_samples)
    manifest = build_label_longtail(
        labels6, plan["train_indices"], TARGET_COUNTS, seed,
        tolerance=0.0, source_split_sha256=plan["split_sha256"],
        target_sample_count=2400, sample_count_tolerance=0.0,
    )
    train_ids = manifest.selected_indices
    roles = {"train": train_ids,
             "early": np.asarray(plan["early_stop_indices"], dtype=np.int64),
             "calibration": np.asarray(plan["calibration_indices"], dtype=np.int64),
             "threshold": np.asarray(plan["threshold_indices"], dtype=np.int64)}
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    model, checkpoint_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    extracted = {role: _extract_features(model, _data_path(config), ids,
                                          config["normalization"], device,
                                          args.feature_batch_size)
                 for role, ids in roles.items()}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rng = np.random.default_rng(95000 + seed)
    order = rng.permutation(len(train_ids))
    cut = int(round(0.8 * len(order)))
    fit, validation = order[:cut], order[cut:]
    mean = extracted["train"]["main"][fit].mean(axis=0, keepdims=True)
    std = np.maximum(extracted["train"]["main"][fit].std(axis=0, keepdims=True), 1e-5)
    features = {role: _standardize(data["main"], mean, std) for role, data in extracted.items()}
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    _write(seed_root / "longtail_manifest.json", manifest.to_dict())
    _write(seed_root / "split_receipt.json", {
        "seed": seed, "split_sha256": plan["split_sha256"],
        "manifest_sha256": manifest.sha256,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "development_v2_used": True, "test_v2_used": False,
        "locked_test_used": False,
        "roles": {name: values.tolist() for name, values in roles.items()},
    })
    rows = []
    for name, output_dim in VARIANTS.items():
        out = seed_root / name
        out.mkdir(parents=True, exist_ok=True)
        head, best_epoch, train_seed, history = _train_head(
            name, output_dim, features["train"][fit], extracted["train"]["targets"][fit],
            weak6[train_ids][fit], features["train"][validation],
            extracted["train"]["targets"][validation], weak6[train_ids][validation],
            seed, device, args)
        selected = np.arange(6) if output_dim == 6 else COMMON_INDICES
        logits = {role: _predict(head, features[role], device)
                  for role in ("early", "calibration", "threshold")}
        targets = {role: extracted[role]["targets"][:, selected]
                   for role in logits}
        weak = {role: weak6[roles[role]][:, selected] for role in logits}
        calibration = fit_temperature(logits["calibration"], targets["calibration"], "calibration")
        probability = {role: _sigmoid(apply_temperature(value, calibration.temperature))
                       for role, value in logits.items()}
        target_common = {role: value[:, COMMON_INDICES] if output_dim == 6 else value
                         for role, value in targets.items()}
        weak_common = {role: value[:, COMMON_INDICES] if output_dim == 6 else value
                       for role, value in weak.items()}
        prob_common = {role: value[:, COMMON_INDICES] if output_dim == 6 else value
                       for role, value in probability.items()}
        threshold = select_per_label_weak_recall_fpr_thresholds(
            target_common["threshold"], prob_common["threshold"],
            weak_common["threshold"], args.max_negative_fpr,
            "threshold_selection", grid=np.arange(0.01, 1.0, 0.01))
        result = _metrics(target_common["early"], prob_common["early"],
                          weak_common["early"], threshold.thresholds,
                          extracted["early"]["targets"][:, 1].astype(bool))
        result.update({"seed": seed, "variant": name, "best_epoch": best_epoch,
                       "train_seed": train_seed, "thresholds": threshold.thresholds,
                       "temperature": calibration.temperature,
                       "split_sha256": plan["split_sha256"],
                       "manifest_sha256": manifest.sha256,
                       "development_v2_used": True, "test_v2_used": False,
                       "locked_test_used": False})
        torch.save({"model_state_dict": head.state_dict(), "best_epoch": best_epoch,
                    "train_seed": train_seed, "feature_mean": mean, "feature_std": std,
                    "common_indices": COMMON_INDICES}, out / "best_head.pth")
        with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
        _write(out / "calibration.json", calibration.__dict__)
        _write(out / "thresholds.json", {"thresholds": threshold.thresholds,
                                          "max_negative_fpr": args.max_negative_fpr,
                                          "source_role": "threshold_selection"})
        _write(out / "metrics.json", result)
        np.savez_compressed(out / "predictions.npz", logits=logits["early"],
                            probabilities=probability["early"], targets=targets["early"],
                            sample_ids=roles["early"])
        rows.append(result)
    return rows


def _aggregate(rows):
    metrics = ["macro_aupr", "macro_f1", "target_weak_recall",
               "target_negative_fpr", "saw_present_macro_f1", "saw_absent_macro_f1"]
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {metric: _mean_ci([row[metric] for row in selected],
                                                97100 + vi * 100 + mi)
                              for mi, metric in enumerate(metrics)}
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        base = next(row for row in rows if row["seed"] == seed and row["variant"] == "C6_PosBCE")
        h5 = next(row for row in rows if row["seed"] == seed and row["variant"] == "H5_SawBackground")
        delta = {"seed": seed}
        for metric in metrics:
            delta[metric] = h5[metric] - base[metric]
        delta["target_aupr_floor"] = min(
            h5["per_label"][name]["aupr"] - base["per_label"][name]["aupr"]
            for name in ("MFT", "VFT", "ISRJ"))
        paired.append(delta)
    means = {metric: float(np.mean([row[metric] for row in paired])) for metric in metrics}
    means["target_aupr_floor"] = float(min(row["target_aupr_floor"] for row in paired))
    gates = {
        "macro_aupr_noninferior": means["macro_aupr"] >= -0.005,
        "macro_f1_gain": means["macro_f1"] >= 0.010,
        "target_weak_recall_gain": means["target_weak_recall"] >= 0.030,
        "target_each_aupr_noninferior": means["target_aupr_floor"] >= -0.010,
        "target_fpr_cap": aggregate["H5_SawBackground"]["target_negative_fpr"]["mean"] <= 0.20,
        "four_of_five_positive_weak_gain": sum(row["target_weak_recall"] > 0 for row in paired) >= 4,
        "saw_present_gain": means["saw_present_macro_f1"] > 0,
        "saw_absent_noninferior": means["saw_absent_macro_f1"] >= -0.005,
    }
    return aggregate, paired, means, gates


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    if "development_v2" not in str(config.get("evidence_role", "")):
        raise ValueError("configuration is not marked development_v2")
    forbidden = str(config.get("test_v2_path", ""))
    if forbidden and Path(forbidden).resolve() == data_path.resolve():
        raise ValueError("test-v2 must remain sealed")
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    component_power = _component_power(parameter_values)
    weak6 = compute_weak_component_mask(labels6, component_power, args.weak_threshold_db)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        print(f"development-v2 seed={seed}", flush=True)
        rows.extend(_run_seed(args, config, labels6, weak6, seed, device))
    aggregate, paired, means, gates = _aggregate(rows)
    payload = {
        "status": "completed", "experiment": "RouteA_H5_SawBackground_v2",
        "research_question": "Does removing SawFM output competition help the other five labels?",
        "seeds": args.seeds, "data_path": str(data_path),
        "data_sha256": _sha256(data_path), "device": str(device),
        "aggregate": aggregate, "paired_deltas_h5_minus_c6": paired,
        "mean_deltas": means, "promotion_gates": gates,
        "promoted": bool(all(gates.values())),
        "next_action": "unfreeze_last_multiscale_stage" if all(gates.values())
                       else "stop_head_only_route_and_test_label_conditioned_local_evidence",
        "development_v2_used": True, "test_v2_used": False,
        "locked_test_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_h5_background_ablation_v2_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_h5_background_ablation_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--max-eval-samples", type=int, default=1200)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
