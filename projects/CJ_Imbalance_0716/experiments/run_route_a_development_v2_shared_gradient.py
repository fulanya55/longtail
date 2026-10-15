"""Paired shared-representation SawFM gradient isolation on development-v2."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import build_label_longtail, compute_weak_component_mask  # noqa: E402
from cjlt.data.split_builder import build_stratified_roles  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    CLASS_NAMES6,
    COMMON_INDICES,
    COMMON_NAMES,
    LABEL_LT_TARGET_COUNTS,
    TARGET_COMMON_POSITIONS,
    TARGET_NAMES,
    _comparison,
    _component_power,
    _evaluate_variant,
    _indices_sha256,
    _mean_ci,
    _read_json,
    _sha256,
    _target_weak_rank,
    _validate_config,
    _write_history,
    _write_json,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


F2_MODEL_NAME = "F2_DFMS_Full"
VARIANTS = ("C6_Joint", "M5_GradientIsolated")


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
        "target_weak_vs_negative_auroc": ("metrics", "target_weak_vs_negative_auroc"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        rows = [_variant_map(seed_row)[variant] for seed_row in seed_rows]
        aggregate[variant] = {
            name: _mean_ci(
                [_metric(row, path) for row in rows],
                seed=20261100 + 100 * variant_index + offset,
            )
            for offset, (name, path) in enumerate(metric_paths.items())
        }
        aggregate[variant]["per_label_aupr"] = {
            label_name: _mean_ci(
                [row["metrics"]["per_label_diagnostics"][label_name]["aupr"] for row in rows],
                seed=20261200 + 100 * variant_index + label_index,
            )
            for label_index, label_name in enumerate(COMMON_NAMES)
        }
    return aggregate


def _promotion_gate(comparison, aggregate):
    delta = comparison["aggregate"]
    candidate = aggregate["M5_GradientIsolated"]
    checks = {
        "macro_aupr_noninferior_minus_0p005": delta["macro_aupr"]["mean"] >= -0.005,
        "macro_f1_gain_0p010": delta["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_0p030": delta["target_mean_weak_recall"]["mean"] >= 0.030,
        "target_label_aupr_noninferior_minus_0p010": all(
            delta["per_label_aupr"][name]["mean"] >= -0.010 for name in TARGET_NAMES
        ),
        "target_mean_fpr_0p20": candidate["target_mean_negative_fpr"]["mean"] <= 0.20,
        "positive_target_weak_recall_at_least_four": delta["positive_target_weak_recall_seeds"] >= 4,
        "sawfm_present_macro_f1_positive": delta["sawfm_present_macro_f1"]["mean"] > 0.0,
        "sawfm_absent_macro_f1_noninferior_minus_0p005": delta["sawfm_absent_macro_f1"]["mean"] >= -0.005,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _load_signal_array(path, ids, normalization, batch_size):
    dataset = V73SignalDataset(
        path=path,
        indices=ids,
        class_mode="six",
        normalization=normalization,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    signals, targets, sample_ids = [], [], []
    try:
        for batch in loader:
            signals.append(batch["signal"].numpy().astype(np.float32))
            targets.append(batch["label"].numpy().astype(np.uint8))
            sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {
        "signals": np.concatenate(signals, axis=0),
        "targets6": np.concatenate(targets, axis=0),
        "sample_ids": np.concatenate(sample_ids, axis=0),
    }


def _slice_roles(extracted, lengths):
    result = {}
    start = 0
    for role, length in lengths:
        end = start + int(length)
        result[role] = {key: np.asarray(value[start:end]) for key, value in extracted.items()}
        start = end
    if start != len(extracted["signals"]):
        raise AssertionError("signal role slicing did not consume all rows")
    return result


def _configure_partial_scope(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    shared = model.temporal_backbone[-1]
    for parameter in shared.parameters():
        parameter.requires_grad_(True)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    return {
        "shared": [parameter for parameter in shared.parameters() if parameter.requires_grad],
        "classifier": [parameter for parameter in model.classifier.parameters() if parameter.requires_grad],
    }


def _loss_components(logits, targets, pos_weight):
    element = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    common = element[:, torch.as_tensor(COMMON_INDICES, device=logits.device)].mean()
    saw = element[:, 1].mean()
    return common, saw


def _gradient_cosine(first, second):
    dot = sum((a * b).sum() for a, b in zip(first, second))
    norm_a = torch.sqrt(sum((a * a).sum() for a in first))
    norm_b = torch.sqrt(sum((b * b).sum() for b in second))
    return float((dot / (norm_a * norm_b + 1e-12)).detach().cpu())


def _gradient_diagnostic(model, signals, targets, pos_weight, device, batches, batch_size):
    model.eval()
    shared = [p for p in model.temporal_backbone[-1].parameters() if p.requires_grad]
    rows = []
    limit = min(len(signals), int(batches) * int(batch_size))
    for start in range(0, limit, int(batch_size)):
        x = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
        y = torch.from_numpy(targets[start:start + int(batch_size)].astype(np.float32)).to(device)
        logits = model(x)
        element = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight, reduction="none"
        )
        saw_loss = element[:, 1].mean()
        saw_grad = torch.autograd.grad(saw_loss, shared, retain_graph=True)
        row = {}
        for name, class_index in zip(TARGET_NAMES, (3, 4, 5)):
            target_grad = torch.autograd.grad(
                element[:, class_index].mean(), shared, retain_graph=True
            )
            row[name] = _gradient_cosine(saw_grad, target_grad)
        rows.append(row)
    return {
        "batch_count": len(rows),
        "per_target_mean_cosine": {
            name: float(np.mean([row[name] for row in rows])) for name in TARGET_NAMES
        },
        "per_target_negative_fraction": {
            name: float(np.mean([row[name] < 0.0 for row in rows])) for name in TARGET_NAMES
        },
    }


def _macro_aupr(targets, probabilities):
    return float(np.mean([
        average_precision_score(targets[:, index], probabilities[:, index])
        for index in range(targets.shape[1])
    ]))


def _predict(model, signals, device, batch_size):
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            x = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
            rows.append(logits.detach().cpu().float().numpy())
    return np.concatenate(rows, axis=0)


def _train_variant(model, variant, roles, weak6, device, seed, args, output_dir):
    scope = _configure_partial_scope(model)
    positives = roles["fit"]["targets6"].sum(axis=0).astype(np.float32)
    pos_weight = torch.as_tensor(
        (len(roles["fit"]["targets6"]) - positives) / np.maximum(positives, 1.0),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": scope["shared"], "lr": float(args.learning_rate) * float(args.representation_lr_scale)},
            {"params": scope["classifier"], "lr": float(args.learning_rate)},
        ],
        weight_decay=float(args.weight_decay),
    )
    train_seed = int(seed) + 73000
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    dataset = TensorDataset(
        torch.from_numpy(roles["fit"]["signals"]),
        torch.from_numpy(roles["fit"]["targets6"].astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(train_seed),
        num_workers=0,
    )
    best_state = None
    best_key = (-float("inf"), -float("inf"))
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                common_loss, saw_loss = _loss_components(model(x), y, pos_weight)
                total = common_loss
                if variant == "C6_Joint":
                    total = total + float(args.saw_loss_weight) * saw_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(scope["shared"] + scope["classifier"], float(args.gradient_clip))
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        validation_logits = _predict(model, roles["validation"]["signals"], device, args.eval_batch_size)
        probability = 1.0 / (1.0 + np.exp(-np.clip(validation_logits[:, COMMON_INDICES], -40.0, 40.0)))
        targets = roles["validation"]["targets6"][:, COMMON_INDICES]
        weak = weak6[roles["validation"]["sample_ids"]][:, COMMON_INDICES]
        macro = _macro_aupr(targets, probability)
        rank = _target_weak_rank(targets, probability, weak)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_common_macro_aupr": macro,
            "validation_target_weak_auroc": rank,
        })
        key = (macro, rank)
        if key > (best_key[0] + 1e-8, best_key[1]) or (
            np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8
        ):
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("partial adaptation produced no checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "variant": variant,
        "best_epoch": best_epoch,
        "train_seed": train_seed,
        "saw_loss_weight": float(args.saw_loss_weight if variant == "C6_Joint" else 0.0),
    }, output_dir / "best_model.pth")
    _write_history(output_dir / "history.csv", history)
    return model, {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_common_macro_aupr": float(best_key[0]),
        "best_target_weak_auroc": float(best_key[1]),
        "train_seconds": float(time.time() - started),
        "trainable_shared_parameters": int(sum(p.numel() for p in scope["shared"])),
        "trainable_classifier_parameters": int(sum(p.numel() for p in scope["classifier"])),
    }


def _run_seed(args, config, labels6, weak6, seed, device, data_sha256):
    seed_dir = Path(args.output_root) / "seed_{}".format(seed)
    summary_path = seed_dir / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return _read_json(summary_path)
    split = build_stratified_roles(labels6, np.arange(len(labels6), dtype=np.int64), seed=int(seed))
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
    order = np.arange(len(selected), dtype=np.int64)
    np.random.RandomState(int(seed) + 31000).shuffle(order)
    fit_count = int(round(0.8 * len(order)))
    ids = {
        "fit": selected[order[:fit_count]],
        "validation": selected[order[fit_count:]],
        "early": split.early_stop.copy(),
        "calibration": split.calibration.copy(),
        "threshold": split.threshold_selection.copy(),
    }
    if args.smoke:
        ids["fit"] = ids["fit"][: int(args.smoke_train_samples)]
        for role in ("validation", "early", "calibration", "threshold"):
            ids[role] = ids[role][: int(args.smoke_eval_samples)]
    all_ids = np.concatenate([ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")])
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError("shared-gradient roles overlap")
    extracted = _load_signal_array(
        _data_path(config), all_ids, str(config["normalization"]), args.io_batch_size
    )
    if not np.array_equal(extracted["sample_ids"], all_ids):
        raise AssertionError("signal extraction changed sample order")
    roles = _slice_roles(extracted, [(role, len(ids[role])) for role in ("fit", "validation", "early", "calibration", "threshold")])
    f2_dir = Path(args.f2_root) / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    template, checkpoint_path = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    _configure_partial_scope(template)
    positives = roles["fit"]["targets6"].sum(axis=0).astype(np.float32)
    pos_weight = torch.as_tensor(
        (len(roles["fit"]["targets6"]) - positives) / np.maximum(positives, 1.0),
        dtype=torch.float32,
        device=device,
    )
    gradient_diagnostic = _gradient_diagnostic(
        template,
        roles["fit"]["signals"],
        roles["fit"]["targets6"],
        pos_weight,
        device,
        args.gradient_diagnostic_batches,
        args.batch_size,
    )
    initial_state = copy.deepcopy({k: v.detach().cpu() for k, v in template.state_dict().items()})
    del template
    variants = []
    training = {}
    common_targets = {role: roles[role]["targets6"][:, COMMON_INDICES] for role in ("early", "calibration", "threshold")}
    common_weak = {role: weak6[roles[role]["sample_ids"]][:, COMMON_INDICES] for role in ("early", "calibration", "threshold")}
    saw = {role: roles[role]["targets6"][:, 1].astype(bool) for role in ("early", "calibration", "threshold")}
    saw["early_sample_ids"] = roles["early"]["sample_ids"]
    for variant in VARIANTS:
        model, _ = _load_backbone(f2_dir, F2_MODEL_NAME, device)
        model.load_state_dict(initial_state, strict=True)
        model, training[variant] = _train_variant(
            model, variant, roles, weak6, device, int(seed), args, seed_dir / variant
        )
        logits = {
            role: _predict(model, roles[role]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES]
            for role in ("early", "calibration", "threshold")
        }
        variants.append(_evaluate_variant(
            variant,
            logits,
            common_targets,
            common_weak,
            saw,
            common_targets["calibration"],
            seed_dir / variant,
            args,
        ))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_json(seed_dir / "longtail_manifest.json", manifest.to_dict())
    np.savez_compressed(
        str(seed_dir / "role_indices.npz"),
        manifest=selected,
        **ids,
    )
    seed_summary = {
        "seed": int(seed),
        "status": "completed",
        "smoke": bool(args.smoke),
        "data_sha256": data_sha256,
        "split_sha256": split.sha256,
        "manifest_sha256": manifest.sha256,
        "role_indices_sha256": _indices_sha256(*[ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")]),
        "f2_checkpoint": str(checkpoint_path),
        "f2_checkpoint_sha256": _sha256(checkpoint_path),
        "gradient_diagnostic": gradient_diagnostic,
        "training": training,
        "variants": variants,
        "role_counts": {role: int(len(values)) for role, values in ids.items()},
        "locked_test_used": False,
        "test_v2_used": False,
    }
    _write_json(summary_path, seed_summary)
    return seed_summary


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameter_values), float(args.weak_threshold_db))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.epochs = int(args.smoke_epochs) if args.smoke else int(args.epochs)
    data_sha256 = _sha256(data_path)
    seed_rows = [
        _run_seed(args, config, labels6, weak6, int(seed), device, data_sha256)
        for seed in args.seeds
    ]
    aggregate = _aggregate_variants(seed_rows)
    comparison = _comparison(seed_rows, "M5_GradientIsolated", "C6_Joint")
    gate = ({"checks": {}, "passed": False, "reason": "smoke_only"}
            if args.smoke else _promotion_gate(comparison, aggregate))
    gradient = {
        name: {
            "mean_cosine": float(np.mean([
                row["gradient_diagnostic"]["per_target_mean_cosine"][name]
                for row in seed_rows
            ])),
            "negative_seed_count": int(sum(
                row["gradient_diagnostic"]["per_target_mean_cosine"][name] < 0.0
                for row in seed_rows
            )),
        }
        for name in TARGET_NAMES
    }
    payload = {
        "status": "completed",
        "experiment": "route_a_development_v2_shared_gradient",
        "smoke": bool(args.smoke),
        "device": str(device),
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "variants": list(VARIANTS),
            "saw_loss_weight_c6": float(args.saw_loss_weight),
            "learning_rate": float(args.learning_rate),
            "representation_lr_scale": float(args.representation_lr_scale),
            "epochs": int(args.epochs),
            "label_lt_target_counts": LABEL_LT_TARGET_COUNTS.tolist(),
            "locked_test_used": False,
            "test_v2_used": False,
        },
        "data": {"path": str(data_path), "sha256": data_sha256, "sample_count": int(len(labels6))},
        "gradient_conflict": gradient,
        "per_seed": seed_rows,
        "aggregate": aggregate,
        "comparison_m5_minus_c6": comparison,
        "promotion_gate": gate,
        "next_decision": ("register_independent_confirmation" if gate.get("passed")
                          else "reject_plain_gradient_mask_and_design_conflict_aware_update"),
        "locked_test_used": False,
        "test_v2_used": False,
    }
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "route_a_development_v2_shared_gradient_summary.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "gradient_conflict": gradient,
        "comparison": comparison["aggregate"],
        "promotion_gate": gate,
        "next_decision": payload["next_decision"],
        "locked_test_used": False,
        "test_v2_used": False,
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v2_shared_gradient")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--representation-lr-scale", type=float, default=0.1)
    parser.add_argument("--saw-loss-weight", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--gradient-diagnostic-batches", type=int, default=4)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=128)
    parser.add_argument("--smoke-eval-samples", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
