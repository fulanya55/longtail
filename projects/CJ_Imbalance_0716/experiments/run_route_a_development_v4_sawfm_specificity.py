"""Development-v4 SawFM-specific paired task-direction causal controls."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import build_label_longtail, compute_weak_component_mask  # noqa: E402
from cjlt.data.split_builder import build_stratified_roles  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    COMMON_INDICES,
    COMMON_NAMES,
    LABEL_LT_TARGET_COUNTS,
    TARGET_NAMES,
    _comparison,
    _evaluate_variant,
    _indices_sha256,
    _read_json,
    _sha256,
    _target_weak_rank,
    _write_history,
    _write_json,
)
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _component_power,
    _configure_partial_scope,
    _load_signal_array,
    _macro_aupr,
    _predict,
    _slice_roles,
)
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


LABEL_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
MASKED_ENDPOINTS = {
    "M5_GradientIsolated": 1,
    "MaskFMB": 0,
    "MaskSinFM": 2,
    "MaskMFT": 3,
    "MaskVFT": 4,
    "MaskISRJ": 5,
}
OTHER_MASK_ENDPOINTS = tuple(name for name in MASKED_ENDPOINTS if name != "M5_GradientIsolated")
OTHER_CONTRASTS = tuple(name + "Contrast1p5" for name in OTHER_MASK_ENDPOINTS)
EVALUATED_VARIANTS = (
    "C6_Joint",
    "M5_GradientIsolated",
    "GradientContrastFixed1p5",
    "ModelAverage0p5",
    "RandomNormMatched1p5",
    "LayerShuffled1p5",
    *OTHER_CONTRASTS,
    "NonPairedSeedContrast1p5",
)
PRIMARY_CANDIDATE = "GradientContrastFixed1p5"
CONTROL_VARIANTS = (
    "ModelAverage0p5",
    "RandomNormMatched1p5",
    "LayerShuffled1p5",
    "NonPairedSeedContrast1p5",
)


def _validate_v4_config(config):
    if bool(config.get("allow_locked_test", False)):
        raise ValueError("development-v4 requires allow_locked_test=false")
    if str(config.get("evidence_role")) != "development_v4":
        raise ValueError("config evidence_role must be development_v4")
    if bool(config.get("test_v4_generated", True)):
        raise ValueError("test-v4 must not be generated during development-v4")
    if bool(config.get("test_v4_access_authorized", True)) or bool(config.get("test_v4_used", True)):
        raise ValueError("test-v4 access/use must remain false")
    if not bool(config.get("test_v3_sealed", False)) or bool(config.get("test_v3_used", True)):
        raise ValueError("test-v3 must remain sealed and unused")
    if not bool(config.get("test_v2_consumed", False)):
        raise ValueError("test-v2 must remain declared consumed")
    if bool(config.get("development_v2_reuse_for_selection", True)):
        raise ValueError("development-v2 reuse for selection must be false")
    if bool(config.get("development_v3_reuse_for_selection", True)):
        raise ValueError("development-v3 reuse for selection must be false")
    data_path = (Path(str(config["data_root"])) / str(config["train_file"])).resolve()
    forbidden = {
        Path(str(config["test_v4_path"])).resolve(),
        Path(str(config["sealed_test_v3_path"])).resolve(),
        Path(str(config["forbidden_test_v2_path"])).resolve(),
    }
    if data_path in forbidden or "test" in data_path.name.lower():
        raise ValueError("development-v4 data path resolves to a test role")


def _state_direction(base_state, endpoint_state):
    if base_state.keys() != endpoint_state.keys():
        raise ValueError("checkpoint state keys differ")
    direction = {}
    for key in base_state:
        left, right = base_state[key], endpoint_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("checkpoint tensor contract differs at {}".format(key))
        if torch.is_floating_point(left):
            direction[key] = right - left
        else:
            direction[key] = torch.zeros_like(left, dtype=torch.float32)
    return direction


def _apply_direction(base_state, direction, scale):
    result = {}
    for key, base in base_state.items():
        if torch.is_floating_point(base):
            result[key] = base + float(scale) * direction[key].to(dtype=base.dtype)
        else:
            result[key] = base.clone()
    return result


def _direction_norm(direction):
    total = sum(float((value.double() * value.double()).sum()) for value in direction.values())
    return float(np.sqrt(max(total, 0.0)))


def _direction_cosine(first, second):
    dot = sum(float((first[key].double() * second[key].double()).sum()) for key in first)
    norm = _direction_norm(first) * _direction_norm(second)
    return float(dot / (norm + 1e-24))


def _random_norm_matched(direction, seed):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    result = {}
    for key, value in direction.items():
        norm = float(torch.linalg.vector_norm(value.double()))
        if norm <= 0.0 or value.numel() == 0:
            result[key] = torch.zeros_like(value)
            continue
        random = torch.randn(value.shape, generator=generator, dtype=torch.float64)
        random = random * (norm / (torch.linalg.vector_norm(random) + 1e-24))
        result[key] = random.to(dtype=value.dtype)
    return result


def _layer_shuffled(direction, seed):
    result = {key: torch.zeros_like(value) for key, value in direction.items()}
    groups = {}
    for key, value in direction.items():
        if float(torch.linalg.vector_norm(value.double())) <= 0.0:
            continue
        groups.setdefault((tuple(value.shape), str(value.dtype)), []).append(key)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for keys in groups.values():
        ordered = sorted(keys)
        if len(ordered) > 1:
            shifted = ordered[1:] + ordered[:1]
            for destination, source in zip(ordered, shifted):
                result[destination] = direction[source].clone()
        else:
            key = ordered[0]
            flat = direction[key].reshape(-1)
            if flat.numel() <= 1:
                result[key] = direction[key].clone()
            else:
                permutation = torch.randperm(flat.numel(), generator=generator)
                result[key] = flat[permutation].reshape_as(direction[key]).clone()
    return result


def _per_tensor_norm_error(reference, candidate):
    errors = []
    for key in reference:
        left = float(torch.linalg.vector_norm(reference[key].double()))
        right = float(torch.linalg.vector_norm(candidate[key].double()))
        if left > 0.0:
            errors.append(abs(right - left) / left)
        elif right > 0.0:
            errors.append(float("inf"))
    return float(max(errors, default=0.0))


def _loss_for_mask(logits, targets, pos_weight, masked_index, saw_weight):
    element = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    active_common = [index for index in COMMON_INDICES if index != masked_index]
    common = element[:, torch.as_tensor(active_common, device=logits.device)].mean()
    if masked_index == 1:
        return common
    return common + float(saw_weight) * element[:, 1].mean()


def _clip_logits(logits, limit=40.0):
    """Apply one fixed numerical contract before temperature calibration."""

    values = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("model produced non-finite logits")
    return np.clip(values, -float(limit), float(limit))


def _evaluate_model_v4(variant, model, roles, weak6, seed_dir, device, args):
    logits = {
        role: _clip_logits(
            _predict(model, roles[role]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES]
        )
        for role in ("early", "calibration", "threshold")
    }
    targets = {
        role: roles[role]["targets6"][:, COMMON_INDICES]
        for role in logits
    }
    weak = {
        role: weak6[roles[role]["sample_ids"]][:, COMMON_INDICES]
        for role in logits
    }
    saw = {
        role: roles[role]["targets6"][:, 1].astype(bool)
        for role in logits
    }
    saw["early_sample_ids"] = roles["early"]["sample_ids"]
    return _evaluate_variant(
        variant,
        logits,
        targets,
        weak,
        saw,
        targets["calibration"],
        seed_dir / variant,
        args,
    )


def _train_endpoint(model, variant, masked_index, roles, weak6, device, seed, args, output_dir):
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
                total = _loss_for_mask(model(x), y, pos_weight, masked_index, args.saw_loss_weight)
            total.backward()
            torch.nn.utils.clip_grad_norm_(scope["shared"] + scope["classifier"], float(args.gradient_clip))
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        validation_logits = _predict(model, roles["validation"]["signals"], device, args.eval_batch_size)
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(validation_logits[:, COMMON_INDICES], -40.0, 40.0))
        )
        targets = roles["validation"]["targets6"][:, COMMON_INDICES]
        weak = weak6[roles["validation"]["sample_ids"]][:, COMMON_INDICES]
        macro = _macro_aupr(targets, probability)
        rank = _target_weak_rank(targets, probability, weak)
        history.append({
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)),
            "validation_common_macro_aupr": float(macro),
            "validation_target_weak_auroc": float(rank),
        })
        key = (float(macro), -float("inf") if not np.isfinite(rank) else float(rank))
        if key > (best_key[0] + 1e-8, best_key[1]) or (
            np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8
        ):
            best_key = key
            best_epoch = int(epoch)
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("endpoint training produced no checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "variant": variant,
        "masked_label_index": masked_index,
        "masked_label": None if masked_index is None else LABEL_NAMES[int(masked_index)],
        "best_epoch": best_epoch,
        "train_seed": train_seed,
        "saw_loss_weight": float(args.saw_loss_weight if masked_index != 1 else 0.0),
    }, output_dir / "best_model.pth")
    _write_history(output_dir / "history.csv", history)
    return model, {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_common_macro_aupr": float(best_key[0]),
        "best_target_weak_auroc": float(best_key[1]),
        "train_seconds": float(time.time() - started),
        "masked_label": None if masked_index is None else LABEL_NAMES[int(masked_index)],
        "trainable_shared_parameters": int(sum(p.numel() for p in scope["shared"])),
        "trainable_classifier_parameters": int(sum(p.numel() for p in scope["classifier"])),
    }


def _save_and_evaluate(variant, state, diagnostics, f2_dir, roles, weak6, seed_dir, device, args):
    variant_dir = seed_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": state,
        "variant": variant,
        "parameter_direction": diagnostics,
    }, variant_dir / "best_model.pth")
    model = _load_model_from_state(f2_dir, device, state)
    try:
        row = _evaluate_model_v4(variant, model, roles, weak6, seed_dir, device, args)
        row["status"] = "valid"
    except FloatingPointError as error:
        row = {
            "variant": variant,
            "status": "invalid_nonfinite_control",
            "error": str(error),
            "metrics": None,
        }
        _write_json(variant_dir / "invalid_control.json", row)
    row["parameter_direction"] = diagnostics
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _build_roles(args, config, labels6, seed):
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
        raise AssertionError("development-v4 roles overlap")
    extracted = _load_signal_array(_data_path(config), all_ids, str(config["normalization"]), args.io_batch_size)
    if not np.array_equal(extracted["sample_ids"], all_ids):
        raise AssertionError("development-v4 signal extraction changed sample order")
    roles = _slice_roles(
        extracted,
        [(role, len(ids[role])) for role in ("fit", "validation", "early", "calibration", "threshold")],
    )
    return split, manifest, ids, roles


def _run_seed_base(args, config, labels6, weak6, seed, device, data_sha256):
    seed_dir = Path(args.output_root) / "seed_{}".format(seed)
    summary_path = seed_dir / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return _read_json(summary_path)
    split, manifest, ids, roles = _build_roles(args, config, labels6, seed)
    f2_dir = Path(args.f2_root) / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    template, checkpoint_path = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial_state = copy.deepcopy({key: value.detach().cpu() for key, value in template.state_dict().items()})
    del template
    endpoint_specs = [("C6_Joint", None), *MASKED_ENDPOINTS.items()]
    endpoint_states = {}
    training = {}
    variants = []
    for variant, masked_index in endpoint_specs:
        model = _load_model_from_state(f2_dir, device, initial_state)
        model, training[variant] = _train_endpoint(
            model, variant, masked_index, roles, weak6, device, int(seed), args, seed_dir / variant
        )
        endpoint_states[variant] = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if variant in ("C6_Joint", "M5_GradientIsolated"):
            variants.append(_evaluate_model_v4(variant, model, roles, weak6, seed_dir, device, args))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    c6_state = endpoint_states["C6_Joint"]
    saw_direction = _state_direction(c6_state, endpoint_states["M5_GradientIsolated"])
    random_direction = _random_norm_matched(saw_direction, int(seed) + 91000)
    shuffled_direction = _layer_shuffled(saw_direction, int(seed) + 92000)
    derived = {
        "GradientContrastFixed1p5": (saw_direction, 1.5, "paired_sawfm_mask_direction"),
        "ModelAverage0p5": (saw_direction, 0.5, "paired_midpoint"),
        "RandomNormMatched1p5": (random_direction, 1.5, "random_per_tensor_norm_matched"),
        "LayerShuffled1p5": (shuffled_direction, 1.5, "layer_tensor_shuffled"),
    }
    for endpoint in OTHER_MASK_ENDPOINTS:
        derived[endpoint + "Contrast1p5"] = (
            _state_direction(c6_state, endpoint_states[endpoint]),
            1.5,
            "other_label_mask_direction",
        )
    direction_diagnostics = {}
    for variant, (direction, scale, source) in derived.items():
        diagnostics = {
            "source": source,
            "scale": float(scale),
            "direction_norm": _direction_norm(direction),
            "scaled_direction_norm": float(scale) * _direction_norm(direction),
            "cosine_vs_paired_sawfm_direction": _direction_cosine(direction, saw_direction),
            "per_tensor_norm_error_vs_paired_sawfm": (
                _per_tensor_norm_error(saw_direction, direction)
                if variant == "RandomNormMatched1p5" else None
            ),
        }
        direction_diagnostics[variant] = diagnostics
        variants.append(_save_and_evaluate(
            variant,
            _apply_direction(c6_state, direction, scale),
            diagnostics,
            f2_dir,
            roles,
            weak6,
            seed_dir,
            device,
            args,
        ))

    _write_json(seed_dir / "longtail_manifest.json", manifest.to_dict())
    np.savez_compressed(str(seed_dir / "role_indices.npz"), manifest=manifest.selected_indices, **ids)
    seed_summary = {
        "seed": int(seed),
        "status": "base_controls_completed",
        "smoke": bool(args.smoke),
        "data_sha256": data_sha256,
        "split_sha256": split.sha256,
        "manifest_sha256": manifest.sha256,
        "role_indices_sha256": _indices_sha256(*[ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")]),
        "f2_checkpoint": str(checkpoint_path),
        "f2_checkpoint_sha256": _sha256(checkpoint_path),
        "training": training,
        "direction_diagnostics": direction_diagnostics,
        "variants": variants,
        "role_counts": {role: int(len(values)) for role, values in ids.items()},
        "identity_audit_passed": True,
        "nonfinite_state_count": 0,
        "invalid_control_count": int(sum(row.get("metrics") is None for row in variants)),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
    }
    _write_json(summary_path, seed_summary)
    return seed_summary


def _load_checkpoint_state(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)["model_state_dict"]


def _append_nonpaired(args, config, labels6, weak6, seed_rows, device):
    if len(seed_rows) < 2:
        raise ValueError("NonPairedSeedContrast1p5 requires at least two seeds")
    by_seed = {int(row["seed"]): row for row in seed_rows}
    ordered = sorted(by_seed)
    result = []
    for index, seed in enumerate(ordered):
        row = by_seed[seed]
        existing = {variant["variant"] for variant in row["variants"]}
        if "NonPairedSeedContrast1p5" in existing:
            result.append(row)
            continue
        source_seed = ordered[(index + 1) % len(ordered)]
        seed_dir = Path(args.output_root) / "seed_{}".format(seed)
        source_dir = Path(args.output_root) / "seed_{}".format(source_seed)
        _, _, _, roles = _build_roles(args, config, labels6, seed)
        f2_dir = Path(args.f2_root) / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
        current_c6 = _load_checkpoint_state(seed_dir / "C6_Joint" / "best_model.pth")
        current_m5 = _load_checkpoint_state(seed_dir / "M5_GradientIsolated" / "best_model.pth")
        source_c6 = _load_checkpoint_state(source_dir / "C6_Joint" / "best_model.pth")
        source_m5 = _load_checkpoint_state(source_dir / "M5_GradientIsolated" / "best_model.pth")
        paired_direction = _state_direction(current_c6, current_m5)
        nonpaired_direction = _state_direction(source_c6, source_m5)
        diagnostics = {
            "source": "adjacent_seed_unpaired_direction",
            "source_seed": int(source_seed),
            "scale": 1.5,
            "direction_norm": _direction_norm(nonpaired_direction),
            "scaled_direction_norm": 1.5 * _direction_norm(nonpaired_direction),
            "cosine_vs_paired_sawfm_direction": _direction_cosine(nonpaired_direction, paired_direction),
        }
        variant_row = _save_and_evaluate(
            "NonPairedSeedContrast1p5",
            _apply_direction(current_c6, nonpaired_direction, 1.5),
            diagnostics,
            f2_dir,
            roles,
            weak6,
            seed_dir,
            device,
            args,
        )
        row["variants"].append(variant_row)
        row["direction_diagnostics"]["NonPairedSeedContrast1p5"] = diagnostics
        row["status"] = "completed"
        _write_json(seed_dir / "seed_summary.json", row)
        result.append(row)
    return result


def _metric(row, path):
    value = row
    for key in path:
        value = value[key]
    return float(value)


def _variant_map(seed_row):
    return {row["variant"]: row for row in seed_row["variants"]}


def _aggregate(seed_rows):
    paths = {
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
    for variant_index, variant in enumerate(EVALUATED_VARIANTS):
        all_rows = [_variant_map(seed_row)[variant] for seed_row in seed_rows]
        rows = [row for row in all_rows if row.get("metrics") is not None]
        if not rows:
            aggregate[variant] = {
                name: {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "count": 0,
                    "bootstrap_95_ci": [float("nan"), float("nan")],
                }
                for name in paths
            }
            aggregate[variant]["per_label_aupr"] = {
                label_name: {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "count": 0,
                    "bootstrap_95_ci": [float("nan"), float("nan")],
                }
                for label_name in COMMON_NAMES
            }
            aggregate[variant]["valid_seed_count"] = 0
            aggregate[variant]["invalid_seed_count"] = len(all_rows)
            continue
        aggregate[variant] = {
            name: _mean_ci([_metric(row, path) for row in rows], 20261800 + 100 * variant_index + offset)
            for offset, (name, path) in enumerate(paths.items())
        }
        aggregate[variant]["per_label_aupr"] = {
            label_name: _mean_ci(
                [row["metrics"]["per_label_diagnostics"][label_name]["aupr"] for row in rows],
                20261900 + 100 * variant_index + label_index,
            )
            for label_index, label_name in enumerate(COMMON_NAMES)
        }
        aggregate[variant]["valid_seed_count"] = len(rows)
        aggregate[variant]["invalid_seed_count"] = len(all_rows) - len(rows)
    return aggregate


def _safe_comparison(seed_rows, candidate, baseline):
    valid_seed_rows = []
    invalid_seeds = []
    for seed_row in seed_rows:
        variants = _variant_map(seed_row)
        if variants[candidate].get("metrics") is None or variants[baseline].get("metrics") is None:
            invalid_seeds.append(int(seed_row["seed"]))
        else:
            valid_seed_rows.append(seed_row)
    if valid_seed_rows:
        result = _comparison(valid_seed_rows, candidate, baseline)
    else:
        metric_names = (
            "macro_aupr",
            "macro_f1",
            "target_mean_weak_recall",
            "target_mean_negative_fpr",
            "sawfm_present_macro_f1",
            "sawfm_absent_macro_f1",
        )
        result = {
            "candidate": candidate,
            "baseline": baseline,
            "per_seed": [],
            "aggregate": {
                name: {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "count": 0,
                    "bootstrap_95_ci": [float("nan"), float("nan")],
                }
                for name in metric_names
            },
        }
        result["aggregate"]["per_label_aupr"] = {
            label_name: {
                "mean": float("nan"),
                "std": float("nan"),
                "count": 0,
                "bootstrap_95_ci": [float("nan"), float("nan")],
            }
            for label_name in COMMON_NAMES
        }
        result["aggregate"]["positive_target_weak_recall_seeds"] = 0
    result["valid_seed_count"] = len(valid_seed_rows)
    result["invalid_seed_count"] = len(invalid_seeds)
    result["invalid_seeds"] = invalid_seeds
    return result


def _effect_gate(comparison, aggregate, seed_rows):
    delta = comparison["aggregate"]
    candidate = aggregate[PRIMARY_CANDIDATE]
    stability = all(
        row["identity_audit_passed"]
        and row["nonfinite_state_count"] == 0
        and not row["locked_test_used"]
        and not row["test_v2_used"]
        and not row["test_v3_used"]
        and not row["test_v4_used"]
        for row in seed_rows
    )
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
        "finite_identity_and_test_safe": bool(stability),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _specificity_gate(comparisons):
    candidate_delta = comparisons[PRIMARY_CANDIDATE]["aggregate"]
    control_weak = {
        name: comparisons[name]["aggregate"]["target_mean_weak_recall"]["mean"]
        for name in CONTROL_VARIANTS
    }
    other_weak = {
        name: comparisons[name]["aggregate"]["target_mean_weak_recall"]["mean"]
        for name in OTHER_CONTRASTS
    }
    candidate_weak = candidate_delta["target_mean_weak_recall"]["mean"]
    finite_control_weak = [value for value in control_weak.values() if np.isfinite(value)]
    finite_other_weak = [value for value in other_weak.values() if np.isfinite(value)]
    control_margin = candidate_weak - max(finite_control_weak) if finite_control_weak else float("nan")
    other_margin = candidate_weak - max(finite_other_weak) if finite_other_weak else float("nan")
    saw_present_values = [
        comparisons[name]["aggregate"]["sawfm_present_macro_f1"]["mean"]
        for name in (*CONTROL_VARIANTS, *OTHER_CONTRASTS)
        if np.isfinite(comparisons[name]["aggregate"]["sawfm_present_macro_f1"]["mean"])
    ]
    saw_present_control_median = float(np.median(saw_present_values)) if saw_present_values else float("nan")
    all_control_outputs_finite = all(
        comparisons[name]["valid_seed_count"] == 5
        for name in (*CONTROL_VARIANTS, *OTHER_CONTRASTS)
    )
    checks = {
        "all_control_outputs_finite": all_control_outputs_finite,
        "candidate_beats_all_evaluable_generic_controls_on_weak_recall": control_margin > 0.0,
        "candidate_beats_all_other_label_directions_on_weak_recall": other_margin > 0.0,
        "one_specificity_margin_at_least_0p005_other_nonnegative": (
            (control_margin >= 0.005 and other_margin >= 0.0)
            or (other_margin >= 0.005 and control_margin >= 0.0)
        ),
        "sawfm_present_gain_above_control_median": (
            candidate_delta["sawfm_present_macro_f1"]["mean"] > saw_present_control_median
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "candidate_weak_recall_delta": candidate_weak,
        "generic_control_weak_recall_deltas": control_weak,
        "other_label_weak_recall_deltas": other_weak,
        "margin_over_best_generic_control": float(control_margin),
        "margin_over_best_other_label_direction": float(other_margin),
        "sawfm_present_control_median": saw_present_control_median,
    }


def run(args):
    config = load_config(args.config)
    _validate_v4_config(config)
    identity = _read_json(args.identity_audit)
    if not identity.get("identity_audit_passed") or identity.get("test_files_opened"):
        raise ValueError("development-v4 identity audit did not pass")
    if any(identity.get(name, True) for name in ("locked_test_used", "test_v2_used", "test_v3_used", "test_v4_used")):
        raise ValueError("identity audit reports forbidden test use")
    data_path = _data_path(config)
    data_sha256 = _sha256(data_path)
    if identity.get("sha256", "").lower() != data_sha256.lower():
        raise ValueError("identity audit SHA256 differs from development-v4 data")
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameter_values), float(args.weak_threshold_db))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.epochs = int(args.smoke_epochs) if args.smoke else int(args.epochs)
    seed_rows = [
        _run_seed_base(args, config, labels6, weak6, int(seed), device, data_sha256)
        for seed in args.seeds
    ]
    seed_rows = _append_nonpaired(args, config, labels6, weak6, seed_rows, device)
    aggregate = _aggregate(seed_rows)
    comparisons = {
        variant: _safe_comparison(seed_rows, variant, "C6_Joint")
        for variant in EVALUATED_VARIANTS
        if variant != "C6_Joint"
    }
    if args.smoke:
        effect_gate = {"checks": {}, "passed": False, "reason": "smoke_only"}
        specificity_gate = {"checks": {}, "passed": False, "reason": "smoke_only"}
        decision = "smoke_only_no_method_claim"
    else:
        effect_gate = _effect_gate(comparisons[PRIMARY_CANDIDATE], aggregate, seed_rows)
        specificity_gate = _specificity_gate(comparisons)
        if effect_gate["passed"] and specificity_gate["passed"]:
            decision = "promote_to_general_multitask_baseline_stage"
        elif effect_gate["passed"]:
            decision = "revise_sawfm_specificity_claim"
        else:
            decision = "stop_and_retain_nonreplication"
    payload = {
        "status": "completed",
        "experiment": "route_a_development_v4_sawfm_specificity",
        "evidence_level": "smoke" if args.smoke else "development",
        "smoke": bool(args.smoke),
        "device": str(device),
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "endpoint_variants": ["C6_Joint", *MASKED_ENDPOINTS.keys()],
            "evaluated_variants": list(EVALUATED_VARIANTS),
            "epochs": int(args.epochs),
            "saw_loss_weight": float(args.saw_loss_weight),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
            "test_v4_used": False,
        },
        "data": {
            "path": str(data_path),
            "sha256": data_sha256,
            "sample_count": int(len(labels6)),
            "identity_audit": identity,
        },
        "per_seed": seed_rows,
        "aggregate": aggregate,
        "comparisons_vs_c6": comparisons,
        "effect_replication_gate": effect_gate,
        "sawfm_specificity_gate": specificity_gate,
        "next_decision": decision,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "route_a_development_v4_sawfm_specificity_summary.json", payload)
    _write_json(output_root / "run_receipt.json", {
        "status": "completed",
        "evidence_level": payload["evidence_level"],
        "data_sha256": data_sha256,
        "identity_audit_passed": True,
        "effect_replication_gate_passed": bool(effect_gate["passed"]),
        "sawfm_specificity_gate_passed": bool(specificity_gate["passed"]),
        "next_decision": decision,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
        "test_access_allowed": False,
    })
    print(json.dumps({
        "status": payload["status"],
        "evidence_level": payload["evidence_level"],
        "comparisons": {name: value["aggregate"] for name, value in comparisons.items()},
        "effect_replication_gate": effect_gate,
        "sawfm_specificity_gate": specificity_gate,
        "next_decision": decision,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v4_used": False,
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v4.yaml",
    )
    parser.add_argument(
        "--identity-audit",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v4_generation" / "identity_audit.json",
    )
    parser.add_argument(
        "--f2-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v4_sawfm_specificity",
    )
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
