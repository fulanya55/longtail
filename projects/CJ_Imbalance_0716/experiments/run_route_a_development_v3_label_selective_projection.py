"""Development-v3 paired label-selective SawFM conflict projection."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import build_label_longtail, compute_weak_component_mask  # noqa: E402
from cjlt.data.split_builder import build_stratified_roles  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_gradient_contrast import _interpolate_state  # noqa: E402
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
    _train_variant,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = (
    "C6_Joint",
    "M5_GradientIsolated",
    "GradientContrastFixed1p5",
    "LSCP_TargetProjected",
)
TARGET_CLASS_INDICES = {"MFT": 3, "ISRJ": 5, "VFT": 4}
PRIMARY_CANDIDATE = "LSCP_TargetProjected"
REPLICATION_CANDIDATE = "GradientContrastFixed1p5"
EXPECTED_V3_SHA256 = "b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47"


def _validate_v3_config(config):
    if bool(config.get("allow_locked_test", False)):
        raise ValueError("development-v3 requires allow_locked_test=false")
    if str(config.get("evidence_role")) != "development_v3":
        raise ValueError("config evidence_role must be development_v3")
    if not bool(config.get("test_v3_sealed", False)):
        raise ValueError("test-v3 must remain declared sealed")
    if bool(config.get("test_v3_access_authorized", True)):
        raise ValueError("test-v3 access must remain unauthorized during development")
    if bool(config.get("test_v3_used", True)):
        raise ValueError("test-v3 must remain unused during development")
    if not bool(config.get("test_v2_consumed", False)):
        raise ValueError("test-v2 must remain declared consumed")
    if bool(config.get("development_v2_reuse_for_selection", True)):
        raise ValueError("development-v2 reuse for selection must be false")
    data_path = Path(str(config["data_root"])) / str(config["train_file"])
    test_v3_path = Path(str(config["test_v3_path"]))
    forbidden_test_v2_path = Path(str(config["forbidden_test_v2_path"]))
    if data_path.resolve() in {
        test_v3_path.resolve(), forbidden_test_v2_path.resolve()
    }:
        raise ValueError("development-v3 path resolves to a frozen test file")


def _dot(first, second):
    return sum((a * b).sum() for a, b in zip(first, second))


def _norm(values):
    return torch.sqrt(sum((value * value).sum() for value in values) + 1e-24)


def _cosine(first, second):
    return _dot(first, second) / (_norm(first) * _norm(second) + 1e-12)


def _project_saw_gradient(saw_gradient, target_gradients, passes=2):
    """Project only negative SawFM components using a fixed target order."""

    projected = [value.clone() for value in saw_gradient]
    pre = {
        name: float(_cosine(saw_gradient, target_gradients[name]).detach().cpu())
        for name in TARGET_CLASS_INDICES
    }
    applications = {name: 0 for name in TARGET_CLASS_INDICES}
    for _ in range(int(passes)):
        for name in TARGET_CLASS_INDICES:
            target = target_gradients[name]
            dot = _dot(projected, target)
            if float(dot.detach().cpu()) < 0.0:
                denominator = sum((value * value).sum() for value in target) + 1e-12
                coefficient = dot / denominator
                projected = [value - coefficient * basis for value, basis in zip(projected, target)]
                applications[name] += 1
    post = {
        name: float(_cosine(projected, target_gradients[name]).detach().cpu())
        for name in TARGET_CLASS_INDICES
    }
    removed = [left - right for left, right in zip(saw_gradient, projected)]
    removed_fraction = float((_norm(removed) / (_norm(saw_gradient) + 1e-12)).detach().cpu())
    return projected, {
        "pre_cosine": pre,
        "post_cosine": post,
        "applications": applications,
        "removed_norm_fraction": removed_fraction,
        "saw_gradient_norm": float(_norm(saw_gradient).detach().cpu()),
        "projected_saw_gradient_norm": float(_norm(projected).detach().cpu()),
    }


def _prediction_key(model, roles, weak6, device, args):
    logits = _predict(model, roles["validation"]["signals"], device, args.eval_batch_size)
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(logits[:, COMMON_INDICES], -40.0, 40.0))
    )
    targets = roles["validation"]["targets6"][:, COMMON_INDICES]
    weak = weak6[roles["validation"]["sample_ids"]][:, COMMON_INDICES]
    return _macro_aupr(targets, probability), _target_weak_rank(targets, probability, weak)


def _mean_nested(rows, group, name):
    return float(np.mean([row[group][name] for row in rows])) if rows else float("nan")


def _train_lscp(model, roles, weak6, device, seed, args, output_dir):
    scope = _configure_partial_scope(model)
    shared = scope["shared"]
    classifier = scope["classifier"]
    positives = roles["fit"]["targets6"].sum(axis=0).astype(np.float32)
    pos_weight = torch.as_tensor(
        (len(roles["fit"]["targets6"]) - positives) / np.maximum(positives, 1.0),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": shared, "lr": float(args.learning_rate) * float(args.representation_lr_scale)},
            {"params": classifier, "lr": float(args.learning_rate)},
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
    all_diagnostics = []
    nonfinite_count = 0
    clipping_count = 0
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss = []
        epoch_diagnostics = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                nonfinite_count += 1
                raise FloatingPointError("LSCP produced non-finite logits")
            element = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight, reduction="none"
            )
            common_loss = element[:, torch.as_tensor(COMMON_INDICES, device=device)].mean()
            saw_loss = element[:, 1].mean()
            target_losses = {
                name: element[:, class_index].mean()
                for name, class_index in TARGET_CLASS_INDICES.items()
            }
            common_shared = torch.autograd.grad(common_loss, shared, retain_graph=True)
            target_shared = {
                name: torch.autograd.grad(loss, shared, retain_graph=True)
                for name, loss in target_losses.items()
            }
            saw_shared = torch.autograd.grad(saw_loss, shared, retain_graph=True)
            common_classifier = torch.autograd.grad(common_loss, classifier, retain_graph=True)
            saw_classifier = torch.autograd.grad(saw_loss, classifier)
            projected_saw, diagnostic = _project_saw_gradient(
                saw_shared, target_shared, passes=int(args.projection_passes)
            )
            combined_shared = [
                common + float(args.saw_loss_weight) * saw
                for common, saw in zip(common_shared, projected_saw)
            ]
            combined_classifier = [
                common + float(args.saw_loss_weight) * saw
                for common, saw in zip(common_classifier, saw_classifier)
            ]
            gradients = combined_shared + combined_classifier
            if not all(torch.isfinite(value).all() for value in gradients):
                nonfinite_count += 1
                raise FloatingPointError("LSCP produced non-finite gradients")
            for parameter, gradient in zip(shared + classifier, gradients):
                parameter.grad = gradient.detach().clone()
            total_norm = torch.nn.utils.clip_grad_norm_(
                shared + classifier, float(args.gradient_clip)
            )
            if float(total_norm.detach().cpu()) > float(args.gradient_clip):
                clipping_count += 1
            optimizer.step()
            diagnostic["common_gradient_norm"] = float(_norm(common_shared).detach().cpu())
            diagnostic["combined_shared_gradient_norm"] = float(_norm(combined_shared).detach().cpu())
            epoch_diagnostics.append(diagnostic)
            all_diagnostics.append(diagnostic)
            epoch_loss.append(float((common_loss + float(args.saw_loss_weight) * saw_loss).detach().cpu()))
        macro, rank = _prediction_key(model, roles, weak6, device, args)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_loss)),
            "validation_common_macro_aupr": macro,
            "validation_target_weak_auroc": rank,
            "removed_norm_fraction": float(np.mean([item["removed_norm_fraction"] for item in epoch_diagnostics])),
            "common_gradient_norm": float(np.mean([item["common_gradient_norm"] for item in epoch_diagnostics])),
            "saw_gradient_norm": float(np.mean([item["saw_gradient_norm"] for item in epoch_diagnostics])),
            "combined_shared_gradient_norm": float(np.mean([item["combined_shared_gradient_norm"] for item in epoch_diagnostics])),
        }
        for name in TARGET_CLASS_INDICES:
            row["{}_pre_cosine".format(name)] = _mean_nested(epoch_diagnostics, "pre_cosine", name)
            row["{}_post_cosine".format(name)] = _mean_nested(epoch_diagnostics, "post_cosine", name)
            row["{}_projection_fraction".format(name)] = float(np.mean([
                item["applications"][name] > 0 for item in epoch_diagnostics
            ]))
        history.append(row)
        key = (macro, rank)
        if key > (best_key[0] + 1e-8, best_key[1]) or (
            np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8
        ):
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("LSCP training produced no checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "variant": PRIMARY_CANDIDATE,
            "best_epoch": best_epoch,
            "train_seed": train_seed,
            "projection_order": list(TARGET_CLASS_INDICES),
            "projection_passes": int(args.projection_passes),
        },
        output_dir / "best_model.pth",
    )
    _write_history(output_dir / "history.csv", history)
    aggregate_diagnostic = {
        "batch_count": len(all_diagnostics),
        "nonfinite_count": int(nonfinite_count),
        "gradient_clipping_count": int(clipping_count),
        "removed_norm_fraction_mean": float(np.mean([item["removed_norm_fraction"] for item in all_diagnostics])),
        "pre_cosine_mean": {
            name: _mean_nested(all_diagnostics, "pre_cosine", name)
            for name in TARGET_CLASS_INDICES
        },
        "post_cosine_mean": {
            name: _mean_nested(all_diagnostics, "post_cosine", name)
            for name in TARGET_CLASS_INDICES
        },
        "projection_batch_fraction": {
            name: float(np.mean([item["applications"][name] > 0 for item in all_diagnostics]))
            for name in TARGET_CLASS_INDICES
        },
        "projection_application_mean": {
            name: float(np.mean([item["applications"][name] for item in all_diagnostics]))
            for name in TARGET_CLASS_INDICES
        },
    }
    _write_json(output_dir / "gradient_diagnostics.json", aggregate_diagnostic)
    return model, {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "best_common_macro_aupr": float(best_key[0]),
        "best_target_weak_auroc": float(best_key[1]),
        "train_seconds": float(time.time() - started),
        "trainable_shared_parameters": int(sum(parameter.numel() for parameter in shared)),
        "trainable_classifier_parameters": int(sum(parameter.numel() for parameter in classifier)),
        "gradient_diagnostics": aggregate_diagnostic,
    }


def _load_model_from_state(f2_dir, device, state):
    model, _ = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    model.load_state_dict(state, strict=True)
    return model


def _evaluate_model(variant, model, roles, weak6, seed_dir, device, args):
    logits = {
        role: _predict(model, roles[role]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES]
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


def _run_seed(args, config, labels6, weak6, seed, device, data_sha256):
    seed_dir = Path(args.output_root) / "seed_{}".format(seed)
    summary_path = seed_dir / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return _read_json(summary_path)
    split = build_stratified_roles(
        labels6, np.arange(len(labels6), dtype=np.int64), seed=int(seed)
    )
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
    all_ids = np.concatenate([
        ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")
    ])
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError("development-v3 roles overlap")
    extracted = _load_signal_array(
        _data_path(config), all_ids, str(config["normalization"]), args.io_batch_size
    )
    if not np.array_equal(extracted["sample_ids"], all_ids):
        raise AssertionError("development-v3 signal extraction changed sample order")
    roles = _slice_roles(
        extracted,
        [(role, len(ids[role])) for role in ("fit", "validation", "early", "calibration", "threshold")],
    )
    f2_dir = Path(args.f2_root) / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    template, checkpoint_path = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial_state = copy.deepcopy({key: value.detach().cpu() for key, value in template.state_dict().items()})
    del template
    training = {}
    states = {}
    variants = []
    for variant in ("C6_Joint", "M5_GradientIsolated"):
        model = _load_model_from_state(f2_dir, device, initial_state)
        model, training[variant] = _train_variant(
            model, variant, roles, weak6, device, int(seed), args, seed_dir / variant
        )
        states[variant] = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        variants.append(_evaluate_model(variant, model, roles, weak6, seed_dir, device, args))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    contrast_state = _interpolate_state(
        states["C6_Joint"], states["M5_GradientIsolated"], 1.5
    )
    contrast_model = _load_model_from_state(f2_dir, device, contrast_state)
    contrast_dir = seed_dir / REPLICATION_CANDIDATE
    contrast_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": contrast_state,
            "variant": REPLICATION_CANDIDATE,
            "fixed_alpha": 1.5,
            "selection_source": "prior_development_v2_only",
        },
        contrast_dir / "best_model.pth",
    )
    variants.append(_evaluate_model(
        REPLICATION_CANDIDATE, contrast_model, roles, weak6, seed_dir, device, args
    ))
    del contrast_model
    lscp_model = _load_model_from_state(f2_dir, device, initial_state)
    lscp_model, training[PRIMARY_CANDIDATE] = _train_lscp(
        lscp_model, roles, weak6, device, int(seed), args, seed_dir / PRIMARY_CANDIDATE
    )
    variants.append(_evaluate_model(
        PRIMARY_CANDIDATE, lscp_model, roles, weak6, seed_dir, device, args
    ))
    del lscp_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _write_json(seed_dir / "longtail_manifest.json", manifest.to_dict())
    np.savez_compressed(str(seed_dir / "role_indices.npz"), manifest=selected, **ids)
    seed_summary = {
        "seed": int(seed),
        "status": "completed",
        "smoke": bool(args.smoke),
        "data_sha256": data_sha256,
        "split_sha256": split.sha256,
        "manifest_sha256": manifest.sha256,
        "role_indices_sha256": _indices_sha256(*[
            ids[role] for role in ("fit", "validation", "early", "calibration", "threshold")
        ]),
        "f2_checkpoint": str(checkpoint_path),
        "f2_checkpoint_sha256": _sha256(checkpoint_path),
        "training": training,
        "variants": variants,
        "role_counts": {role: int(len(values)) for role, values in ids.items()},
        "identity_audit_passed": True,
        "nonfinite_state_count": int(
            training[PRIMARY_CANDIDATE]["gradient_diagnostics"]["nonfinite_count"]
        ),
        "locked_test_used": False,
        "test_v2_used": False,
    }
    _write_json(summary_path, seed_summary)
    return seed_summary


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
    for variant_index, variant in enumerate(VARIANTS):
        rows = [_variant_map(seed_row)[variant] for seed_row in seed_rows]
        aggregate[variant] = {
            name: _mean_ci(
                [_metric(row, path) for row in rows],
                20261500 + 100 * variant_index + offset,
            )
            for offset, (name, path) in enumerate(paths.items())
        }
        aggregate[variant]["per_label_aupr"] = {
            label_name: _mean_ci(
                [row["metrics"]["per_label_diagnostics"][label_name]["aupr"] for row in rows],
                20261600 + 100 * variant_index + label_index,
            )
            for label_index, label_name in enumerate(COMMON_NAMES)
        }
    return aggregate


def _gate(candidate_name, comparison, aggregate, seed_rows):
    delta = comparison["aggregate"]
    candidate = aggregate[candidate_name]
    stability = all(
        row["identity_audit_passed"]
        and row["nonfinite_state_count"] == 0
        and not row["locked_test_used"]
        and not row["test_v2_used"]
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
        "finite_and_identity_safe": bool(stability),
        "locked_and_test_v2_unused": bool(stability),
    }
    return {"candidate": candidate_name, "checks": checks, "passed": bool(all(checks.values()))}


def _aggregate_lscp_diagnostics(seed_rows):
    rows = [row["training"][PRIMARY_CANDIDATE]["gradient_diagnostics"] for row in seed_rows]
    return {
        "removed_norm_fraction_mean": float(np.mean([row["removed_norm_fraction_mean"] for row in rows])),
        "pre_cosine_mean": {
            name: float(np.mean([row["pre_cosine_mean"][name] for row in rows]))
            for name in TARGET_CLASS_INDICES
        },
        "post_cosine_mean": {
            name: float(np.mean([row["post_cosine_mean"][name] for row in rows]))
            for name in TARGET_CLASS_INDICES
        },
        "projection_batch_fraction": {
            name: float(np.mean([row["projection_batch_fraction"][name] for row in rows]))
            for name in TARGET_CLASS_INDICES
        },
        "nonfinite_count": int(sum(row["nonfinite_count"] for row in rows)),
        "gradient_clipping_count": int(sum(row["gradient_clipping_count"] for row in rows)),
    }


def run(args):
    config = load_config(args.config)
    _validate_v3_config(config)
    data_path = _data_path(config)
    data_sha256 = _sha256(data_path)
    if data_sha256.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA256 differs from preregistration")
    identity = _read_json(args.identity_audit)
    if identity.get("sha256", "").lower() != data_sha256.lower():
        raise ValueError("identity audit SHA256 differs from development-v3 file")
    if not (
        identity.get("receiver_identity_disjoint_from_v2_train")
        and identity.get("receiver_identity_disjoint_from_v2_test")
        and not identity.get("test_v2_file_opened")
    ):
        raise ValueError("development-v3 identity audit did not pass")
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(
        labels6, _component_power(parameter_values), float(args.weak_threshold_db)
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.epochs = int(args.smoke_epochs) if args.smoke else int(args.epochs)
    seed_rows = [
        _run_seed(args, config, labels6, weak6, int(seed), device, data_sha256)
        for seed in args.seeds
    ]
    aggregate = _aggregate(seed_rows)
    comparisons = {
        candidate: _comparison(seed_rows, candidate, "C6_Joint")
        for candidate in ("M5_GradientIsolated", REPLICATION_CANDIDATE, PRIMARY_CANDIDATE)
    }
    if args.smoke:
        gates = {
            candidate: {"candidate": candidate, "checks": {}, "passed": False, "reason": "smoke_only"}
            for candidate in (REPLICATION_CANDIDATE, PRIMARY_CANDIDATE)
        }
    else:
        gates = {
            candidate: _gate(candidate, comparisons[candidate], aggregate, seed_rows)
            for candidate in (REPLICATION_CANDIDATE, PRIMARY_CANDIDATE)
        }
    if gates[PRIMARY_CANDIDATE]["passed"]:
        promoted = PRIMARY_CANDIDATE
    elif gates[REPLICATION_CANDIDATE]["passed"]:
        promoted = REPLICATION_CANDIDATE
    else:
        promoted = None
    payload = {
        "status": "completed",
        "experiment": "route_a_development_v3_label_selective_projection",
        "smoke": bool(args.smoke),
        "device": str(device),
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "variants": list(VARIANTS),
            "projection_order": list(TARGET_CLASS_INDICES),
            "projection_passes": int(args.projection_passes),
            "saw_loss_weight": float(args.saw_loss_weight),
            "epochs": int(args.epochs),
            "locked_test_used": False,
            "test_v2_used": False,
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
        "lscp_gradient_diagnostics": _aggregate_lscp_diagnostics(seed_rows),
        "promotion_gates": gates,
        "promoted_candidate": promoted,
        "next_decision": (
            "freeze_and_preregister_one_time_test_v2"
            if promoted is not None
            else "retain_failure_and_design_next_falsifiable_mechanism_on_fresh_data"
        ),
        "locked_test_used": False,
        "test_v2_used": False,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "route_a_development_v3_label_selective_projection_summary.json",
        payload,
    )
    _write_json(
        output_root / "run_receipt.json",
        {
            "status": "completed",
            "data_sha256": data_sha256,
            "identity_audit_passed": True,
            "promoted_candidate": promoted,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v2_access_allowed": bool(promoted is not None and not args.smoke),
        },
    )
    print(json.dumps({
        "status": payload["status"],
        "smoke": payload["smoke"],
        "lscp_gradient_diagnostics": payload["lscp_gradient_diagnostics"],
        "comparisons": {name: value["aggregate"] for name, value in comparisons.items()},
        "promotion_gates": gates,
        "promoted_candidate": promoted,
        "next_decision": payload["next_decision"],
        "locked_test_used": False,
        "test_v2_used": False,
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml",
    )
    parser.add_argument(
        "--identity-audit",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_generation" / "identity_audit.json",
    )
    parser.add_argument(
        "--f2-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection",
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
    parser.add_argument("--projection-passes", type=int, default=2)
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
