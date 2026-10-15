"""Fair development-v3 mainstream loss baselines with a shared frozen RawSaw route."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import binomtest, wilcoxon
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.losses.classification import (  # noqa: E402
    DistributionBalancedLoss, MultiLabelClassificationLoss, RobustAsymmetricLoss, effective_number_weights,
)
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    COMMON_INDICES, TARGET_NAMES, _sha256, _target_weak_rank, _write_history, _write_json,
)
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME, _component_power, _configure_partial_scope, _macro_aupr, _predict,
)
from experiments.run_route_a_development_v3_label_selective_projection import (  # noqa: E402
    EXPECTED_V3_SHA256, _load_model_from_state,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    RawStage1SawFMFrontend, _evaluate, _load_roles, _raw_scores, _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


SAW_INDEX = 1
LOSS_VARIANTS = (
    "L0_BCE_RawSaw",
    "L1_PosBCE_RawSaw",
    "L2_Focal_RawSaw",
    "L3_ASL_RawSaw",
    "L4_DB_RawSaw",
    "L5_CB_BCE_RawSaw",
    "L6_LogitAdjustment_RawSaw",
    "L7_DRLoss_RawSaw",
    "L8_RAL_RawSaw",
)
REFERENCES = ("R0_C4_Reference", "R1_F3_Reference")
ALL_VARIANTS = REFERENCES + LOSS_VARIANTS
LOSS_MODE = {
    "L0_BCE_RawSaw": "bce",
    "L1_PosBCE_RawSaw": "pos_bce",
    "L2_Focal_RawSaw": "focal",
    "L3_ASL_RawSaw": "asl",
    "L4_DB_RawSaw": "db",
    "L5_CB_BCE_RawSaw": "cb_bce",
    "L6_LogitAdjustment_RawSaw": "logit_adjustment",
    "L7_DRLoss_RawSaw": "dr_loss",
    "L8_RAL_RawSaw": "ral",
}


def _fit_statistics(targets):
    target = np.asarray(targets, dtype=np.float32)
    positives = target.sum(axis=0)
    count = len(target)
    pos_weight = (count - positives) / np.maximum(positives, 1.0)
    prior = np.clip(positives / float(count), 1e-4, 1.0 - 1e-4)
    logit_prior = np.log(prior / (1.0 - prior))
    cb_weight = effective_number_weights(positives.astype(np.int64), beta=0.99)
    return {
        "positive_counts": positives.astype(np.int64), "sample_count": int(count),
        "pos_weight": pos_weight.astype(np.float32), "logit_prior": logit_prior.astype(np.float32),
        "cb_weight": cb_weight.astype(np.float32),
    }


def _elementwise_loss(mode, logits, targets, stats, device):
    if mode == "pos_bce":
        weight = torch.as_tensor(stats["pos_weight"], dtype=torch.float32, device=device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight, reduction="none")
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if mode == "focal":
        return MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0).to(device).elementwise(logits, targets)
    if mode == "asl":
        return MultiLabelClassificationLoss("asl", gamma_pos=0.0, gamma_neg=3.0, clip=0.05).to(device).elementwise(logits, targets)
    if mode == "db":
        return DistributionBalancedLoss(stats["positive_counts"], stats["sample_count"]).to(device).elementwise(logits, targets)
    if mode == "cb_bce":
        return MultiLabelClassificationLoss("cb_bce", label_weights=stats["cb_weight"]).to(device).elementwise(logits, targets)
    if mode == "logit_adjustment":
        prior = torch.as_tensor(stats["logit_prior"], dtype=torch.float32, device=device)
        return F.binary_cross_entropy_with_logits(logits + prior[None, :], targets, reduction="none")
    raise ValueError(f"unsupported loss mode: {mode}")


def _dr_loss(logits, targets, gamma1=1.0, gamma2=1.0):
    """Literal logits/labels core of ECCV-2024 DR-Loss, reduced by label."""
    signed = (1.0 - 2.0 * targets) * logits
    neg_score = signed - targets * 1e12
    pos_score = signed - (1.0 - targets) * 1e12
    return torch.log(
        1.0
        + torch.exp(torch.logsumexp(pos_score * float(gamma1), dim=0))
        * torch.exp(torch.logsumexp(neg_score * float(gamma1), dim=0))
        + torch.exp(torch.logsumexp(neg_score * float(gamma2), dim=0))
    )


def _route_rawsaw(logits_by_role, raw_scores):
    output = {}
    for role, logits in logits_by_role.items():
        values = np.asarray(logits, dtype=np.float32).copy()
        values[:, SAW_INDEX] = np.asarray(raw_scores[role], dtype=np.float32)
        output[role] = values
    return output


def _cosine_multiplier(epoch_index, total_epochs, minimum_ratio):
    progress = min(max(float(epoch_index) / float(max(total_epochs, 1)), 0.0), 1.0)
    return float(minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress)))


def _validation_key(model, roles, weak6, device, args):
    logits = _predict(model, roles["validation"]["signals"], device, args.eval_batch_size)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[:, COMMON_INDICES], -40.0, 40.0)))
    targets = roles["validation"]["targets6"][:, COMMON_INDICES]
    weak = weak6[roles["validation"]["sample_ids"]][:, COMMON_INDICES]
    return _macro_aupr(targets, probability), _target_weak_rank(targets, probability, weak)


def _train_loss_model(model, variant, roles, weak6, seed, device, args, output_dir):
    mode = LOSS_MODE[variant]
    scope = _configure_partial_scope(model)
    stats = _fit_statistics(roles["fit"]["targets6"])
    optimizer = torch.optim.AdamW([
        {"params": scope["shared"], "lr": args.learning_rate * args.representation_lr_scale},
        {"params": scope["classifier"], "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)
    epochs = args.smoke_epochs if args.smoke else args.epochs
    schedule_name = getattr(args, "lr_schedule", "constant")
    scheduler = None
    if schedule_name == "cosine":
        minimum_ratio = float(getattr(args, "minimum_lr_ratio", .01))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda index: _cosine_multiplier(index, int(epochs), minimum_ratio)
        )
    elif schedule_name != "constant":
        raise ValueError(f"unsupported lr schedule: {schedule_name}")
    train_seed = int(seed) + 73000
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    dataset = TensorDataset(torch.from_numpy(roles["fit"]["signals"]),
                            torch.from_numpy(roles["fit"]["targets6"].astype(np.float32)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    started = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train(); losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                if mode == "dr_loss":
                    common = _dr_loss(logits[:, COMMON_INDICES], y[:, COMMON_INDICES]).mean()
                    saw = _dr_loss(logits[:, SAW_INDEX:SAW_INDEX + 1], y[:, SAW_INDEX:SAW_INDEX + 1]).mean()
                elif mode == "ral":
                    common = RobustAsymmetricLoss(reduction="mean").to(device)(logits[:, COMMON_INDICES], y[:, COMMON_INDICES])
                    saw = RobustAsymmetricLoss(reduction="mean").to(device)(logits[:, SAW_INDEX:SAW_INDEX + 1], y[:, SAW_INDEX:SAW_INDEX + 1])
                else:
                    element = _elementwise_loss(mode, logits, y, stats, device)
                    common = element[:, torch.as_tensor(COMMON_INDICES, device=device)].mean()
                    saw = element[:, SAW_INDEX].mean()
                loss = common + args.saw_loss_weight * saw
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scope["shared"] + scope["classifier"], args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        macro, weak_rank = _validation_key(model, roles, weak6, device, args)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_common_macro_aupr": macro,
                        "validation_target_weak_auroc": weak_rank,
                        "classifier_lr": float(optimizer.param_groups[1]["lr"]),
                        "shared_lr": float(optimizer.param_groups[0]["lr"])})
        key = (macro, weak_rank)
        if best_key is None or key[0] > best_key[0] + 1e-8 or (
            np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8
        ):
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_key, best_epoch, stale = key, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if scheduler is not None:
            scheduler.step()
    if best_state is None:
        raise RuntimeError(f"{variant} did not produce checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "variant": variant, "loss_mode": mode,
                "best_epoch": best_epoch, "train_seed": train_seed,
                "fit_statistics": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
                                   for key, value in stats.items()}}, output_dir / "best_model.pth")
    _write_history(output_dir / "history.csv", history)
    return model, {
        "loss_mode": mode, "best_epoch": int(best_epoch), "epochs_ran": len(history),
        "lr_schedule": schedule_name,
        "best_common_macro_aupr": float(best_key[0]), "best_target_weak_auroc": float(best_key[1]),
        "train_seconds": float(time.time() - started),
        "trainable_shared_parameters": int(sum(p.numel() for p in scope["shared"])),
        "trainable_classifier_parameters": int(sum(p.numel() for p in scope["classifier"])),
    }


def _load_rawsaw(args, seed, device):
    checkpoint_path = args.unified_root / f"seed_{seed}" / "checkpoints" / "rawsaw.pth"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RawStage1SawFMFrontend(mode="raw", hidden_channels=args.raw_hidden_channels,
                                   topk=args.raw_topk, dropout=args.raw_dropout).to(device)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    return model, checkpoint_path


def _reference_result(root, seed, directory, variant):
    source = root / f"seed_{seed}" / directory
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    calibration = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
    thresholds = json.loads((source / "thresholds.json").read_text(encoding="utf-8"))
    prediction_path = source / "predictions.npz"
    return {"variant": variant, "metrics": metrics, "calibration": calibration,
            "thresholds": thresholds, "predictions_sha256": _sha256(prediction_path)}


def _truncate_roles(roles, train_count, eval_count):
    output = {}
    for role, values in roles.items():
        count = train_count if role == "fit" else eval_count
        output[role] = {key: np.asarray(value[:count]) for key, value in values.items()}
    return output


def _run_seed(args, config, labels6, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    if args.smoke:
        roles = _truncate_roles(roles, args.smoke_train_samples, args.smoke_eval_samples)
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    template, f2_checkpoint = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial_state = copy.deepcopy({key: value.detach().cpu() for key, value in template.state_dict().items()})
    del template
    raw, raw_path = _load_rawsaw(args, seed, device)
    raw_scores = {role: _raw_scores(raw, values["signals"], device, args.raw_batch_size)
                  for role, values in roles.items()}
    del raw
    results = [
        _reference_result(args.unified_root, seed, "C4_SameCapacity_NoVisibility", REFERENCES[0]),
        _reference_result(args.f3_root, seed, "F3_AlignedPeriodicityFusionHead", REFERENCES[1]),
    ]
    training, model_hashes = {}, {}
    for variant in LOSS_VARIANTS:
        if variant == "L1_PosBCE_RawSaw":
            checkpoint_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = _load_model_from_state(f2_dir, device, payload["model_state_dict"])
            training[variant] = {"loss_mode": "pos_bce", "reused_exact_c6_checkpoint": True,
                                 "checkpoint_sha256": _sha256(checkpoint_path)}
            model_hashes[variant] = _sha256(checkpoint_path)
            _write_json(root / variant / "reused_checkpoint.json", {
                "source_checkpoint": str(checkpoint_path.resolve()),
                "source_checkpoint_sha256": model_hashes[variant],
                "reuse_reason": "Exact development-v3 C6 PosBCE checkpoint with the same F2 initialization and training scope.",
                "locked_test_used": False,
                "test_v3_used": False,
            })
        else:
            model = _load_model_from_state(f2_dir, device, initial_state)
            model, training[variant] = _train_loss_model(
                model, variant, roles, weak6, seed, device, args, root / variant
            )
            checkpoint_path = root / variant / "best_model.pth"
            model_hashes[variant] = _sha256(checkpoint_path)
        logits = {role: _predict(model, values["signals"], device, args.eval_batch_size)
                  for role, values in roles.items()}
        routed = _route_rawsaw(logits, raw_scores)
        results.append(_evaluate(variant, routed, roles, weak6, root / variant, args))
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    payload = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
        "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path),
        "f2_checkpoint_sha256": _sha256(f2_checkpoint), "rawsaw_checkpoint_sha256": _sha256(raw_path),
        "training": training, "model_sha256": model_hashes, "variants": results,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, payload); return payload


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _value(item, metric):
    m = item["metrics"]
    if metric in ("macro_aupr", "macro_f1"):
        return float(m[metric])
    if metric == "isrj_aupr": return float(m["per_label"]["ISRJ"]["aupr"])
    if metric == "isrj_weak_recall": return float(m["per_label"]["ISRJ"]["weak_recall"])
    if metric == "isrj_strong_recall": return float(m["per_label"]["ISRJ"]["strong_recall"])
    if metric == "isrj_fpr": return float(m["per_label"]["ISRJ"]["negative_fpr"])
    if metric == "isrj_weak_auroc": return float(m["per_label"]["ISRJ"]["weak_vs_negative_auroc"])
    if metric == "target_weak_recall": return float(np.mean([m["per_label"][name]["weak_recall"] for name in TARGET_NAMES]))
    if metric == "target_weak_auroc": return float(np.mean([m["per_label"][name]["weak_vs_negative_auroc"] for name in TARGET_NAMES]))
    if metric == "target_fpr": return float(np.mean([m["per_label"][name]["negative_fpr"] for name in TARGET_NAMES]))
    raise KeyError(metric)


METRICS = ("macro_aupr", "macro_f1", "target_weak_recall", "target_weak_auroc", "target_fpr",
           "isrj_aupr", "isrj_weak_recall", "isrj_strong_recall", "isrj_fpr", "isrj_weak_auroc")


def _comparison(rows, candidate, baseline, offset):
    per_seed = []
    for row in rows:
        variants = _variant_map(row)
        per_seed.append({"seed": row["seed"], **{
            metric: _value(variants[candidate], metric) - _value(variants[baseline], metric)
            for metric in METRICS
        }})
    aggregate = {metric: _mean_ci([row[metric] for row in per_seed], offset + index)
                 for index, metric in enumerate(METRICS)}
    aggregate["positive_isrj_weak_recall_seeds"] = int(sum(row["isrj_weak_recall"] > 0 for row in per_seed))
    aggregate["positive_isrj_weak_auroc_seeds"] = int(sum(row["isrj_weak_auroc"] > 0 for row in per_seed))
    statistics = {metric: _paired_directional_statistics([row[metric] for row in per_seed])
                  for metric in METRICS}
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed,
            "aggregate": aggregate, "paired_directional_statistics": statistics}


def _paired_directional_statistics(differences):
    values = np.asarray(differences, dtype=np.float64)
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return {"n": int(values.size), "nonzero_n": 0, "positive_n": 0,
                "wilcoxon_greater_p": 1.0, "sign_test_greater_p": 1.0}
    try:
        wilcoxon_p = float(wilcoxon(nonzero, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    positive_n = int(np.sum(nonzero > 0.0))
    sign_p = float(binomtest(positive_n, int(nonzero.size), p=0.5, alternative="greater").pvalue)
    return {"n": int(values.size), "nonzero_n": int(nonzero.size), "positive_n": positive_n,
            "wilcoxon_greater_p": wilcoxon_p, "sign_test_greater_p": sign_p}


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    labels6 = read_labels(data_path); values79, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values79), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_DevelopmentV3_MainstreamLossBaselines", "seeds": seeds,
        "losses": LOSS_MODE, "epochs": args.epochs, "patience": args.patience,
        "learning_rate": args.learning_rate, "representation_lr_scale": args.representation_lr_scale,
        "saw_loss_weight": args.saw_loss_weight, "shared_rawsaw_route": True,
        "locked_test_used": False, "test_v3_used": False,
    })
    rows = [_run_seed(args, config, labels6, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {variant: {metric: _mean_ci([
        _value(_variant_map(row)[variant], metric) for row in rows
    ], 20318000 + vi * 100 + mi) for mi, metric in enumerate(METRICS)}
        for vi, variant in enumerate(ALL_VARIANTS)}
    traditional = list(LOSS_VARIANTS) + [REFERENCES[0]]
    strongest = max(traditional, key=lambda name: (aggregate[name]["macro_aupr"]["mean"],
                                                    aggregate[name]["target_weak_auroc"]["mean"]))
    comparison = _comparison(rows, REFERENCES[1], strongest, 20319000)
    if args.smoke:
        gate = {"passed": False, "checks": {}, "evidence_level": "smoke_only"}
    else:
        delta = comparison["aggregate"]
        checks = {
            "f3_isrj_aupr_positive": delta["isrj_aupr"]["mean"] > 0,
            "f3_isrj_weak_recall_ge_0p030": delta["isrj_weak_recall"]["mean"] >= .030,
            "f3_isrj_weak_auroc_ge_0p010": delta["isrj_weak_auroc"]["mean"] >= .010,
            "four_of_five_positive_weak_recall": delta["positive_isrj_weak_recall_seeds"] >= 4,
            "four_of_five_positive_weak_auroc": delta["positive_isrj_weak_auroc_seeds"] >= 4,
            "f3_isrj_fpr_le_0p20": aggregate[REFERENCES[1]]["isrj_fpr"]["mean"] <= .20,
            "test_boundary_safe": all(not row["test_v3_used"] for row in rows),
        }
        gate = {"passed": bool(all(checks.values())), "checks": checks,
                "evidence_level": "five_seed_development_v3_fair_loss_baselines"}
    payload = {"status": "completed", "experiment": "RouteA_DevelopmentV3_MainstreamLossBaselines",
               "smoke": bool(args.smoke), "seeds": seeds, "data_sha256": data_sha,
               "aggregate": aggregate, "strongest_traditional_baseline": strongest,
               "f3_vs_strongest": comparison, "evidence_gate": gate,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    path = args.output_root / "route_a_development_v3_mainstream_loss_baselines_summary.json"
    _write_json(path, payload)
    print(json.dumps({"status": "completed", "smoke": bool(args.smoke), "strongest": strongest,
                      "gate": gate, "summary": str(path)}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--f3-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_mainstream_loss_baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128); parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4); parser.add_argument("--representation-lr-scale", type=float, default=.1)
    parser.add_argument("--saw-loss-weight", type=float, default=.2); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0); parser.add_argument("--max-negative-fpr", type=float, default=.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--raw-hidden-channels", type=int, default=16); parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=.1); parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2); parser.add_argument("--smoke-train-samples", type=int, default=128)
    parser.add_argument("--smoke-eval-samples", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
