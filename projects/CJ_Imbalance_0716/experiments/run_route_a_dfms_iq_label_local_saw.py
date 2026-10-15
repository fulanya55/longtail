"""Frozen F2 label-query, local top-k, and early SawFM specialist screen."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import (  # noqa: E402
    V73SignalDataset,
    read_labels,
    read_parameters,
)
from cjlt.models.evidence_readout import (  # noqa: E402
    EarlySawFMEnvelopeIFExpert,
    LabelConditionedEvidenceReadout,
)
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import (  # noqa: E402
    CLASS_NAMES,
    HEAD_METRICS,
    SAW_INDEX,
    FrozenF2Head,
    _binary_rank_metrics,
    _component_power,
    _evaluate_variant,
    _macro_aupr,
    _sha256,
    _sigmoid,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


F2_MODEL_NAME = "F2_DFMS_Full"
VARIANTS = ("C0_H2", "Q1_LabelQuery", "Q2_LocalTopK", "Q3_EarlySawExpert")


class StructuredResidualAddOn(nn.Module):
    def __init__(self, readout, saw_expert=None):
        super().__init__()
        self.readout = readout
        self.saw_expert = saw_expert

    def forward(
        self,
        temporal,
        stage2_real,
        stage2_imag,
        base_logits,
        return_diagnostics=False,
    ):
        if return_diagnostics:
            logits, readout_diagnostics = self.readout(
                temporal, base_logits, return_diagnostics=True
            )
        else:
            logits = self.readout(temporal, base_logits)
            readout_diagnostics = None
        expert_diagnostics = None
        if self.saw_expert is not None:
            if return_diagnostics:
                expert_residual, expert_diagnostics = self.saw_expert(
                    stage2_real, stage2_imag, return_diagnostics=True
                )
            else:
                expert_residual = self.saw_expert(stage2_real, stage2_imag)
            adjustment = torch.zeros_like(logits)
            adjustment[:, SAW_INDEX : SAW_INDEX + 1] = expert_residual
            logits = logits + adjustment
        if return_diagnostics:
            return logits, {
                "readout": readout_diagnostics,
                "saw_expert": expert_diagnostics,
            }
        return logits


def _load_h2_head(seed, root, device):
    directory = root / "seed_{}".format(seed) / "H2_PosBCE"
    checkpoint_path = directory / "best_head.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = FrozenF2Head(
        feature_dim=128,
        num_labels=6,
        saw_expert=False,
        expert_weight=0.20,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, checkpoint_path, directory


def _extract_maps(
    backbone,
    base_head,
    main_mean,
    main_std,
    data_path,
    indices,
    normalization,
    device,
    batch_size,
):
    dataset = V73SignalDataset(
        path=data_path,
        indices=np.asarray(indices, dtype=np.int64),
        class_mode="six",
        normalization=normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    temporal = []
    global_features_list = []
    stage2_real = []
    stage2_imag = []
    base_logits = []
    targets = []
    sample_ids = []
    mean_tensor = torch.as_tensor(main_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(main_std, dtype=torch.float32, device=device)
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    maps = backbone.forward_structure_maps(inputs)
                global_features = maps["global"].float()
                normalized_global = (global_features - mean_tensor) / std_tensor
                logits = base_head(normalized_global, normalized_global)
                global_features_list.append(
                    global_features.detach().cpu().to(torch.float16).numpy()
                )
                temporal.append(
                    maps["temporal"].detach().cpu().to(torch.float16).numpy()
                )
                stage2_real.append(
                    maps["stage2_real"].detach().cpu().to(torch.float16).numpy()
                )
                stage2_imag.append(
                    maps["stage2_imag"].detach().cpu().to(torch.float16).numpy()
                )
                base_logits.append(logits.detach().cpu().float().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(
                    np.asarray(batch["sample_index"], dtype=np.int64)
                )
    finally:
        dataset.close()
    return {
        "global": np.concatenate(global_features_list, axis=0),
        "temporal": np.concatenate(temporal, axis=0),
        "stage2_real": np.concatenate(stage2_real, axis=0),
        "stage2_imag": np.concatenate(stage2_imag, axis=0),
        "base_logits": np.concatenate(base_logits, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "sample_ids": np.concatenate(sample_ids, axis=0),
    }


def _variant_specification(variant):
    if variant == "Q1_LabelQuery":
        return {"local": False, "expert": False, "epochs": 40, "batch": 64}
    if variant == "Q2_LocalTopK":
        return {"local": True, "expert": False, "epochs": 30, "batch": 64}
    if variant == "Q3_EarlySawExpert":
        return {"local": True, "expert": True, "epochs": 30, "batch": 32}
    raise ValueError("Unknown structured variant: {}".format(variant))


def _build_addon(variant, previous_checkpoint=None):
    specification = _variant_specification(variant)
    readout = LabelConditionedEvidenceReadout(
        channels=128,
        num_labels=6,
        query_dim=64,
        evidence_dim=64,
        attention_floor=0.20,
        topk=8,
        dropout=0.10,
        enable_local=specification["local"],
        initial_scale=0.05,
    )
    expert = (
        EarlySawFMEnvelopeIFExpert(
            complex_channels=32,
            hidden_channels=32,
            topk=8,
            dropout=0.10,
            initial_scale=0.05,
        )
        if specification["expert"]
        else None
    )
    model = StructuredResidualAddOn(readout, expert)
    if previous_checkpoint is not None:
        previous = torch.load(
            previous_checkpoint, map_location="cpu", weights_only=False
        )
        current = model.state_dict()
        inherited = {
            key: value
            for key, value in previous["model_state_dict"].items()
            if key in current and current[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(inherited, strict=False)
        if unexpected:
            raise ValueError("Unexpected inherited keys: {}".format(unexpected))
        if variant == "Q2_LocalTopK" and not any(
            key.startswith("readout.local_") for key in missing
        ):
            raise ValueError("Q2 did not create new local-path parameters")
        if variant == "Q3_EarlySawExpert" and not any(
            key.startswith("saw_expert.") for key in missing
        ):
            raise ValueError("Q3 did not create new expert parameters")
    if variant == "Q2_LocalTopK":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.readout.named_parameters():
            if name.startswith("local_"):
                parameter.requires_grad_(True)
    elif variant == "Q3_EarlySawExpert":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.saw_expert.parameters():
            parameter.requires_grad_(True)
    return model, specification


def _set_training_mode(model, variant):
    model.train()
    if variant == "Q2_LocalTopK":
        model.readout.key_projection.eval()
        model.readout.query_projection.eval()
    elif variant == "Q3_EarlySawExpert":
        model.readout.eval()
        model.saw_expert.train()


def _addon_logits(model, arrays, device, batch_size=64):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(arrays["targets"]), int(batch_size)):
            stop = start + int(batch_size)
            temporal = torch.from_numpy(arrays["temporal"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            real = torch.from_numpy(arrays["stage2_real"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            imag = torch.from_numpy(arrays["stage2_imag"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            base = torch.from_numpy(arrays["base_logits"][start:stop]).to(device)
            outputs.append(model(temporal, real, imag, base).cpu().float().numpy())
    return np.concatenate(outputs, axis=0)


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train_addon(
    variant,
    model,
    specification,
    train_arrays,
    fit_indices,
    validation_indices,
    train_weak,
    seed,
    device,
    args,
):
    variant_seed = 91000 + int(seed) + VARIANTS.index(variant) * 1000
    torch.manual_seed(variant_seed)
    np.random.seed(variant_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(variant_seed)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters for {}".format(variant))
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit_targets = train_arrays["targets"][fit_indices]
    positives = fit_targets.sum(axis=0).astype(np.float64)
    negatives = len(fit_targets) - positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(
            negatives / np.maximum(positives, 1.0),
            dtype=torch.float32,
            device=device,
        )
    )
    dataset = TensorDataset(
        torch.from_numpy(train_arrays["temporal"][fit_indices]),
        torch.from_numpy(train_arrays["stage2_real"][fit_indices]),
        torch.from_numpy(train_arrays["stage2_imag"][fit_indices]),
        torch.from_numpy(train_arrays["base_logits"][fit_indices]),
        torch.from_numpy(fit_targets.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=specification["batch"],
        shuffle=True,
        generator=torch.Generator().manual_seed(variant_seed),
        num_workers=0,
    )
    validation_arrays = {
        key: value[validation_indices] for key, value in train_arrays.items()
    }
    validation_targets = train_arrays["targets"][validation_indices]
    validation_weak = train_weak[validation_indices]
    best_state = None
    best_epoch = 0
    best_macro = -float("inf")
    best_weak = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, specification["epochs"] + 1):
        _set_training_mode(model, variant)
        losses = []
        for temporal, real, imag, base, targets in loader:
            temporal = temporal.to(device=device, dtype=torch.float32)
            real = real.to(device=device, dtype=torch.float32)
            imag = imag.to(device=device, dtype=torch.float32)
            base = base.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(temporal, real, imag, base)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_logits = _addon_logits(
            model, validation_arrays, device, batch_size=64
        )
        validation_probability = _sigmoid(validation_logits)
        validation_macro = _macro_aupr(
            validation_targets, validation_probability
        )
        validation_weak_auc = _binary_rank_metrics(
            validation_targets, validation_probability, validation_weak
        )["weak_vs_negative_auroc"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_aupr": validation_macro,
                "validation_weak_vs_negative_auroc": validation_weak_auc,
            }
        )
        improved = validation_macro > best_macro + 1e-8
        tied_better = np.isclose(validation_macro, best_macro) and (
            validation_weak_auc > best_weak + 1e-8
        )
        if improved or tied_better:
            best_macro = validation_macro
            best_weak = validation_weak_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state, strict=True)
    return model, best_epoch, history, variant_seed


def _module_diagnostics(model, arrays, device):
    model.eval()
    entropies = []
    topk_coverages = []
    expert_logits = []
    with torch.no_grad():
        for start in range(0, len(arrays["targets"]), 64):
            stop = start + 64
            temporal = torch.from_numpy(arrays["temporal"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            real = torch.from_numpy(arrays["stage2_real"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            imag = torch.from_numpy(arrays["stage2_imag"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            base = torch.from_numpy(arrays["base_logits"][start:stop]).to(device)
            _, diagnostics = model(
                temporal, real, imag, base, return_diagnostics=True
            )
            readout = diagnostics["readout"]
            entropies.append(readout["attention_entropy"].cpu().numpy())
            if readout["topk_indices"] is not None:
                indices = readout["topk_indices"].cpu().numpy()
                for sample in indices:
                    topk_coverages.append(
                        len(np.unique(sample)) / float(temporal.shape[-1])
                    )
            if diagnostics["saw_expert"] is not None:
                expert_logits.append(
                    diagnostics["saw_expert"]["expert_logit"].cpu().numpy()
                )
    entropy = np.concatenate(entropies, axis=0)
    payload = {
        "attention_entropy_mean_per_label": entropy.mean(axis=0).tolist(),
        "query_scale": torch.tanh(model.readout.query_scale).detach().cpu().tolist(),
        "local_scale": (
            torch.tanh(model.readout.local_scale).detach().cpu().tolist()
            if model.readout.local_scale is not None
            else None
        ),
        "topk_unique_coverage_mean": (
            float(np.mean(topk_coverages)) if topk_coverages else None
        ),
        "saw_expert_scale": (
            float(torch.tanh(model.saw_expert.scale).detach().cpu())
            if model.saw_expert is not None
            else None
        ),
    }
    if expert_logits:
        values = np.concatenate(expert_logits, axis=0)
        payload["saw_expert_logit_mean"] = float(values.mean())
        payload["saw_expert_logit_std"] = float(values.std())
    return payload


def _parameter_counts(model):
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }


def _copy_control(seed, source_dir, output_dir, f2_hash, h2_hash):
    row = dict(_read(source_dir / "variant_summary.json"))
    row["variant"] = "C0_H2"
    row["source_variant"] = "H2_PosBCE"
    row["f2_checkpoint_sha256"] = f2_hash
    row["h2_checkpoint_sha256"] = h2_hash
    row["locked_test_used"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "variant_summary.json", row)
    _write(
        output_dir / "run_receipt.json",
        {
            "status": "completed",
            "variant": "C0_H2",
            "seed": int(seed),
            "source": str(source_dir),
            "f2_checkpoint_sha256": f2_hash,
            "h2_checkpoint_sha256": h2_hash,
            "split_sha256": row["split_sha256"],
            "manifest_sha256": row["manifest_sha256"],
            "locked_test_used": False,
        },
    )
    return row


def _run_seed(args, config, labels, weak_mask, seed, device):
    f2_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    f2_receipt = _read(f2_dir / "run_receipt.json")
    manifest = _read(f2_dir / "longtail_manifest.json")
    backbone, f2_checkpoint_path = _load_backbone(
        f2_dir, F2_MODEL_NAME, device
    )
    base_head, h2_checkpoint, h2_checkpoint_path, h2_source_dir = _load_h2_head(
        seed, args.frozen_head_root, device
    )
    plan = build_run_plan(
        config,
        loss_name="db",
        seed=seed,
        max_train_samples=0,
        max_eval_samples=args.max_eval_samples,
    )
    if plan["split_sha256"] != f2_receipt["split_sha256"]:
        raise ValueError("Split hash drifted for seed {}".format(seed))
    role_ids = {
        "train": np.asarray(manifest["selected_indices"], dtype=np.int64),
        "early": np.asarray(plan["early_stop_indices"], dtype=np.int64),
        "calibration": np.asarray(plan["calibration_indices"], dtype=np.int64),
        "threshold": np.asarray(plan["threshold_indices"], dtype=np.int64),
    }
    extracted = {}
    for role, indices in role_ids.items():
        extracted[role] = _extract_maps(
            backbone,
            base_head,
            h2_checkpoint["main_mean"],
            h2_checkpoint["main_std"],
            _data_path(config),
            indices,
            config["normalization"],
            device,
            args.feature_batch_size,
        )
        if not np.array_equal(extracted[role]["sample_ids"], indices):
            raise ValueError("Feature/sample alignment failed for {}".format(role))
        if not np.array_equal(extracted[role]["targets"], labels[indices]):
            raise ValueError("Target/sample alignment failed for {}".format(role))
    rng = np.random.default_rng(57000 + int(seed))
    order = rng.permutation(len(role_ids["train"]))
    split = int(round(len(order) * 0.8))
    fit_indices = order[:split]
    validation_indices = order[split:]
    weak_by_role = {
        role: weak_mask[role_ids[role]]
        for role in ("early", "calibration", "threshold")
    }
    targets_by_role = {
        role: extracted[role]["targets"]
        for role in ("early", "calibration", "threshold")
    }
    f2_hash = _sha256(f2_checkpoint_path)
    h2_hash = _sha256(h2_checkpoint_path)
    seed_root = args.output_root / "seed_{}".format(seed)
    rows = [
        _copy_control(
            seed,
            h2_source_dir,
            seed_root / "C0_H2",
            f2_hash,
            h2_hash,
        )
    ]
    previous_checkpoint = None
    for variant in VARIANTS[1:]:
        print("seed={} variant={}".format(seed, variant), flush=True)
        output_dir = seed_root / variant
        summary_path = output_dir / "variant_summary.json"
        checkpoint_path = output_dir / "best_addon.pth"
        if summary_path.exists() and checkpoint_path.exists() and not args.overwrite:
            rows.append(_read(summary_path))
            previous_checkpoint = checkpoint_path
            continue
        model, specification = _build_addon(variant, previous_checkpoint)
        if args.smoke_epochs > 0:
            specification["epochs"] = int(args.smoke_epochs)
        model, best_epoch, history, train_seed = _train_addon(
            variant,
            model,
            specification,
            extracted["train"],
            fit_indices,
            validation_indices,
            weak_mask[role_ids["train"]],
            seed,
            device,
            args,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "variant": variant,
                "best_epoch": best_epoch,
                "train_seed": train_seed,
                "specification": specification,
            },
            checkpoint_path,
        )
        _write_history(output_dir / "history.csv", history)
        logits_by_role = {
            role: _addon_logits(model, extracted[role], device)
            for role in ("early", "calibration", "threshold")
        }
        row = _evaluate_variant(
            variant,
            logits_by_role,
            targets_by_role,
            weak_by_role,
            output_dir,
            args,
        )
        row.update(
            {
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "train_seed": int(train_seed),
                "f2_checkpoint_sha256": f2_hash,
                "h2_checkpoint_sha256": h2_hash,
                "addon_checkpoint_sha256": _sha256(checkpoint_path),
                "split_sha256": f2_receipt["split_sha256"],
                "manifest_sha256": f2_receipt["manifest_sha256"],
                "parameter_counts": _parameter_counts(model),
                "module_diagnostics": _module_diagnostics(
                    model, extracted["early"], device
                ),
                "locked_test_used": False,
            }
        )
        _write(summary_path, row)
        _write(
            output_dir / "run_receipt.json",
            {
                "status": "completed",
                "variant": variant,
                "seed": int(seed),
                "f2_checkpoint_sha256": f2_hash,
                "h2_checkpoint_sha256": h2_hash,
                "addon_checkpoint_sha256": row["addon_checkpoint_sha256"],
                "split_sha256": row["split_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "locked_test_used": False,
            },
        )
        rows.append(row)
        previous_checkpoint = checkpoint_path
    return rows


def _aggregate(rows):
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                101000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(HEAD_METRICS)
        }
    return aggregate


def _comparison(rows, candidate, baseline, seed_offset):
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        candidate_row = next(
            row for row in rows if row["seed"] == seed and row["variant"] == candidate
        )
        baseline_row = next(
            row for row in rows if row["seed"] == seed and row["variant"] == baseline
        )
        paired.append(
            {
                "seed": int(seed),
                "delta": {
                    metric: candidate_row[metric] - baseline_row[metric]
                    for metric in HEAD_METRICS
                },
            }
        )
    aggregate = {
        metric: _mean_ci(
            [row["delta"][metric] for row in paired], seed_offset + index
        )
        for index, metric in enumerate(HEAD_METRICS)
    }
    for metric in (
        "weak_vs_negative_auroc",
        "weak_component_recall",
        "saw_weak_vs_negative_auroc",
        "saw_weak_recall",
    ):
        aggregate["{}_positive_seeds".format(metric)] = sum(
            row["delta"][metric] > 0 for row in paired
        )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "per_seed": paired,
        "aggregate": aggregate,
    }


def _gate_q1(comparison):
    delta = comparison["aggregate"]
    checks = {
        "macro_noninferior": delta["macro_aupr"]["mean"] >= -0.003,
        "weak_auroc_gain": delta["weak_vs_negative_auroc"]["mean"] >= 0.010,
        "weak_recall_gain": delta["weak_component_recall"]["mean"] >= 0.020,
        "hamming_noninferior": delta["hamming_loss"]["mean"] <= 0.005,
        "fpr_delta": delta["negative_fpr"]["mean"] <= 0.020,
        "weak_auroc_two_seeds": delta[
            "weak_vs_negative_auroc_positive_seeds"
        ] >= 2,
        "weak_recall_two_seeds": delta[
            "weak_component_recall_positive_seeds"
        ] >= 2,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _gate_q2(comparison):
    delta = comparison["aggregate"]
    checks = {
        "macro_noninferior": delta["macro_aupr"]["mean"] >= -0.003,
        "weak_auroc_noninferior": delta["weak_vs_negative_auroc"]["mean"] >= 0.0,
        "weak_recall_gain": delta["weak_component_recall"]["mean"] >= 0.020,
        "fpr_delta": delta["negative_fpr"]["mean"] <= 0.010,
        "weak_recall_two_seeds": delta[
            "weak_component_recall_positive_seeds"
        ] >= 2,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _gate_q3(comparison):
    delta = comparison["aggregate"]
    checks = {
        "macro_noninferior": delta["macro_aupr"]["mean"] >= -0.003,
        "saw_aupr_noninferior": delta["saw_aupr"]["mean"] >= 0.0,
        "saw_weak_auroc_gain": (
            delta["saw_weak_vs_negative_auroc"]["mean"] >= 0.020
        ),
        "saw_weak_recall_gain": delta["saw_weak_recall"]["mean"] >= 0.030,
        "saw_fpr_delta": delta["saw_negative_fpr"]["mean"] <= 0.020,
        "saw_weak_auroc_two_seeds": delta[
            "saw_weak_vs_negative_auroc_positive_seeds"
        ] >= 2,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _gate_full(comparison, aggregate):
    delta = comparison["aggregate"]
    q3 = aggregate["Q3_EarlySawExpert"]
    checks = {
        "macro_noninferior": delta["macro_aupr"]["mean"] >= -0.005,
        "weak_auroc_gain": delta["weak_vs_negative_auroc"]["mean"] >= 0.015,
        "weak_recall_gain": delta["weak_component_recall"]["mean"] >= 0.030,
        "hamming_noninferior": delta["hamming_loss"]["mean"] <= 0.005,
        "negative_fpr_absolute": q3["negative_fpr"]["mean"] <= 0.25,
        "negative_fpr_delta": delta["negative_fpr"]["mean"] <= 0.020,
        "saw_aupr_noninferior": delta["saw_aupr"]["mean"] >= 0.0,
        "saw_weak_auroc_gain": (
            delta["saw_weak_vs_negative_auroc"]["mean"] >= 0.020
        ),
        "weak_auroc_two_seeds": delta[
            "weak_vs_negative_auroc_positive_seeds"
        ] >= 2,
        "weak_recall_two_seeds": delta[
            "weak_component_recall_positive_seeds"
        ] >= 2,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    labels = select_labels(read_labels(data_path), "six")
    values79, _ = read_parameters(data_path)
    weak_mask = compute_weak_component_mask(
        labels, _component_power(values79), args.weak_threshold_db
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        rows.extend(_run_seed(args, config, labels, weak_mask, seed, device))
    aggregate = _aggregate(rows)
    comparisons = {
        "q1_vs_c0": _comparison(rows, "Q1_LabelQuery", "C0_H2", 102000),
        "q2_vs_q1": _comparison(rows, "Q2_LocalTopK", "Q1_LabelQuery", 102100),
        "q3_vs_q2": _comparison(rows, "Q3_EarlySawExpert", "Q2_LocalTopK", 102200),
        "q3_vs_c0": _comparison(rows, "Q3_EarlySawExpert", "C0_H2", 102300),
    }
    gates = {
        "q1": _gate_q1(comparisons["q1_vs_c0"]),
        "q2": _gate_q2(comparisons["q2_vs_q1"]),
        "q3": _gate_q3(comparisons["q3_vs_q2"]),
        "full": _gate_full(comparisons["q3_vs_c0"], aggregate),
    }
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "device": str(device),
        "protocol": {
            "seeds": list(args.seeds),
            "variants": list(VARIANTS),
            "base": "frozen F2 + frozen H2 pos-weight BCE",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "max_negative_fpr": args.max_negative_fpr,
            "weak_threshold_db": args.weak_threshold_db,
        },
        "per_run": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "gates": gates,
        "unfreeze_decision": (
            "eligible_for_last_stage_unfreeze"
            if gates["full"]["passed"]
            else "stop_before_unfreeze"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(
        args.output_root / "route_a_dfms_iq_label_local_saw_summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_dfms_iq_structure_screen",
    )
    parser.add_argument(
        "--frozen-head-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_dfms_iq_frozen_heads",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_dfms_iq_label_local_saw",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--smoke-epochs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
