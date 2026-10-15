"""P3n-A fair B1/B2/B3 training on grouped regime-power identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import ConcatDataset, DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.multitask import (
    PARAMETER_COLUMNS,
    ParameterNormalizer,
    build_multitask_model,
    masked_parameter_smooth_l1,
    parameter_metrics,
    reshape_selected_parameters,
)
from cj0724.multitask_benchmark import _combination_ids, read_selected_parameters
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import worst_combo_predicted_gate_nmae
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3f_shared_isrj_router import classification_metrics
from cj0724.p3g_cascaded_isrj_prefilter import collect_frozen_route
from cj0724.paper_core_evidence import gated_metrics, load_normalizer
from cj0724.train import get_device, multilabel_metrics, set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr, optimize_class_thresholds


VARIANTS = ("B1_original_bce", "B2_hard_negative", "B3_conditional_margin")
MODEL_SEEDS = (2041, 2042, 2043)
MFT_INDEX = 3
ISRJ_INDEX = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def conditional_margin_loss(
    logits: torch.Tensor, labels: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Symmetric margin only for MFT-only or ISRJ-only samples."""
    mft = labels[:, MFT_INDEX] > 0.5
    isrj = labels[:, ISRJ_INDEX] > 0.5
    mft_only = mft & ~isrj
    isrj_only = isrj & ~mft
    losses = []
    if bool(mft_only.any()):
        difference = logits[mft_only, MFT_INDEX] - logits[mft_only, ISRJ_INDEX]
        losses.append(F.softplus(float(margin) - difference))
    if bool(isrj_only.any()):
        difference = logits[isrj_only, ISRJ_INDEX] - logits[isrj_only, MFT_INDEX]
        losses.append(F.softplus(float(margin) - difference))
    if not losses:
        return logits.sum() * 0.0
    return torch.cat(losses).mean()


def variant_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    variant: str,
    hard_negative_weight: float = 3.0,
    xci_weight: float = 0.1,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown P3n variant {variant!r}")
    element = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if variant == "B2_hard_negative":
        weights = torch.ones_like(element)
        hard = (labels[:, MFT_INDEX] > 0.5) & (labels[:, ISRJ_INDEX] < 0.5)
        weights[hard, ISRJ_INDEX] = float(hard_negative_weight)
        bce = (element * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        bce = element.mean()
    margin_loss = (
        conditional_margin_loss(logits, labels, margin)
        if variant == "B3_conditional_margin"
        else logits.sum() * 0.0
    )
    total = bce + (float(xci_weight) * margin_loss if variant == "B3_conditional_margin" else 0.0)
    return total, {"bce": bce, "conditional_margin": margin_loss}


def make_identity_roles(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Allocate 7/1/1/1 identities from each of the 63 combinations."""
    combo = _combination_ids(labels)
    rng = np.random.default_rng(int(seed))
    parts = {role: [] for role in ("fit", "early", "calibration", "selection")}
    for combination in np.unique(combo):
        indices = rng.permutation(np.flatnonzero(combo == combination))
        if len(indices) != 10:
            raise ValueError(f"P3n expects 10 identities per combination, got {len(indices)}")
        parts["fit"].append(indices[:7])
        parts["early"].append(indices[7:8])
        parts["calibration"].append(indices[8:9])
        parts["selection"].append(indices[9:10])
    return {
        role: np.sort(np.concatenate(values)).astype(np.int64)
        for role, values in parts.items()
    }


def ordered_paths(files_by_regime: dict[str, dict[int, Path]]) -> list[Path]:
    return [files_by_regime[regime][power] for regime in REGIMES for power in POWERS]


def build_concat_dataset(paths: list[Path], indices: np.ndarray) -> ConcatDataset:
    return ConcatDataset([V73IQDataset(path, indices, PARAMETER_COLUMNS) for path in paths])


def close_concat_dataset(dataset: ConcatDataset) -> None:
    for part in dataset.datasets:
        if isinstance(part, V73IQDataset):
            part.close()


def fit_multi_file_normalizer(
    paths: list[Path], indices: np.ndarray, labels: np.ndarray
) -> ParameterNormalizer:
    values_parts, valid_parts = [], []
    for path in paths:
        values, mask = read_selected_parameters(path, indices)
        values_parts.append(values)
        valid_parts.append(mask & (labels[indices, :, None] > 0))
    values = np.concatenate(values_parts)
    valid = np.concatenate(valid_parts)
    return ParameterNormalizer().fit(torch.from_numpy(values), torch.from_numpy(valid))


def role_metadata(
    files_by_regime: dict[str, dict[int, Path]],
    metadata: dict[str, dict[str, np.ndarray]],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    parts = {key: [] for key in ("identity", "regime", "power", "sample_index")}
    for regime_index, regime in enumerate(REGIMES):
        for power in POWERS:
            parts["identity"].append(metadata[regime]["identities"][indices])
            parts["regime"].append(np.full(len(indices), regime_index, dtype=np.int8))
            parts["power"].append(np.full(len(indices), power, dtype=np.int16))
            parts["sample_index"].append(indices.astype(np.int64))
    return {key: np.concatenate(values) for key, values in parts.items()}


def collect_outputs(
    model,
    loader: DataLoader,
    device: torch.device,
    normalizer: ParameterNormalizer,
    variant: str,
    parameter_weight: float,
    hard_negative_weight: float,
    xci_weight: float,
    margin: float,
) -> dict[str, object]:
    model.eval()
    parts = {name: [] for name in ("logits", "labels", "prediction", "target", "mask")}
    totals, classifications, bces, margins, parameters = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
            mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
            normalized_target = normalizer.transform(target)
            autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
            with autocast:
                logits, prediction = model(iq)
                cls, cls_parts = variant_classification_loss(
                    logits, labels, variant, hard_negative_weight, xci_weight, margin
                )
                parameter = masked_parameter_smooth_l1(
                    prediction, normalized_target, labels, mask
                )
                total = cls + float(parameter_weight) * parameter
            parts["logits"].append(logits.float().cpu())
            parts["labels"].append(labels.float().cpu())
            parts["prediction"].append(normalizer.inverse_transform(prediction.float().cpu()))
            parts["target"].append(target.float().cpu())
            parts["mask"].append(mask.cpu())
            totals.append(float(total)); classifications.append(float(cls))
            bces.append(float(cls_parts["bce"])); margins.append(float(cls_parts["conditional_margin"]))
            parameters.append(float(parameter))
    return {
        "logits": torch.cat(parts["logits"]),
        "labels": torch.cat(parts["labels"]),
        "prediction": torch.cat(parts["prediction"]).numpy(),
        "target": torch.cat(parts["target"]).numpy(),
        "mask": torch.cat(parts["mask"]).numpy().astype(bool),
        "total_loss": float(np.mean(totals)),
        "classification_loss": float(np.mean(classifications)),
        "bce_loss": float(np.mean(bces)),
        "conditional_margin_loss": float(np.mean(margins)),
        "parameter_loss": float(np.mean(parameters)),
    }


def train_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    normalizer: ParameterNormalizer,
    variant: str,
    optimizer,
    scaler,
    args,
) -> dict[str, float]:
    model.train()
    totals, classifications, margins, parameters = [], [], [], []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
        mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
        normalized_target = normalizer.transform(target)
        autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with autocast:
            logits, prediction = model(iq)
            cls, cls_parts = variant_classification_loss(
                logits, labels, variant, args.hard_negative_weight, args.xci_weight, args.margin
            )
            parameter = masked_parameter_smooth_l1(
                prediction, normalized_target, labels, mask
            )
            total = cls + float(args.parameter_weight) * parameter
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer); scaler.update()
        else:
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm); optimizer.step()
        totals.append(float(total.detach())); classifications.append(float(cls.detach()))
        margins.append(float(cls_parts["conditional_margin"].detach())); parameters.append(float(parameter.detach()))
    return {
        "total_loss": float(np.mean(totals)),
        "classification_loss": float(np.mean(classifications)),
        "conditional_margin_loss": float(np.mean(margins)),
        "parameter_loss": float(np.mean(parameters)),
    }


def output_parameter_metrics(outputs: dict[str, object], normalizer: ParameterNormalizer) -> dict[str, object]:
    labels = outputs["labels"].numpy()
    valid = outputs["mask"] & (labels[:, :, None] > 0.5)
    return parameter_metrics(outputs["prediction"], outputs["target"], valid, normalizer)


def evaluate_outputs(
    outputs: dict[str, object],
    thresholds: np.ndarray,
    metadata: dict[str, np.ndarray],
    normalizer: ParameterNormalizer,
) -> dict[str, object]:
    logits = outputs["logits"]
    labels_tensor = outputs["labels"]
    labels = labels_tensor.numpy()
    probabilities = torch.sigmoid(logits).numpy()
    classification = classification_metrics(
        probabilities, labels, thresholds, metadata["regime"], metadata["power"]
    )
    predicted = probabilities[:, ISRJ_INDEX] >= thresholds[ISRJ_INDEX]
    truth = labels[:, ISRJ_INDEX].astype(bool)
    per_power = {}
    for power in POWERS:
        selected = metadata["power"] == power
        per_power[str(power)] = {
            "isrj_recall": float(predicted[selected & truth].mean()),
            "isrj_fpr": float(predicted[selected & ~truth].mean()),
        }
    scale = normalizer.scale.numpy()
    record = {
        "sample_indices": np.arange(len(labels), dtype=np.int64),
        "labels": labels.astype(np.uint8),
        "probabilities": probabilities,
        "prediction": outputs["prediction"],
        "target": outputs["target"],
        "mask": outputs["mask"],
        "thresholds": thresholds,
        "scale": scale,
    }
    gated = gated_metrics(record)
    gated["worst_combo_predicted_gate_nmae"] = worst_combo_predicted_gate_nmae(record)
    base = multilabel_metrics(logits, labels_tensor, thresholds)
    base.update({
        "macro_aupr": float(average_precision_score(labels, probabilities, average="macro")),
        "isrj_recall": float(predicted[truth].mean()),
        "isrj_fpr": float(predicted[~truth].mean()),
        "maximum_cell_fpr": float(max(row["fpr"] for row in classification["cells"])),
        "parameter": output_parameter_metrics(outputs, normalizer),
        "predicted_gate": gated,
        "per_power": per_power,
        "ultra_close_cells": [row for row in classification["cells"] if row["regime"] == "ultra_close"],
        "cells": classification["cells"],
        "losses": {key: float(outputs[key]) for key in (
            "total_loss", "classification_loss", "bce_loss", "conditional_margin_loss", "parameter_loss"
        )},
    })
    return base


def identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def discover_prior_identity_files(project: Path, ignored_seeds: set[int]) -> list[Path]:
    selected = []
    for directory in sorted((project / "generated_data").glob("*_rechunked")):
        match = re.search(r"seed(\d{9,10})", directory.name)
        if not match or int(match.group(1)) in ignored_seeds:
            continue
        path = next(directory.rglob("closedset_6class_train.mat"), None)
        if path is not None:
            selected.append(path)
    return selected


def audit_identities(
    project: Path,
    train_identity: np.ndarray,
    development_identity: np.ndarray,
    ignored_seeds: set[int],
    original_data_file: Path,
) -> dict[str, object]:
    if np.intersect1d(train_identity, development_identity).size:
        raise RuntimeError("P3n train/development identity overlap")
    prior_files = [original_data_file] + discover_prior_identity_files(project, ignored_seeds)
    rows, maximum = [], 0
    for path in prior_files:
        previous, _ = read_identity_metadata(path)
        train_overlap = int(np.intersect1d(train_identity, previous).size)
        development_overlap = int(np.intersect1d(development_identity, previous).size)
        maximum = max(maximum, train_overlap, development_overlap)
        rows.append({"path": str(path.resolve()), "train_overlap": train_overlap, "development_overlap": development_overlap})
    if maximum:
        raise RuntimeError("P3n overlaps a prior identity role")
    return {
        "train_unique": int(len(np.unique(train_identity))),
        "development_unique": int(len(np.unique(development_identity))),
        "train_sha256": identity_hash(train_identity),
        "development_sha256": identity_hash(development_identity),
        "cross_role_overlap": 0,
        "prior_files_checked": len(rows),
        "maximum_prior_overlap": maximum,
        "rows": rows,
    }


def add_route_summary(metric: dict[str, object], probability: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, regime: np.ndarray, power: np.ndarray) -> dict[str, object]:
    classification = classification_metrics(probability, labels, thresholds, regime, power)
    pred = probability[:, ISRJ_INDEX] >= thresholds[ISRJ_INDEX]
    truth = labels[:, ISRJ_INDEX].astype(bool)
    return {
        "macro_aupr": classification["macro_aupr"],
        "isrj_recall": float(pred[truth].mean()),
        "isrj_fpr": float(pred[~truth].mean()),
        "maximum_cell_fpr": float(max(row["fpr"] for row in classification["cells"])),
        "ultra_close_cells": [row for row in classification["cells"] if row["regime"] == "ultra_close"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3n_cross_interference_train_seed760000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3n_cross_interference_dev_seed770000000_{regime}_rechunked")
    parser.add_argument("--preregister", default="configs/p3n_cross_interference_backbone_preregister_20260801.json")
    parser.add_argument("--method-freeze", default="configs/p3n_cross_interference_backbone_method_freeze_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3n_cross_interference_backbone_development_20260801")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=63); parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--parameter-weight", type=float, default=0.1); parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--xci-weight", type=float, default=0.1); parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0); parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    preregister = project / args.preregister; freeze_path = project / args.method_freeze
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(preregister): raise RuntimeError("P3n preregister hash mismatch")
    if freeze["implementation_sha256"] != sha256_file(Path(__file__).resolve()): raise RuntimeError("P3n implementation hash mismatch")
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    device = get_device(args.device)
    train_roots = {regime: transition_files(project / args.train_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES}
    development_roots = {regime: transition_files(project / args.development_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES}
    metadata = {"train": {}, "development": {}}; references = {}
    for role, roots in (("train", train_roots), ("development", development_roots)):
        reference = None
        for regime in REGIMES:
            audit_power_alignment(roots[regime]); identities, labels = read_identity_metadata(roots[regime][POWERS[0]])
            if reference is None: reference = identities
            elif not np.array_equal(reference, identities): raise RuntimeError(f"P3n {role} regimes are not identity paired")
            metadata[role][regime] = {"identities": identities, "labels": labels}
        references[role] = reference
    train_labels = metadata["train"]["ID_control"]["labels"]
    development_labels = metadata["development"]["ID_control"]["labels"]
    if not np.array_equal(train_labels, development_labels): raise RuntimeError("P3n label contracts differ across identity seeds")
    roles = make_identity_roles(train_labels, args.split_seed)
    original_protocol = json.loads((project / args.run_root / "protocol.json").read_text(encoding="utf-8"))
    audit = audit_identities(project, references["train"], references["development"], {760000000, 770000000}, Path(original_protocol["data_file"]))
    (output / "identity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    np.savez_compressed(output / "identity_roles.npz", **roles)
    train_paths, development_paths = ordered_paths(train_roots), ordered_paths(development_roots)
    normalizer = fit_multi_file_normalizer(train_paths, roles["fit"], train_labels)
    role_meta = {role: role_metadata(train_roots, metadata["train"], indices) for role, indices in roles.items()}
    development_indices = np.arange(len(development_labels), dtype=np.int64)
    development_meta = role_metadata(development_roots, metadata["development"], development_indices)
    protocol = vars(args) | {
        "device_resolved": str(device), "variants": list(VARIANTS), "model_seeds": list(MODEL_SEEDS),
        "role_identity_counts": {role: int(len(indices)) for role, indices in roles.items()},
        "role_view_counts": {role: int(len(indices) * len(REGIMES) * len(POWERS)) for role, indices in roles.items()},
        "normalizer": normalizer.state_dict(), "identity_audit_summary": {key: audit[key] for key in audit if key != "rows"},
        "preregister_sha256": freeze["preregister_sha256"], "implementation_sha256": freeze["implementation_sha256"],
        "locked_test_used": False, "test_generated": False, "test_opened": False,
    }
    (output / "protocol.json").write_text(json.dumps(_jsonable(protocol), indent=2), encoding="utf-8")
    datasets = {role: build_concat_dataset(train_paths, indices) for role, indices in roles.items()}
    development_dataset = build_concat_dataset(development_paths, development_indices)
    results = []
    try:
        for variant in VARIANTS:
            for seed in MODEL_SEEDS:
                print(f"START variant={variant} seed={seed}", flush=True); set_seed(seed)
                generator = torch.Generator().manual_seed(seed)
                loaders = {
                    role: DataLoader(dataset, batch_size=args.batch_size, shuffle=role == "fit", generator=generator if role == "fit" else None, num_workers=args.workers, pin_memory=device.type == "cuda")
                    for role, dataset in datasets.items()
                }
                development_loader = DataLoader(development_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
                model = build_multitask_model("mlcnn_iq_grouped_task_routed", 24, 0.2, 256, 64).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None
                run_dir = output / variant / f"seed_{seed}"; run_dir.mkdir(parents=True)
                best_value, best_epoch, history = float("inf"), 0, []
                for epoch in range(1, args.epochs + 1):
                    lr = _set_warmup_cosine_lr(optimizer, args.learning_rate, epoch, args.epochs, args.warmup_epochs)
                    train = train_epoch(model, loaders["fit"], device, normalizer, variant, optimizer, scaler, args)
                    early_outputs = collect_outputs(model, loaders["early"], device, normalizer, variant, args.parameter_weight, args.hard_negative_weight, args.xci_weight, args.margin)
                    early_parameter = output_parameter_metrics(early_outputs, normalizer)
                    value = float(early_parameter["oracle_macro_target_nmae"])
                    row = {"epoch": epoch, "lr": lr, "train": train, "early_parameter_macro_nmae": value, "early_macro_aupr": float(average_precision_score(early_outputs["labels"].numpy(), torch.sigmoid(early_outputs["logits"]).numpy(), average="macro"))}
                    history.append(row); print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
                    if value < best_value:
                        best_value, best_epoch = value, epoch; torch.save(model.state_dict(), run_dir / "best_parameter.pt")
                model.load_state_dict(torch.load(run_dir / "best_parameter.pt", map_location=device, weights_only=True)); model.eval()
                calibration_outputs = collect_outputs(model, loaders["calibration"], device, normalizer, variant, args.parameter_weight, args.hard_negative_weight, args.xci_weight, args.margin)
                thresholds = optimize_class_thresholds(torch.sigmoid(calibration_outputs["logits"]).numpy(), calibration_outputs["labels"].numpy())
                selection_outputs = collect_outputs(model, loaders["selection"], device, normalizer, variant, args.parameter_weight, args.hard_negative_weight, args.xci_weight, args.margin)
                development_outputs = collect_outputs(model, development_loader, device, normalizer, variant, args.parameter_weight, args.hard_negative_weight, args.xci_weight, args.margin)
                selection_metrics = evaluate_outputs(selection_outputs, thresholds, role_meta["selection"], normalizer)
                development_metrics = evaluate_outputs(development_outputs, thresholds, development_meta, normalizer)
                run = {"variant": variant, "seed": seed, "best_epoch": best_epoch, "best_early_parameter_macro_nmae": best_value, "thresholds": thresholds, "selection": selection_metrics, "development": development_metrics}
                (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
                (run_dir / "result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
                np.savez_compressed(run_dir / "development_predictions.npz", identity=development_meta["identity"], regime=development_meta["regime"], power=development_meta["power"], sample_index=development_meta["sample_index"], labels=development_outputs["labels"].numpy(), probabilities=torch.sigmoid(development_outputs["logits"]).numpy(), prediction=development_outputs["prediction"], target=development_outputs["target"], mask=development_outputs["mask"].astype(np.uint8), thresholds=thresholds, scale=normalizer.scale.numpy())
                results.append(run)
                print(json.dumps({"variant": variant, "seed": seed, "isrj_recall": development_metrics["isrj_recall"], "maximum_cell_fpr": development_metrics["maximum_cell_fpr"], "macro_aupr": development_metrics["macro_aupr"], "oracle_parameter_nmae": development_metrics["predicted_gate"]["oracle_parameter_nmae"], "predicted_gate_nmae": development_metrics["predicted_gate"]["predicted_gate_parameter_nmae"]}), flush=True)
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
    finally:
        for dataset in datasets.values(): close_concat_dataset(dataset)
        close_concat_dataset(development_dataset)
    indexed = {(row["variant"], int(row["seed"])): row for row in results}
    comparisons = []
    core_wins = {name: [] for name in ("maximum_cell_fpr", "ultra_close_max_fpr", "isrj_recall", "macro_aupr")}
    for seed in MODEL_SEEDS:
        b1, b2, b3 = (indexed[(variant, seed)]["development"] for variant in VARIANTS)
        ultra = lambda metric: max(row["fpr"] for row in metric["ultra_close_cells"])
        comparison = {
            "seed": seed,
            "b3_isrj_recall_delta_vs_b1": b3["isrj_recall"] - b1["isrj_recall"],
            "b3_maximum_cell_fpr_delta_vs_b1": b3["maximum_cell_fpr"] - b1["maximum_cell_fpr"],
            "b3_ultra_close_max_fpr_delta_vs_b1": ultra(b3) - ultra(b1),
            "b3_macro_aupr_delta_vs_b1": b3["macro_aupr"] - b1["macro_aupr"],
            "b3_oracle_parameter_nmae_ratio_vs_b1": b3["predicted_gate"]["oracle_parameter_nmae"] / b1["predicted_gate"]["oracle_parameter_nmae"],
            "b3_predicted_gate_nmae_ratio_vs_b1": b3["predicted_gate"]["predicted_gate_parameter_nmae"] / b1["predicted_gate"]["predicted_gate_parameter_nmae"],
            "b3_miss_penalty_delta_vs_b1": b3["predicted_gate"]["miss_penalty_parameter_nmae"] - b1["predicted_gate"]["miss_penalty_parameter_nmae"],
            "b3_worst_combo_delta_vs_b1": b3["predicted_gate"]["worst_combo_predicted_gate_nmae"] - b1["predicted_gate"]["worst_combo_predicted_gate_nmae"],
            "b3_each_power_recall_delta_vs_b1": {str(power): b3["per_power"][str(power)]["isrj_recall"] - b1["per_power"][str(power)]["isrj_recall"] for power in POWERS},
        }
        comparisons.append(comparison)
        core_wins["maximum_cell_fpr"].append(b3["maximum_cell_fpr"] < b2["maximum_cell_fpr"])
        core_wins["ultra_close_max_fpr"].append(ultra(b3) < ultra(b2))
        core_wins["isrj_recall"].append(b3["isrj_recall"] > b2["isrj_recall"])
        core_wins["macro_aupr"].append(b3["macro_aupr"] > b2["macro_aupr"])
    stable_core_wins = [name for name, values in core_wins.items() if all(values)]
    b3_runs = [indexed[("B3_conditional_margin", seed)]["development"] for seed in MODEL_SEEDS]
    gates = {
        "identity_overlap_zero": audit["maximum_prior_overlap"] == 0 and audit["cross_role_overlap"] == 0,
        "b3_each_seed_isrj_recall_delta_vs_b1_nonnegative": all(row["b3_isrj_recall_delta_vs_b1"] >= 0 for row in comparisons),
        "b3_maximum_cell_fpr_at_most_0p10": all(row["maximum_cell_fpr"] <= 0.10 for row in b3_runs),
        "b3_each_power_recall_drop_vs_b1_at_most_0p02": all(min(row["b3_each_power_recall_delta_vs_b1"].values()) >= -0.02 for row in comparisons),
        "b3_each_seed_macro_aupr_delta_vs_b1_at_least_minus_0p005": all(row["b3_macro_aupr_delta_vs_b1"] >= -0.005 for row in comparisons),
        "b3_each_seed_oracle_parameter_nmae_ratio_at_most_1p01": all(row["b3_oracle_parameter_nmae_ratio_vs_b1"] <= 1.01 for row in comparisons),
        "b3_each_seed_predicted_gate_nmae_ratio_at_most_1p01": all(row["b3_predicted_gate_nmae_ratio_vs_b1"] <= 1.01 for row in comparisons),
        "b3_miss_penalty_and_worst_combo_nonworse": all(row["b3_miss_penalty_delta_vs_b1"] <= 0 and row["b3_worst_combo_delta_vs_b1"] <= 0 for row in comparisons),
        "b3_stably_beats_b2_on_at_least_two_core_metrics": len(stable_core_wins) >= 2,
        "three_of_three_direction_consistent": all(row["b3_isrj_recall_delta_vs_b1"] >= 0 and row["b3_maximum_cell_fpr_delta_vs_b1"] <= 0 for row in comparisons),
    }
    passed = bool(all(gates.values()))
    # B0 is computed only after the new methods are fully frozen and evaluated; it is external context, not selection.
    run_root = project / args.run_root; p2c_root = project / args.p2c_root
    old_rows = json.loads((run_root / "summary.json").read_text(encoding="utf-8")); old_protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8")); p2c = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8")); old_normalizer = load_normalizer(old_protocol["normalizer"])
    selected_rows = {int(row["seed"]): row for row in old_rows if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}; freezes = {int(row["seed"]): row for row in p2c["seed_freezes"]}; b0 = []
    for seed in MODEL_SEEDS:
        route = collect_frozen_route(project, development_roots, metadata["development"], seed, selected_rows[seed], freezes[seed], old_normalizer, device, p2c_root)
        thresholds = np.asarray(freezes[seed]["routed_thresholds"], dtype=np.float64)
        b0.append({"seed": seed, "old_main": add_route_summary({}, route["main"], route["labels"], thresholds, route["regime"], route["power"]), "old_p1n": add_route_summary({}, route["full"], route["labels"], thresholds, route["regime"], route["power"])})
    attribution = "conditional_margin_supported" if passed else "P3n-A failed one or more frozen gates; apply preregistered attribution audit without retuning seed770000000"
    summary = {"experiment": "P3n-A cross-interference identifiable backbone", "development_passed": passed, "promotion_gates": gates, "stable_core_wins_vs_B2": stable_core_wins, "comparisons_B3_vs_B1": comparisons, "runs": results, "B0_external_reference": b0, "attribution": attribution, "identity_audit": {key: audit[key] for key in audit if key != "rows"}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "stable_core_wins_vs_B2": stable_core_wins, "attribution": attribution}, indent=2), flush=True)


if __name__ == "__main__":
    main()
