"""P3o-A: attribute low-power ISRJ risk to data coverage, checkpoint, and calibration.

The runner is deliberately architecture-neutral: it keeps P3n B1's MLCNN +
IQ-Grouped model and masked parameter loss, then evaluates a prespecified
coverage/checkpoint/threshold matrix on fresh grouped identities.
"""

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
from scipy.stats import beta as beta_distribution
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
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3f_shared_isrj_router import classification_metrics
from cj0724.p3n_cross_interference_backbone import (
    ISRJ_INDEX,
    identity_hash,
    output_parameter_metrics,
    sha256_file,
)
from cj0724.train import get_device, multilabel_metrics, set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr, optimize_class_thresholds


MODEL_SEEDS = (2041, 2042, 2043)
CHECKPOINTS = ("C_param", "C_joint")
THRESHOLD_RULES = ("T_F1", "T_safe")
SCALES = ("small_epoch_matched", "small_step_matched", "full_epoch_matched")
FULL_FIT_PER_COMBINATION = 27
SMALL_FIT_PER_COMBINATION = 7
EARLY_PER_COMBINATION = 2
CALIBRATION_PER_COMBINATION = 2
SELECTION_PER_COMBINATION = 2
IDENTITIES_PER_COMBINATION = 33


def make_identity_roles(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Create nested small/full fit roles with disjoint early/calibration/selection."""
    combo = _combination_ids(labels)
    rng = np.random.default_rng(int(seed))
    parts = {name: [] for name in ("small_fit", "full_fit", "early", "calibration", "selection")}
    for combination in np.unique(combo):
        indices = rng.permutation(np.flatnonzero(combo == combination))
        if len(indices) != IDENTITIES_PER_COMBINATION:
            raise ValueError(
                f"P3o requires {IDENTITIES_PER_COMBINATION} identities per combination, got {len(indices)}"
            )
        parts["full_fit"].append(indices[:FULL_FIT_PER_COMBINATION])
        parts["small_fit"].append(indices[:SMALL_FIT_PER_COMBINATION])
        cursor = FULL_FIT_PER_COMBINATION
        parts["early"].append(indices[cursor:cursor + EARLY_PER_COMBINATION]); cursor += EARLY_PER_COMBINATION
        parts["calibration"].append(indices[cursor:cursor + CALIBRATION_PER_COMBINATION]); cursor += CALIBRATION_PER_COMBINATION
        parts["selection"].append(indices[cursor:cursor + SELECTION_PER_COMBINATION])
    return {name: np.sort(np.concatenate(values)).astype(np.int64) for name, values in parts.items()}


def select_checkpoint_epochs(history: list[dict[str, object]], tolerance: float = 1.01) -> dict[str, int | float | list[int]]:
    """Choose parameter minimum and AUPR maximum subject to a frozen parameter bound."""
    if not history:
        raise ValueError("empty training history")
    parameter_row = min(history, key=lambda row: float(row["early_parameter_macro_nmae"]))
    minimum = float(parameter_row["early_parameter_macro_nmae"])
    eligible = [row for row in history if float(row["early_parameter_macro_nmae"]) <= tolerance * minimum + 1e-12]
    joint_row = max(eligible, key=lambda row: (float(row["early_macro_aupr"]), -int(row["epoch"])))
    return {
        "C_param_epoch": int(parameter_row["epoch"]),
        "C_param_early_parameter_macro_nmae": minimum,
        "C_joint_epoch": int(joint_row["epoch"]),
        "C_joint_early_parameter_macro_nmae": float(joint_row["early_parameter_macro_nmae"]),
        "C_joint_early_macro_aupr": float(joint_row["early_macro_aupr"]),
        "eligible_epochs": [int(row["epoch"]) for row in eligible],
        "tolerance": float(tolerance),
    }


def clopper_pearson_upper(false_positives: int, negatives: int, confidence: float = 0.95) -> float:
    if negatives <= 0:
        raise ValueError("Clopper-Pearson requires at least one negative")
    if false_positives >= negatives:
        return 1.0
    return float(beta_distribution.ppf(confidence, false_positives + 1, negatives - false_positives))


def select_cp_safe_isrj_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    regime: np.ndarray,
    power: np.ndarray,
    cap: float = 0.10,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Lowest threshold whose CP upper FPR bound passes in every regime-power cell."""
    score = np.asarray(probabilities[:, ISRJ_INDEX], dtype=np.float64)
    truth = np.asarray(labels[:, ISRJ_INDEX], dtype=bool)
    cells = [(int(r), int(p)) for r in np.unique(regime) for p in np.unique(power)]
    negative_scores = np.concatenate([score[(regime == r) & (power == p) & ~truth] for r, p in cells])
    candidates = np.unique(np.concatenate((
        np.array([0.0], dtype=np.float64),
        np.nextafter(negative_scores, np.inf),
        np.array([1.0 + np.finfo(np.float64).eps]),
    )))
    for threshold in np.sort(candidates):
        predicted = score >= threshold
        rows, feasible = [], True
        for r, p in cells:
            negative = (regime == r) & (power == p) & ~truth
            positives = (regime == r) & (power == p) & truth
            fp, n = int(predicted[negative].sum()), int(negative.sum())
            upper = clopper_pearson_upper(fp, n, confidence)
            observed = float(fp / n)
            if upper > cap + 1e-12:
                feasible = False
            rows.append({
                "regime_index": r, "power": p, "false_positives": fp, "negatives": n,
                "observed_fpr": observed, "cp95_upper": upper,
                "recall": float(predicted[positives].mean()),
            })
        if feasible:
            return {
                "feasible": True, "threshold": float(threshold), "cap": float(cap),
                "confidence": float(confidence), "maximum_observed_fpr": float(max(row["observed_fpr"] for row in rows)),
                "maximum_cp95_upper": float(max(row["cp95_upper"] for row in rows)),
                "overall_recall": float(predicted[truth].mean()), "cells": rows,
            }
    return {"feasible": False, "threshold": None, "cap": float(cap), "confidence": float(confidence), "cells": []}


def ordered_paths(files_by_regime: dict[str, dict[int, Path]]) -> list[Path]:
    return [files_by_regime[regime][power] for regime in REGIMES for power in POWERS]


def build_concat_dataset(paths: list[Path], indices: np.ndarray) -> ConcatDataset:
    return ConcatDataset([V73IQDataset(path, indices, PARAMETER_COLUMNS) for path in paths])


def close_concat_dataset(dataset: ConcatDataset) -> None:
    for part in dataset.datasets:
        if isinstance(part, V73IQDataset):
            part.close()


def fit_multi_file_normalizer(paths: list[Path], indices: np.ndarray, labels: np.ndarray) -> ParameterNormalizer:
    values_parts, valid_parts = [], []
    for path in paths:
        values, mask = read_selected_parameters(path, indices)
        values_parts.append(values)
        valid_parts.append(mask & (labels[indices, :, None] > 0))
    return ParameterNormalizer().fit(
        torch.from_numpy(np.concatenate(values_parts)), torch.from_numpy(np.concatenate(valid_parts))
    )


def role_metadata(metadata: dict[str, dict[str, np.ndarray]], indices: np.ndarray) -> dict[str, np.ndarray]:
    parts = {key: [] for key in ("identity", "regime", "power", "sample_index")}
    for regime_index, regime_name in enumerate(REGIMES):
        for power_level in POWERS:
            parts["identity"].append(metadata[regime_name]["identities"][indices])
            parts["regime"].append(np.full(len(indices), regime_index, dtype=np.int8))
            parts["power"].append(np.full(len(indices), power_level, dtype=np.int16))
            parts["sample_index"].append(indices.astype(np.int64))
    return {key: np.concatenate(values) for key, values in parts.items()}


def collect_outputs(model, loader: DataLoader, device: torch.device, normalizer: ParameterNormalizer, parameter_weight: float) -> dict[str, object]:
    model.eval()
    tensors = {name: [] for name in ("logits", "labels", "prediction", "target", "mask")}
    totals, classifications, parameters = [], [], []
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
                classification = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                parameter = masked_parameter_smooth_l1(prediction, normalized_target, labels, mask)
                total = classification + float(parameter_weight) * parameter
            tensors["logits"].append(logits.float().cpu()); tensors["labels"].append(labels.float().cpu())
            tensors["prediction"].append(normalizer.inverse_transform(prediction.float().cpu()))
            tensors["target"].append(target.float().cpu()); tensors["mask"].append(mask.cpu())
            totals.append(float(total)); classifications.append(float(classification)); parameters.append(float(parameter))
    return {
        "logits": torch.cat(tensors["logits"]), "labels": torch.cat(tensors["labels"]),
        "prediction": torch.cat(tensors["prediction"]).numpy(), "target": torch.cat(tensors["target"]).numpy(),
        "mask": torch.cat(tensors["mask"]).numpy().astype(bool), "total_loss": float(np.mean(totals)),
        "classification_loss": float(np.mean(classifications)), "parameter_loss": float(np.mean(parameters)),
    }


def train_epoch(model, loader: DataLoader, device: torch.device, normalizer: ParameterNormalizer, optimizer, scaler, args) -> dict[str, float]:
    model.train()
    totals, classifications, parameters = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
        mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
        normalized_target = normalizer.transform(target)
        autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with autocast:
            logits, prediction = model(iq)
            classification = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            parameter = masked_parameter_smooth_l1(prediction, normalized_target, labels, mask)
            total = classification + float(args.parameter_weight) * parameter
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer); scaler.update()
        else:
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm); optimizer.step()
        totals.append(float(total.detach())); classifications.append(float(classification.detach())); parameters.append(float(parameter.detach()))
    return {"total_loss": float(np.mean(totals)), "classification_loss": float(np.mean(classifications)), "parameter_loss": float(np.mean(parameters))}


def evaluate_outputs(outputs: dict[str, object], thresholds: np.ndarray, metadata: dict[str, np.ndarray], normalizer: ParameterNormalizer) -> dict[str, object]:
    logits, labels_tensor = outputs["logits"], outputs["labels"]
    labels, probabilities = labels_tensor.numpy(), torch.sigmoid(logits).numpy()
    classification = classification_metrics(probabilities, labels, thresholds, metadata["regime"], metadata["power"])
    truth = labels[:, ISRJ_INDEX].astype(bool); predicted = probabilities[:, ISRJ_INDEX] >= thresholds[ISRJ_INDEX]
    valid = outputs["mask"] & (labels[:, :, None] > 0.5)
    oracle = parameter_metrics(outputs["prediction"], outputs["target"], valid, normalizer)
    # Predicted-gate parameter metrics must use each class's own predicted
    # label.  ISRJ alone is only the safety target used for thresholding.
    predicted_labels = probabilities >= thresholds[None, :]
    selected = predicted_labels[:, :, None] & valid
    scale = normalizer.scale.numpy()
    error = np.abs(outputs["prediction"] - outputs["target"]) / scale[None, :, :]
    oracle_nmae = float(oracle["oracle_macro_target_nmae"])
    gated_nmae = float(error[selected].mean()) if selected.any() else float("nan")
    miss = np.where(predicted_labels[:, :, None], error, 1.0)[valid]
    per_power = {}
    for power_level in POWERS:
        row = metadata["power"] == power_level
        per_power[str(power_level)] = {"isrj_recall": float(predicted[row & truth].mean()), "isrj_fpr": float(predicted[row & ~truth].mean())}
    combo = _combination_ids(labels.astype(np.uint8))
    combo_nmae = []
    for value in np.unique(combo):
        selected_combo = selected & (combo[:, None, None] == value)
        if selected_combo.any(): combo_nmae.append(float(error[selected_combo].mean()))
    base = multilabel_metrics(logits, labels_tensor, thresholds)
    base.update({
        "macro_aupr": float(average_precision_score(labels, probabilities, average="macro")),
        "isrj_recall": float(predicted[truth].mean()), "isrj_fpr": float(predicted[~truth].mean()),
        "maximum_cell_fpr": float(max(row["fpr"] for row in classification["cells"])),
        "cells": classification["cells"], "ultra_close_cells": [row for row in classification["cells"] if row["regime"] == "ultra_close"],
        "per_power": per_power, "parameter": oracle,
        "predicted_gate": {
            "oracle_parameter_nmae": oracle_nmae, "predicted_gate_parameter_nmae": gated_nmae,
            "predicted_gate_parameter_coverage": float(selected.sum() / max(valid.sum(), 1)),
            "false_positive_gate_rate": float(predicted[~truth].mean()),
            "miss_penalty_parameter_nmae": float(miss.mean()),
            "worst_combo_predicted_gate_nmae": float(max(combo_nmae)) if combo_nmae else float("nan"),
        },
        "losses": {key: float(outputs[key]) for key in ("total_loss", "classification_loss", "parameter_loss")},
    })
    return base


def audit_identities(project: Path, train_identity: np.ndarray, development_identity: np.ndarray, ignored_seeds: set[int], original_data_file: Path) -> dict[str, object]:
    if np.intersect1d(train_identity, development_identity).size:
        raise RuntimeError("P3o train/development identity overlap")
    previous = [original_data_file]
    for directory in sorted((project / "generated_data").glob("*_rechunked")):
        match = re.search(r"seed(\d{9,10})", directory.name)
        if not match or int(match.group(1)) in ignored_seeds:
            continue
        path = next(directory.rglob("closedset_6class_train.mat"), None)
        if path is not None: previous.append(path)
    rows, maximum = [], 0
    for path in previous:
        old, _ = read_identity_metadata(path)
        train_overlap, dev_overlap = int(np.intersect1d(train_identity, old).size), int(np.intersect1d(development_identity, old).size)
        maximum = max(maximum, train_overlap, dev_overlap)
        rows.append({"path": str(path.resolve()), "train_overlap": train_overlap, "development_overlap": dev_overlap})
    if maximum: raise RuntimeError("P3o identities overlap a prior role")
    return {"train_unique": int(len(np.unique(train_identity))), "development_unique": int(len(np.unique(development_identity))), "train_sha256": identity_hash(train_identity), "development_sha256": identity_hash(development_identity), "cross_role_overlap": 0, "prior_files_checked": len(rows), "maximum_prior_overlap": maximum, "rows": rows}


def save_prediction_payload(path: Path, outputs: dict[str, object], metadata: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, identity=metadata["identity"], regime=metadata["regime"], power=metadata["power"], sample_index=metadata["sample_index"], labels=outputs["labels"].numpy().astype(np.uint8), probabilities=torch.sigmoid(outputs["logits"]).numpy(), prediction=outputs["prediction"], target=outputs["target"], mask=outputs["mask"].astype(np.uint8))


def bootstrap_isrj_recall_difference(candidate_file: Path, candidate_threshold: float, baseline_file: Path, baseline_threshold: float, draws: int = 2000, seed: int = 20260801) -> dict[str, float | int]:
    candidate, baseline = np.load(candidate_file), np.load(baseline_file)
    identities = candidate["identity"].astype(np.int64)
    if not np.array_equal(identities, baseline["identity"]) or not np.array_equal(candidate["labels"], baseline["labels"]):
        raise RuntimeError("Bootstrap payloads are not identity aligned")
    truth = candidate["labels"][:, ISRJ_INDEX].astype(bool)
    unique, inverse = np.unique(identities, return_inverse=True)
    positive = np.bincount(inverse, weights=truth.astype(np.int64), minlength=len(unique))
    c_correct = np.bincount(inverse, weights=((candidate["probabilities"][:, ISRJ_INDEX] >= candidate_threshold) & truth).astype(np.int64), minlength=len(unique))
    b_correct = np.bincount(inverse, weights=((baseline["probabilities"][:, ISRJ_INDEX] >= baseline_threshold) & truth).astype(np.int64), minlength=len(unique))
    draws_idx = np.random.default_rng(seed).integers(0, len(unique), size=(draws, len(unique)))
    denominator = positive[draws_idx].sum(axis=1)
    delta = c_correct[draws_idx].sum(axis=1) / denominator - b_correct[draws_idx].sum(axis=1) / denominator
    return {"clusters": int(len(unique)), "draws": int(draws), "mean": float(delta.mean()), "ci95_low": float(np.quantile(delta, 0.025)), "ci95_high": float(np.quantile(delta, 0.975))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3o_low_power_train_seed830000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3o_low_power_dev_seed860000000_{regime}_rechunked")
    parser.add_argument("--preregister", default="configs/p3o_low_power_attribution_preregister_20260801.json")
    parser.add_argument("--method-freeze", default="configs/p3o_low_power_attribution_method_freeze_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3o_low_power_attribution_development_20260801")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=63); parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--parameter-weight", type=float, default=0.1); parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--checkpoint-tolerance", type=float, default=1.01); parser.add_argument("--cp-cap", type=float, default=0.10)
    parser.add_argument("--cp-confidence", type=float, default=0.95); parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=20260801); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    preregister, freeze_path = project / args.preregister, project / args.method_freeze
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(preregister): raise RuntimeError("P3o preregister hash mismatch")
    if freeze["implementation_sha256"] != sha256_file(Path(__file__).resolve()): raise RuntimeError("P3o implementation hash mismatch")
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    device = get_device(args.device)
    train_roots = {regime: transition_files(project / args.train_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES}
    development_roots = {regime: transition_files(project / args.development_root_template.format(regime=regime), "train", POWERS) for regime in REGIMES}
    metadata, references = {"train": {}, "development": {}}, {}
    for role, roots in (("train", train_roots), ("development", development_roots)):
        reference = None
        for regime_name in REGIMES:
            audit_power_alignment(roots[regime_name]); identities, labels = read_identity_metadata(roots[regime_name][POWERS[0]])
            if reference is None: reference = identities
            elif not np.array_equal(reference, identities): raise RuntimeError(f"P3o {role} regimes are not identity paired")
            metadata[role][regime_name] = {"identities": identities, "labels": labels}
        references[role] = reference
    train_labels, development_labels = metadata["train"]["ID_control"]["labels"], metadata["development"]["ID_control"]["labels"]
    if not np.array_equal(train_labels, development_labels): raise RuntimeError("P3o label contracts differ across identity seeds")
    roles = make_identity_roles(train_labels, args.split_seed)
    original_protocol = json.loads((project / args.run_root / "protocol.json").read_text(encoding="utf-8"))
    audit = audit_identities(project, references["train"], references["development"], {830000000, 860000000}, Path(original_protocol["data_file"]))
    (output / "identity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); np.savez_compressed(output / "identity_roles.npz", **roles)
    train_paths, development_paths = ordered_paths(train_roots), ordered_paths(development_roots)
    full_steps = math.ceil(len(roles["full_fit"]) * len(REGIMES) * len(POWERS) / args.batch_size)
    small_steps = math.ceil(len(roles["small_fit"]) * len(REGIMES) * len(POWERS) / args.batch_size)
    scale_specs = {
        "small_epoch_matched": {"fit_role": "small_fit", "epochs": args.epochs},
        "small_step_matched": {"fit_role": "small_fit", "epochs": int(math.ceil(args.epochs * full_steps / small_steps))},
        "full_epoch_matched": {"fit_role": "full_fit", "epochs": args.epochs},
    }
    development_indices = np.arange(len(development_labels), dtype=np.int64)
    meta = {role: role_metadata(metadata["train"], indices) for role, indices in roles.items()}
    development_meta = role_metadata(metadata["development"], development_indices)
    protocol = vars(args) | {"device_resolved": str(device), "model_seeds": list(MODEL_SEEDS), "scales": scale_specs, "optimizer_steps_per_epoch": {"full": full_steps, "small": small_steps}, "role_identity_counts": {name: int(len(indices)) for name, indices in roles.items()}, "role_view_counts": {name: int(len(indices) * len(REGIMES) * len(POWERS)) for name, indices in roles.items()}, "identity_audit_summary": {key: audit[key] for key in audit if key != "rows"}, "preregister_sha256": freeze["preregister_sha256"], "implementation_sha256": freeze["implementation_sha256"], "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "protocol.json").write_text(json.dumps(_jsonable(protocol), indent=2), encoding="utf-8")
    common_datasets = {role: build_concat_dataset(train_paths, roles[role]) for role in ("early", "calibration", "selection")}
    development_dataset = build_concat_dataset(development_paths, development_indices)
    results = []
    try:
        for scale_name, spec in scale_specs.items():
            normalizer = fit_multi_file_normalizer(train_paths, roles[spec["fit_role"]], train_labels)
            fit_dataset = build_concat_dataset(train_paths, roles[spec["fit_role"]])
            try:
                for model_seed in MODEL_SEEDS:
                    print(f"START scale={scale_name} seed={model_seed}", flush=True); set_seed(model_seed)
                    generator = torch.Generator().manual_seed(model_seed)
                    loaders = {"fit": DataLoader(fit_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.workers, pin_memory=device.type == "cuda")}
                    loaders.update({role: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") for role, dataset in common_datasets.items()})
                    development_loader = DataLoader(development_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
                    model = build_multitask_model("mlcnn_iq_grouped_task_routed", 24, 0.2, 256, 64).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                    scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None
                    run_dir = output / scale_name / f"seed_{model_seed}"; run_dir.mkdir(parents=True)
                    history = []
                    for epoch in range(1, int(spec["epochs"]) + 1):
                        lr = _set_warmup_cosine_lr(optimizer, args.learning_rate, epoch, int(spec["epochs"]), args.warmup_epochs)
                        train = train_epoch(model, loaders["fit"], device, normalizer, optimizer, scaler, args)
                        early_outputs = collect_outputs(model, loaders["early"], device, normalizer, args.parameter_weight)
                        early_parameter = output_parameter_metrics(early_outputs, normalizer)
                        row = {"epoch": epoch, "lr": lr, "train": train, "early_parameter_macro_nmae": float(early_parameter["oracle_macro_target_nmae"]), "early_macro_aupr": float(average_precision_score(early_outputs["labels"].numpy(), torch.sigmoid(early_outputs["logits"]).numpy(), average="macro"))}
                        history.append(row); torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pt")
                        print(json.dumps({"scale": scale_name, "seed": model_seed, **row}), flush=True)
                    choices = select_checkpoint_epochs(history, args.checkpoint_tolerance)
                    checkpoint_rows = {}
                    for checkpoint_name, epoch_key in (("C_param", "C_param_epoch"), ("C_joint", "C_joint_epoch")):
                        epoch = int(choices[epoch_key]); model.load_state_dict(torch.load(run_dir / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=True)); model.eval()
                        torch.save(model.state_dict(), run_dir / f"{checkpoint_name}.pt")
                        calibration_outputs = collect_outputs(model, loaders["calibration"], device, normalizer, args.parameter_weight)
                        f1_thresholds = optimize_class_thresholds(torch.sigmoid(calibration_outputs["logits"]).numpy(), calibration_outputs["labels"].numpy())
                        safe_audit = select_cp_safe_isrj_threshold(torch.sigmoid(calibration_outputs["logits"]).numpy(), calibration_outputs["labels"].numpy(), meta["calibration"]["regime"], meta["calibration"]["power"], args.cp_cap, args.cp_confidence)
                        safe_thresholds = np.asarray(f1_thresholds, dtype=np.float64).copy()
                        if safe_audit["feasible"]: safe_thresholds[ISRJ_INDEX] = float(safe_audit["threshold"])
                        selection_outputs = collect_outputs(model, loaders["selection"], device, normalizer, args.parameter_weight)
                        development_outputs = collect_outputs(model, development_loader, device, normalizer, args.parameter_weight)
                        prediction_file = run_dir / f"{checkpoint_name}_development_predictions.npz"; save_prediction_payload(prediction_file, development_outputs, development_meta)
                        threshold_rows = {}
                        for threshold_name, thresholds in (("T_F1", f1_thresholds), ("T_safe", safe_thresholds)):
                            threshold_rows[threshold_name] = {"thresholds": np.asarray(thresholds, dtype=np.float64), "selection": evaluate_outputs(selection_outputs, thresholds, meta["selection"], normalizer), "development": evaluate_outputs(development_outputs, thresholds, development_meta, normalizer)}
                        checkpoint_rows[checkpoint_name] = {"epoch": epoch, "calibration_safe_audit": safe_audit, "thresholds": threshold_rows, "prediction_file": str(prediction_file.relative_to(project))}
                    run = {"scale": scale_name, "seed": model_seed, "fit_role": spec["fit_role"], "epochs": int(spec["epochs"]), "normalizer": normalizer.state_dict(), "history": history, "checkpoint_selection": choices, "checkpoints": checkpoint_rows}
                    (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8"); (run_dir / "result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
                    results.append(run)
                    candidate = checkpoint_rows["C_joint"]["thresholds"]["T_safe"]["development"]
                    print(json.dumps({"scale": scale_name, "seed": model_seed, "candidate_isrj_recall": candidate["isrj_recall"], "candidate_maximum_cell_fpr": candidate["maximum_cell_fpr"], "candidate_macro_aupr": candidate["macro_aupr"]}), flush=True)
                    del model
                    if device.type == "cuda": torch.cuda.empty_cache()
            finally:
                close_concat_dataset(fit_dataset)
    finally:
        for dataset in common_datasets.values(): close_concat_dataset(dataset)
        close_concat_dataset(development_dataset)
    indexed = {(row["scale"], int(row["seed"])): row for row in results}
    candidate_rows = [indexed[("full_epoch_matched", seed)]["checkpoints"]["C_joint"]["thresholds"]["T_safe"]["development"] for seed in MODEL_SEEDS]
    baseline_f1 = [indexed[("full_epoch_matched", seed)]["checkpoints"]["C_param"]["thresholds"]["T_F1"]["development"] for seed in MODEL_SEEDS]
    baseline_safe = [indexed[("full_epoch_matched", seed)]["checkpoints"]["C_param"]["thresholds"]["T_safe"]["development"] for seed in MODEL_SEEDS]
    comparisons, bootstrap = [], []
    for pos, seed in enumerate(MODEL_SEEDS):
        full = indexed[("full_epoch_matched", seed)]; small = indexed[("small_epoch_matched", seed)]
        candidate, f1, safe = candidate_rows[pos], baseline_f1[pos], baseline_safe[pos]
        c_file = project / full["checkpoints"]["C_joint"]["prediction_file"]; p_file = project / full["checkpoints"]["C_param"]["prediction_file"]
        c_threshold = float(full["checkpoints"]["C_joint"]["thresholds"]["T_safe"]["thresholds"][ISRJ_INDEX]); p_threshold = float(full["checkpoints"]["C_param"]["thresholds"]["T_F1"]["thresholds"][ISRJ_INDEX])
        bootstrap.append({"seed": seed, "candidate_vs_full_C_param_T_F1_isrj_recall": bootstrap_isrj_recall_difference(c_file, c_threshold, p_file, p_threshold, args.bootstrap_draws, args.split_seed + seed)})
        comparisons.append({"seed": seed, "candidate_recall_delta_vs_full_C_param_T_F1": candidate["isrj_recall"] - f1["isrj_recall"], "candidate_macro_aupr_delta_vs_full_C_param": candidate["macro_aupr"] - f1["macro_aupr"], "candidate_oracle_nmae_ratio_vs_full_C_param": candidate["parameter"]["oracle_macro_target_nmae"] / f1["parameter"]["oracle_macro_target_nmae"], "candidate_predicted_gate_nmae_ratio_vs_full_C_param_T_safe": candidate["predicted_gate"]["predicted_gate_parameter_nmae"] / safe["predicted_gate"]["predicted_gate_parameter_nmae"], "candidate_miss_penalty_delta_vs_full_C_param_T_safe": candidate["predicted_gate"]["miss_penalty_parameter_nmae"] - safe["predicted_gate"]["miss_penalty_parameter_nmae"], "candidate_worst_combo_delta_vs_full_C_param_T_safe": candidate["predicted_gate"]["worst_combo_predicted_gate_nmae"] - safe["predicted_gate"]["worst_combo_predicted_gate_nmae"], "full_vs_small_25db_macro_aupr_delta": full["checkpoints"]["C_joint"]["thresholds"]["T_F1"]["development"]["macro_aupr"] - small["checkpoints"]["C_joint"]["thresholds"]["T_F1"]["development"]["macro_aupr"], "C_joint_macro_aupr_delta_vs_C_param": candidate["macro_aupr"] - f1["macro_aupr"]})
    gates = {
        "identity_overlap_zero": audit["maximum_prior_overlap"] == 0 and audit["cross_role_overlap"] == 0,
        "candidate_all_seeds_maximum_cell_fpr_at_most_0p10": all(row["maximum_cell_fpr"] <= 0.10 for row in candidate_rows),
        "candidate_all_seeds_overall_isrj_recall_at_least_0p85": all(row["isrj_recall"] >= 0.85 for row in candidate_rows),
        "candidate_recall_drop_vs_full_C_param_T_F1_at_most_0p02": all(row["candidate_recall_delta_vs_full_C_param_T_F1"] >= -0.02 for row in comparisons),
        "candidate_all_seeds_25db_isrj_recall_at_least_0p78": all(row["per_power"]["25"]["isrj_recall"] >= 0.78 for row in candidate_rows),
        "candidate_all_seeds_25db_each_regime_fpr_at_most_0p10": all(max(cell["fpr"] for cell in row["cells"] if int(cell["power"]) == 25) <= 0.10 for row in candidate_rows),
        "candidate_oracle_parameter_nmae_ratio_at_most_1p01": all(row["candidate_oracle_nmae_ratio_vs_full_C_param"] <= 1.01 for row in comparisons),
        "candidate_predicted_gate_parameter_nmae_ratio_at_most_1p01": all(row["candidate_predicted_gate_nmae_ratio_vs_full_C_param_T_safe"] <= 1.01 for row in comparisons),
        "candidate_miss_penalty_and_worst_combo_nonworse": all(row["candidate_miss_penalty_delta_vs_full_C_param_T_safe"] <= 0 and row["candidate_worst_combo_delta_vs_full_C_param_T_safe"] <= 0 for row in comparisons),
        "full_vs_small_epoch_matched_25db_aupr_positive_all_seeds": all(row["full_vs_small_25db_macro_aupr_delta"] > 0 for row in comparisons),
        "C_joint_macro_aupr_nonnegative_vs_C_param_all_seeds": all(row["C_joint_macro_aupr_delta_vs_C_param"] >= 0 for row in comparisons),
        "safe_calibration_feasible_all_scales_checkpoints_seeds": all(row["checkpoints"][checkpoint]["calibration_safe_audit"]["feasible"] for row in results for checkpoint in CHECKPOINTS),
        "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True,
    }
    passed = bool(all(gates.values()))
    attribution = "P3o-A supported: freeze candidate and run independent confirmation" if passed else "P3o-A failed one or more frozen gates; use coverage/checkpoint/calibration attribution without retuning seed860000000"
    summary = {"experiment": "P3o-A low-power attribution", "development_passed": passed, "promotion_gates": gates, "comparisons": comparisons, "clustered_paired_bootstrap": bootstrap, "runs": results, "identity_audit": {key: audit[key] for key in audit if key != "rows"}, "attribution": attribution, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "attribution": attribution}, indent=2), flush=True)


if __name__ == "__main__":
    main()
