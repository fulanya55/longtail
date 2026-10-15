"""Five-seed B0 training on the fresh Route-A backbone development source."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.v73_dataset import (  # noqa: E402
    V73SignalDataset,
    read_labels,
    read_parameters,
)
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.thresholding import (  # noqa: E402
    apply_thresholds,
    select_per_label_weak_recall_fpr_thresholds,
)
from cjlt.losses.classification import MultiLabelClassificationLoss  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
COMMON_INDICES = (0, 2, 3, 4, 5)
TARGET_INDICES = (3, 4, 5)
SEEDS = (2039, 2040, 2041, 2042, 2043)
EXPECTED_DATA_SHA256 = "c4122ffae67de855f97a74b42c17ef3df824fad31888dcd6e9dd28b9952c00ac"
METRIC_NAMES = (
    "macro_aupr",
    "macro_f1",
    "hamming_loss",
    "target_mean_weak_auroc",
    "target_mean_weak_recall",
    "target_mean_strong_recall",
    "target_mean_negative_fpr",
)
MODEL_SPECS = {
    "b0": {
        "experiment": "RouteA_BackboneDevelopmentV1_B0",
        "display_name": "B0_F2_DFMS_Full",
        "backbone_name": "dfms_iq",
        "summary_name": "b0_summary.json",
        "formal_evidence_level": "five_seed_development_only_b0_baseline",
        "seed_evidence_level": "development_only_b0_seed_component",
    },
    "m1": {
        "experiment": "RouteA_BackboneDevelopmentV1_M1",
        "display_name": "B1_F2_DFMS_AntiAlias",
        "backbone_name": "dfms_iq_anti_alias",
        "summary_name": "m1_summary.json",
        "formal_evidence_level": "five_seed_development_only_m1_candidate",
        "seed_evidence_level": "development_only_m1_seed_component",
    },
    "m2": {
        "experiment": "RouteA_BackboneDevelopmentV1_M2",
        "display_name": "B2_F2_DFMS_CSFP",
        "backbone_name": "dfms_iq_csfp",
        "summary_name": "m2_summary.json",
        "formal_evidence_level": "five_seed_development_only_m2_candidate",
        "seed_evidence_level": "development_only_m2_seed_component",
    },
    "m3": {
        "experiment": "RouteA_BackboneDevelopmentV1_M3",
        "display_name": "B3_F2_DFMS_EPB",
        "backbone_name": "dfms_iq_epb",
        "summary_name": "m3_summary.json",
        "formal_evidence_level": "five_seed_development_only_m3_candidate",
        "seed_evidence_level": "development_only_m3_seed_component",
    },
    "m4k8": {
        "experiment": "RouteA_BackboneDevelopmentV1_M4K8",
        "display_name": "B4_F2_DFMS_LocalStatsK8",
        "backbone_name": "dfms_iq_local_topk8",
        "summary_name": "m4_k8_summary.json",
        "formal_evidence_level": "five_seed_development_only_m4_k8_candidate",
        "seed_evidence_level": "development_only_m4_k8_seed_component",
    },
    "m4k16": {
        "experiment": "RouteA_BackboneDevelopmentV1_M4K16",
        "display_name": "B4_F2_DFMS_LocalStatsK16",
        "backbone_name": "dfms_iq_local_topk16",
        "summary_name": "m4_k16_summary.json",
        "formal_evidence_level": "five_seed_development_only_m4_k16_candidate",
        "seed_evidence_level": "development_only_m4_k16_seed_component",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _component_power(values79: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            values79[:, get_parameter_specs(name)[0].column_index]
            for name in CLASS_NAMES
        ],
        axis=1,
    )


def _seal_flags(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    flags = {
        key: payload.get("test_v3", {}).get(key)
        for key in ("access_authorized", "analysis_allowed", "used")
    }
    if flags != {
        "access_authorized": False,
        "analysis_allowed": False,
        "used": False,
    }:
        raise ValueError("test-v3 seal is not intact")
    return flags


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(
        np.float32
    )


def _macro_aupr(targets: np.ndarray, scores: np.ndarray, indices) -> float:
    values = []
    for index in indices:
        truth = targets[:, index]
        if 0 < int(truth.sum()) < len(truth):
            values.append(average_precision_score(truth, scores[:, index]))
    return float(np.mean(values))


def _weak_auroc(
    targets: np.ndarray,
    scores: np.ndarray,
    weak: np.ndarray,
    label_index: int,
) -> float:
    positive = (targets[:, label_index] == 1) & weak[:, label_index]
    negative = targets[:, label_index] == 0
    if not positive.any() or not negative.any():
        return float("nan")
    truth = np.concatenate([np.ones(positive.sum()), np.zeros(negative.sum())])
    values = np.concatenate([scores[positive, label_index], scores[negative, label_index]])
    return float(roc_auc_score(truth, values))


def _target_weak_auroc(targets, scores, weak) -> float:
    values = [_weak_auroc(targets, scores, weak, index) for index in TARGET_INDICES]
    return float(np.nanmean(values))


def _loss_value(criterion, logits, targets):
    element = criterion.elementwise(logits, targets)
    common = element[:, torch.as_tensor(COMMON_INDICES, device=logits.device)].mean()
    saw = element[:, 1].mean()
    return common + 0.2 * saw


def _dataset_loader(data_path, indices, batch_size, shuffle, seed):
    dataset = V73SignalDataset(
        data_path,
        indices,
        class_mode="six",
        normalization="complex_rms",
    )
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    return dataset, loader


def _predict(model, data_path, indices, device, batch_size):
    dataset, loader = _dataset_loader(data_path, indices, batch_size, False, 0)
    logits, targets, sample_ids = [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                values = batch["signal"].to(device, non_blocking=True)
                output = model(values)
                logits.append(output.detach().cpu().numpy().astype(np.float32))
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return {
        "logits": np.concatenate(logits),
        "targets": np.concatenate(targets),
        "sample_ids": np.concatenate(sample_ids),
    }


def _fit_platt(logits: np.ndarray, targets: np.ndarray) -> dict:
    slopes, intercepts, converged = [], [], []
    for label_index in range(logits.shape[1]):
        truth = targets[:, label_index].astype(np.uint8)
        if len(np.unique(truth)) < 2:
            slopes.append(1.0)
            intercepts.append(0.0)
            converged.append(False)
            continue
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        model.fit(logits[:, label_index : label_index + 1], truth)
        slopes.append(float(model.coef_[0, 0]))
        intercepts.append(float(model.intercept_[0]))
        converged.append(bool(model.n_iter_[0] < model.max_iter))
    return {
        "method": "per_label_platt_logistic_regression",
        "C": 1e6,
        "solver": "lbfgs",
        "slopes": slopes,
        "intercepts": intercepts,
        "converged": converged,
        "source_role": "calibration",
    }


def _apply_platt(logits: np.ndarray, calibration: dict) -> np.ndarray:
    slopes = np.asarray(calibration["slopes"], dtype=np.float64)
    intercepts = np.asarray(calibration["intercepts"], dtype=np.float64)
    return _sigmoid(np.asarray(logits, dtype=np.float64) * slopes + intercepts)


def _per_label_metrics(targets, probabilities, predictions, weak):
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        positive = targets[:, index] == 1
        weak_positive = positive & weak[:, index]
        strong_positive = positive & ~weak[:, index]
        negative = ~positive
        result[name] = {
            "aupr": float(average_precision_score(targets[:, index], probabilities[:, index])),
            "weak_vs_negative_auroc": _weak_auroc(targets, probabilities, weak, index),
            "weak_recall": float(predictions[weak_positive, index].mean()) if weak_positive.any() else float("nan"),
            "strong_recall": float(predictions[strong_positive, index].mean()) if strong_positive.any() else float("nan"),
            "negative_fpr": float(predictions[negative, index].mean()) if negative.any() else float("nan"),
            "positive_count": int(positive.sum()),
            "weak_positive_count": int(weak_positive.sum()),
            "strong_positive_count": int(strong_positive.sum()),
            "negative_count": int(negative.sum()),
        }
    return result


def _evaluate(early, calibration_role, threshold_role, weak6, output_dir, args):
    calibration = _fit_platt(calibration_role["logits"], calibration_role["targets"])
    calibrated = {
        "early": _apply_platt(early["logits"], calibration),
        "threshold": _apply_platt(threshold_role["logits"], calibration),
    }
    threshold_weak = weak6[threshold_role["sample_ids"]]
    selection = select_per_label_weak_recall_fpr_thresholds(
        threshold_role["targets"],
        calibrated["threshold"],
        weak_component_mask=threshold_weak,
        max_negative_fpr=0.20,
        source_role="threshold_selection",
        grid=np.arange(0.05, 0.951, 0.01),
    )
    predictions = apply_thresholds(calibrated["early"], selection.thresholds)
    early_weak = weak6[early["sample_ids"]]
    metrics = compute_multilabel_metrics(
        early["targets"],
        predictions,
        probabilities=calibrated["early"],
        class_names=CLASS_NAMES,
        weak_component_mask=early_weak,
    )
    per_label = _per_label_metrics(
        early["targets"], calibrated["early"], predictions, early_weak
    )
    metrics["per_label_diagnostics"] = per_label
    metrics["target_mean_weak_auroc"] = float(
        np.mean([per_label[CLASS_NAMES[index]]["weak_vs_negative_auroc"] for index in TARGET_INDICES])
    )
    metrics["target_mean_weak_recall"] = float(
        np.mean([per_label[CLASS_NAMES[index]]["weak_recall"] for index in TARGET_INDICES])
    )
    metrics["target_mean_strong_recall"] = float(
        np.mean([per_label[CLASS_NAMES[index]]["strong_recall"] for index in TARGET_INDICES])
    )
    metrics["target_mean_negative_fpr"] = float(
        np.mean([per_label[CLASS_NAMES[index]]["negative_fpr"] for index in TARGET_INDICES])
    )
    _write_json(output_dir / "calibration.json", calibration)
    _write_json(
        output_dir / "thresholds.json",
        {
            "thresholds": selection.thresholds,
            "mode": selection.mode,
            "source_role": selection.source_role,
            "max_negative_fpr": 0.20,
            "grid_start": 0.05,
            "grid_stop": 0.95,
            "grid_step": 0.01,
        },
    )
    _write_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(
        output_dir / "predictions.npz",
        sample_ids=early["sample_ids"],
        logits=early["logits"],
        probabilities=calibrated["early"],
        predictions=predictions,
        targets=early["targets"],
        weak_mask=early_weak.astype(np.uint8),
    )
    return metrics, calibration, selection


def _profile_macs(model, device):
    total = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv1d):
            batch, out_channels, length = output.shape
            operations = (
                batch
                * out_channels
                * length
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
            )
        elif isinstance(module, nn.Linear):
            operations = int(np.prod(output.shape[:-1])) * module.in_features * module.out_features
        else:
            return
        total += int(operations)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 2, 3000, device=device))
    for handle in handles:
        handle.remove()
    return int(total)


def _profile_latency(model, device):
    values = torch.randn(1, 2, 3000, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(values)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(100):
            model(values)
        torch.cuda.synchronize(device)
    return float((time.perf_counter() - started) * 1000.0 / 100.0)


def _run_seed(args, labels6, weak6, seed, data_sha, seal_flags, device):
    model_spec = MODEL_SPECS[args.model]
    seed_root = args.output_root / f"seed_{seed}"
    receipt_path = seed_root / "run_receipt.json"
    role_path = args.role_root / f"seed_{seed}" / "role_indices.npz"
    role_sha = _sha256(role_path)
    if args.resume and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") == "completed"
            and receipt.get("data_sha256") == data_sha
            and receipt.get("role_indices_sha256") == role_sha
        ):
            return json.loads((seed_root / "seed_summary.json").read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)
    base_receipt = {
        "seed": int(seed),
        "model": model_spec["display_name"],
        "data_sha256": data_sha,
        "role_indices_sha256": role_sha,
        "test_v3_seal": seal_flags,
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    started = time.time()
    try:
        saved = np.load(role_path)
        roles = {name: saved[name].astype(np.int64) for name in ("fit", "validation", "early", "calibration", "threshold")}
        all_ids = np.concatenate(list(roles.values()))
        if len(np.unique(all_ids)) != len(all_ids):
            raise AssertionError("role sample overlap detected")
        train_seed = int(seed) + 73000
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        torch.cuda.manual_seed_all(train_seed)
        torch.cuda.reset_peak_memory_stats(device)
        model = build_backbone(
            model_spec["backbone_name"], num_classes=6, width=16, dropout=0.2
        ).to(device)
        initial_state_sha = hashlib.sha256(
            b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
        ).hexdigest()
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        expected_parameter_count = {
            "b0": 257511,
            "m1": 257511,
            "m2": 310376,
            "m3": 313191,
            "m4k8": 290664,
            "m4k16": 290664,
        }[args.model]
        if parameter_count != expected_parameter_count:
            raise AssertionError(f"{args.model.upper()} parameter count drift")
        classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
        backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids]
        classifier_parameters = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": 5e-5},
                {"params": classifier_parameters, "lr": 5e-4},
            ],
            weight_decay=1e-4,
        )
        criterion = MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0).to(device)
        fit_dataset, fit_loader = _dataset_loader(
            args.data_path, roles["fit"], args.batch_size, True, train_seed
        )
        best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
        try:
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for batch in fit_loader:
                    values = batch["signal"].to(device, non_blocking=True)
                    targets = batch["label"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(values)
                    loss = _loss_value(criterion, logits, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("non-finite gradient norm")
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                validation = _predict(model, args.data_path, roles["validation"], device, args.eval_batch_size)
                probabilities = _sigmoid(validation["logits"])
                validation_weak = weak6[validation["sample_ids"]]
                macro = _macro_aupr(validation["targets"], probabilities, COMMON_INDICES)
                weak_auroc = _target_weak_auroc(validation["targets"], probabilities, validation_weak)
                history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(np.mean(losses)),
                        "validation_common_macro_aupr": macro,
                        "validation_target_weak_auroc": weak_auroc,
                        "backbone_lr": 5e-5,
                        "classifier_lr": 5e-4,
                    }
                )
                key = (macro, weak_auroc)
                if best_key is None or key[0] > best_key[0] + 1e-8 or (
                    np.isclose(key[0], best_key[0]) and key[1] > best_key[1] + 1e-8
                ):
                    best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
                    best_key, best_epoch, stale = key, epoch, 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
        finally:
            fit_dataset.close()
        if best_state is None:
            raise RuntimeError(f"{args.model.upper()} did not produce a checkpoint")
        model.load_state_dict(best_state, strict=True)
        checkpoint = seed_root / "best_model.pth"
        torch.save(
            {
                "model_state_dict": best_state,
                "model": model_spec["display_name"],
                "seed": int(seed),
                "train_seed": train_seed,
                "initial_state_sha256": initial_state_sha,
                "best_epoch": int(best_epoch),
                "best_validation_common_macro_aupr": float(best_key[0]),
                "best_validation_target_weak_auroc": float(best_key[1]),
            },
            checkpoint,
        )
        _write_history(seed_root / "history.csv", history)
        early = _predict(model, args.data_path, roles["early"], device, args.eval_batch_size)
        calibration_role = _predict(model, args.data_path, roles["calibration"], device, args.eval_batch_size)
        threshold_role = _predict(model, args.data_path, roles["threshold"], device, args.eval_batch_size)
        metrics, calibration, selection = _evaluate(
            early, calibration_role, threshold_role, weak6, seed_root, args
        )
        macs = _profile_macs(model, device)
        latency = _profile_latency(model, device)
        summary = {
            "status": "completed",
            "evidence_level": (
                model_spec["seed_evidence_level"]
                if args.formal_protocol
                else "smoke_only_not_formal_model_evidence"
            ),
            "seed": int(seed),
            "model": model_spec["display_name"],
            "best_epoch": int(best_epoch),
            "epochs_ran": len(history),
            "initial_state_sha256": initial_state_sha,
            "metrics": {name: float(metrics[name]) for name in METRIC_NAMES},
            "per_label_diagnostics": metrics["per_label_diagnostics"],
            "thresholds": [float(value) for value in selection.thresholds],
            "platt_slopes": calibration["slopes"],
            "parameter_count": parameter_count,
            "conv_linear_macs_per_sample": macs,
            "cuda_batch1_latency_ms": latency,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint_sha256": _sha256(checkpoint),
            "predictions_sha256": _sha256(seed_root / "predictions.npz"),
            "elapsed_seconds": float(time.time() - started),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
        _write_json(seed_root / "seed_summary.json", summary)
        receipt = {
            **base_receipt,
            "status": "completed",
            "train_seed": train_seed,
            "initial_state_sha256": initial_state_sha,
            "parameter_count": parameter_count,
            "best_epoch": int(best_epoch),
            "epochs_ran": len(history),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "predictions_sha256": summary["predictions_sha256"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        _write_json(receipt_path, receipt)
        print(json.dumps({"seed": seed, "status": "completed", "best_epoch": best_epoch, "metrics": summary["metrics"]}, ensure_ascii=False), flush=True)
        return summary
    except Exception as exception:
        _write_json(
            receipt_path,
            {
                **base_receipt,
                "status": "aborted_engineering_failure",
                "error_type": type(exception).__name__,
                "error": str(exception),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": float(time.time() - started),
            },
        )
        raise


def _aggregate(rows):
    aggregate = {}
    for index, name in enumerate(METRIC_NAMES):
        aggregate[name] = _mean_ci(
            [row["metrics"][name] for row in rows], 20806000 + index
        )
    aggregate["per_label"] = {}
    for label_index, label_name in enumerate(CLASS_NAMES):
        aggregate["per_label"][label_name] = {}
        for metric_index, metric_name in enumerate(
            ("aupr", "weak_vs_negative_auroc", "weak_recall", "strong_recall", "negative_fpr")
        ):
            aggregate["per_label"][label_name][metric_name] = _mean_ci(
                [row["per_label_diagnostics"][label_name][metric_name] for row in rows],
                20807000 + label_index * 100 + metric_index,
            )
    return aggregate


def run(args):
    model_spec = MODEL_SPECS[args.model]
    args.output_root.mkdir(parents=True, exist_ok=True)
    terminal_receipt = args.output_root / "terminal_receipt.json"
    started = time.time()
    try:
        data_sha = _sha256(args.data_path)
        if data_sha != EXPECTED_DATA_SHA256:
            raise ValueError("backbone-development-v1 data SHA mismatch")
        seal_flags = _seal_flags(args.seal_manifest)
        labels6 = read_labels(args.data_path)
        values79, _ = read_parameters(args.data_path)
        weak6 = compute_weak_component_mask(labels6, _component_power(values79), 10.0)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise RuntimeError("formal B0 training requires CUDA")
        seeds = tuple(args.seeds)
        args.formal_protocol = bool(
            seeds == SEEDS
            and args.epochs == 36
            and args.patience == 8
            and args.batch_size == 64
            and args.model in MODEL_SPECS
        )
        evidence_level = (
            model_spec["formal_evidence_level"]
            if args.formal_protocol
            else "smoke_only_not_formal_model_evidence"
        )
        _write_json(
            args.output_root / "experiment_config.json",
            {
                "experiment": model_spec["experiment"],
                "model_key": args.model,
                "model": model_spec["display_name"],
                "backbone_name": model_spec["backbone_name"],
                "seeds": list(seeds),
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "backbone_lr": 5e-5,
                "classifier_lr": 5e-4,
                "weight_decay": 1e-4,
                "gradient_clip": 5.0,
                "loss": "focal_gamma_pos_0_gamma_neg_3_non_saw_mean_plus_0p2_saw",
                "data_sha256": data_sha,
                "formal_protocol": args.formal_protocol,
                "evidence_level": evidence_level,
                "test_v3_seal": seal_flags,
                "test_v3_used": False,
            },
        )
        rows = [
            _run_seed(args, labels6, weak6, seed, data_sha, seal_flags, device)
            for seed in seeds
        ]
        payload = {
            "status": "completed",
            "evidence_level": evidence_level,
            "formal_protocol": args.formal_protocol,
            "model_key": args.model,
            "model": model_spec["display_name"],
            "seeds": list(seeds),
            "data_sha256": data_sha,
            "aggregate": _aggregate(rows),
            "per_seed": rows,
            "test_files_opened": False,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        }
        summary_path = args.output_root / model_spec["summary_name"]
        _write_json(summary_path, payload)
        _write_json(
            terminal_receipt,
            {
                "status": "completed",
                "evidence_level": evidence_level,
                "formal_protocol": args.formal_protocol,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": _sha256(summary_path),
                "elapsed_seconds": float(time.time() - started),
                "test_v3_used": False,
            },
        )
        print(json.dumps({"status": "completed", "summary": str(summary_path), "aggregate": payload["aggregate"]}, ensure_ascii=False, indent=2))
        return payload
    except Exception as exception:
        _write_json(
            terminal_receipt,
            {
                "status": "aborted_engineering_failure",
                "error_type": type(exception).__name__,
                "error": str(exception),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": float(time.time() - started),
                "test_v3_used": False,
            },
        )
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(r"E:\LJQ_code\干扰数据\route_a_backbone-development-v1_20260806")
        / "closedset_6class_train.mat",
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="b0")
    parser.add_argument(
        "--role-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/tapc/route_a_backbone_development_v1/roles",
    )
    parser.add_argument(
        "--seal-manifest",
        type=Path,
        default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/tapc/route_a_backbone_b0_development_v1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
