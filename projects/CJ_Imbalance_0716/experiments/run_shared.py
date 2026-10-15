"""Shared four/six-class supervised baseline runner."""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.registry import (  # noqa: E402
    get_class_names,
    select_labels,
    valid_sample_mask,
)
from cjlt.data.split_builder import build_stratified_roles  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from cjlt.engine.calibration import apply_temperature, fit_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.thresholding import (  # noqa: E402
    select_global_threshold,
    select_shrinkage_per_label_thresholds,
)
from cjlt.engine.trainer import predict_loader, train_model  # noqa: E402
from cjlt.engine.run_manager import (  # noqa: E402
    configuration_sha256,
    decide_run,
    stable_run_arguments,
)
from cjlt.losses.classification import build_classification_loss  # noqa: E402
from cjlt.models.backbone_adapter import (  # noqa: E402
    build_backbone,
    count_trainable_parameters,
)


def load_config(path: Path) -> Dict[str, object]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    config["_config_path"] = str(config_path.resolve())
    return config


def _data_path(config: Dict[str, object]) -> Path:
    path = Path(str(config["data_root"])) / str(config["train_file"])
    if not path.is_file():
        raise FileNotFoundError("Train MAT not found: {}".format(path))
    return path


def _limit(indices: np.ndarray, maximum: int) -> np.ndarray:
    count = int(maximum)
    if count <= 0 or len(indices) <= count:
        return indices.copy()
    return indices[:count].copy()


def _development_state(config, seed):
    labels6 = read_labels(_data_path(config))
    class_mode = str(config["class_mode"])
    labels = select_labels(labels6, class_mode)
    eligible = np.flatnonzero(valid_sample_mask(labels6, class_mode))
    split = build_stratified_roles(labels, eligible, seed=int(seed))
    return labels, split


def build_run_plan(
    config: Dict[str, object],
    loss_name: str,
    seed: int,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
) -> Dict[str, object]:
    labels, split = _development_state(config, seed)
    train_indices = _limit(split.optimization, max_train_samples)
    early_indices = _limit(split.early_stop, max_eval_samples)
    calibration_indices = _limit(split.calibration, max_eval_samples)
    threshold_indices = _limit(split.threshold_selection, max_eval_samples)
    class_mode = str(config["class_mode"])
    return {
        "experiment_name": str(config["experiment_name"]),
        "class_mode": class_mode,
        "class_names": get_class_names(class_mode),
        "num_classes": int(labels.shape[1]),
        "loss": str(loss_name),
        "seed": int(seed),
        "train_samples": int(len(train_indices)),
        "early_stop_samples": int(len(early_indices)),
        "calibration_samples": int(len(calibration_indices)),
        "threshold_samples": int(len(threshold_indices)),
        "split_sha256": split.sha256,
        "allow_locked_test": bool(config.get("allow_locked_test", False)),
        "train_indices": train_indices,
        "early_stop_indices": early_indices,
        "calibration_indices": calibration_indices,
        "threshold_indices": threshold_indices,
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: {!r}".format(type(value)))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_jsonable,
        )


def _write_history(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _sigmoid(logits):
    values = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(
        np.float32
    )


def run_experiment(args) -> Dict[str, object]:
    config = load_config(args.config)
    plan = build_run_plan(
        config,
        loss_name=args.loss,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    printable_plan = {
        key: value
        for key, value in plan.items()
        if not key.endswith("_indices")
    }
    if args.dry_run:
        print(json.dumps(printable_plan, ensure_ascii=False, indent=2))
        return printable_plan

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "outputs"
        / "shared"
        / str(plan["experiment_name"])
        / str(args.loss)
        / "seed_{}".format(args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "run_receipt.json"
    config_hash = configuration_sha256(
        {
            "config": config,
            "arguments": stable_run_arguments(vars(args)),
            "plan": printable_plan,
        }
    )
    decision = decide_run(output_dir, config_hash, resume=args.resume)
    if decision.action == "skip":
        print("Skipping completed run: {}".format(output_dir))
        return decision.receipt

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    data_path = _data_path(config)
    common_dataset_args = {
        "path": data_path,
        "class_mode": str(plan["class_mode"]),
        "normalization": str(config["normalization"]),
        "include_parameters": False,
    }
    train_dataset = V73SignalDataset(
        indices=plan["train_indices"], **common_dataset_args
    )
    early_dataset = V73SignalDataset(
        indices=plan["early_stop_indices"], **common_dataset_args
    )
    calibration_dataset = V73SignalDataset(
        indices=plan["calibration_indices"], **common_dataset_args
    )
    threshold_dataset = V73SignalDataset(
        indices=plan["threshold_indices"], **common_dataset_args
    )
    loader_args = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.workers),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    early_loader = DataLoader(early_dataset, shuffle=False, **loader_args)
    calibration_loader = DataLoader(
        calibration_dataset, shuffle=False, **loader_args
    )
    threshold_loader = DataLoader(
        threshold_dataset, shuffle=False, **loader_args
    )

    positive_counts = train_dataset.labels6[
        plan["train_indices"]
    ][:, train_dataset.selected_indices].sum(axis=0)
    loss_fn = build_classification_loss(
        args.loss,
        positive_counts=positive_counts,
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.negative_clip,
        beta=args.effective_beta,
    )
    model = build_backbone(
        args.backbone,
        num_classes=int(plan["num_classes"]),
        width=args.width,
        dropout=args.dropout,
    )
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = output_dir / "best_model.pth"
    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=early_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        threshold=0.5,
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
    )

    early_output = predict_loader(model, early_loader, device, use_amp=args.use_amp)
    early_probability = _sigmoid(early_output.logits)
    early_prediction = (early_probability >= 0.5).astype(np.uint8)
    early_metrics = compute_multilabel_metrics(
        early_output.targets,
        early_prediction,
        probabilities=early_probability,
        class_names=plan["class_names"],
    )

    calibration_output = predict_loader(
        model, calibration_loader, device, use_amp=args.use_amp
    )
    temperature = fit_temperature(
        calibration_output.logits,
        calibration_output.targets,
        source_role="calibration",
        max_iter=20,
    )
    threshold_output = predict_loader(
        model, threshold_loader, device, use_amp=args.use_amp
    )
    threshold_probability = _sigmoid(
        apply_temperature(threshold_output.logits, temperature.temperature)
    )
    global_threshold = select_global_threshold(
        threshold_output.targets,
        threshold_probability,
        source_role="threshold_selection",
    )
    shrinkage_threshold = select_shrinkage_per_label_thresholds(
        threshold_output.targets,
        threshold_probability,
        positive_counts=positive_counts,
        beta=100.0,
        source_role="threshold_selection",
    )

    resolved = dict(config)
    resolved.update(
        {
            key: value
            for key, value in vars(args).items()
            if key not in ("dry_run",)
        }
    )
    resolved["plan"] = printable_plan
    resolved["positive_counts"] = positive_counts.tolist()
    _write_json(output_dir / "config_resolved.json", resolved)
    _write_json(output_dir / "metrics.json", early_metrics)
    _write_json(
        output_dir / "calibration.json",
        {
            "temperature": temperature.temperature,
            "loss_before": temperature.loss_before,
            "loss_after": temperature.loss_after,
            "source_role": temperature.source_role,
        },
    )
    _write_json(
        output_dir / "thresholds.json",
        {
            "global": global_threshold.thresholds,
            "global_subset_acc_on_selection": global_threshold.subset_acc,
            "shrinkage_per_label": shrinkage_threshold.thresholds,
            "shrinkage_subset_acc_on_selection": shrinkage_threshold.subset_acc,
            "source_role": "threshold_selection",
        },
    )
    _write_history(output_dir / "history.csv", result.history)
    np.savez_compressed(
        str(output_dir / "predictions.npz"),
        logits=early_output.logits,
        probabilities=early_probability,
        targets=early_output.targets,
        sample_ids=early_output.sample_ids,
    )
    receipt = {
        "status": "completed",
        "output_dir": str(output_dir),
        "config_sha256": config_hash,
        "best_epoch": result.best_epoch,
        "epochs_ran": result.epochs_ran,
        "best_subset_acc": result.best_subset_acc,
        "best_hamming_loss": result.best_hamming_loss,
        "train_seconds": result.train_seconds,
        "parameters": count_trainable_parameters(model),
        "device": str(device),
    }
    _write_json(receipt_path, receipt)
    for dataset in (
        train_dataset,
        early_dataset,
        calibration_dataset,
        threshold_dataset,
    ):
        dataset.close()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "shared" / "four_class.yaml",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "focal", "asl", "cb_bce", "cb_asl"],
        default="bce",
    )
    parser.add_argument("--backbone", default="complex_1d_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=3.0)
    parser.add_argument("--negative-clip", type=float, default=0.05)
    parser.add_argument("--effective-beta", type=float, default=0.99)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
