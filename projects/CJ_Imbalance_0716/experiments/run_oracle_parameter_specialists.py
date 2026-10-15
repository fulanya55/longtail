"""Run oracle-positive MFT/VFT identifiability specialists without classification loss."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.engine.oracle_parameter_trainer import (  # noqa: E402
    ScalarTargetNormalizer,
    compute_specialist_metrics,
    specialist_predictions,
    train_oracle_specialist,
)
from cjlt.engine.run_manager import (  # noqa: E402
    configuration_sha256,
    decide_run,
    stable_run_arguments,
)
from cjlt.models.backbone_adapter import count_trainable_parameters  # noqa: E402
from cjlt.models.oracle_parameter_specialists import build_oracle_specialist  # noqa: E402
from experiments.run_shared import (  # noqa: E402
    _data_path,
    _write_history,
    _write_json,
    build_run_plan,
    load_config,
)


VARIANTS = {
    "MFT_RAW_ORD": {
        "target": "MFT",
        "class_index": 3,
        "parameter_column": 28,
        "target_kind": "ordinal",
        "representation": "raw_iq",
    },
    "MFT_ACF_ORD": {
        "target": "MFT",
        "class_index": 3,
        "parameter_column": 28,
        "target_kind": "ordinal",
        "representation": "multiscale_autocorrelation_and_delay_peaks",
    },
    "VFT_RAW": {
        "target": "VFT",
        "class_index": 4,
        "parameter_column": 49,
        "target_kind": "regression",
        "representation": "raw_iq",
    },
    "VFT_STFT": {
        "target": "VFT",
        "class_index": 4,
        "parameter_column": 49,
        "target_kind": "regression",
        "representation": "stft_spectrogram_and_time_frequency_ridge",
    },
    "VFT_IFREQ": {
        "target": "VFT",
        "class_index": 4,
        "parameter_column": 49,
        "target_kind": "regression",
        "representation": "instantaneous_frequency_sequence",
    },
}


def _indices_sha256(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _target_stratified_limit(indices, target_values, maximum, seed, target_kind):
    values = np.asarray(indices, dtype=np.int64)
    count = int(maximum)
    if count <= 0 or len(values) <= count:
        return values.copy()
    target = np.asarray(target_values, dtype=np.float64)[values]
    if str(target_kind) == "ordinal":
        group_codes = np.rint(target).astype(np.int64)
    else:
        order = np.argsort(target, kind="mergesort")
        group_codes = np.empty(len(values), dtype=np.int64)
        group_codes[order] = np.minimum(
            np.arange(len(values), dtype=np.int64) * 10 // len(values), 9
        )
    rng = np.random.RandomState(int(seed))
    groups = []
    for code in sorted(np.unique(group_codes)):
        group = values[group_codes == code].copy()
        rng.shuffle(group)
        groups.append(group)
    offsets = np.zeros(len(groups), dtype=np.int64)
    selected = []
    while len(selected) < count:
        progressed = False
        for group_index, group in enumerate(groups):
            if offsets[group_index] < len(group):
                selected.append(int(group[offsets[group_index]]))
                offsets[group_index] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _positive_valid_indices(role_indices, labels, parameter_mask, class_index, column):
    indices = np.asarray(role_indices, dtype=np.int64)
    valid = np.logical_and(
        labels[indices, int(class_index)] == 1,
        parameter_mask[indices, int(column)],
    )
    return indices[valid]


def _build_loader(data_path, config, indices, batch_size, workers, shuffle, seed):
    dataset = V73SignalDataset(
        path=data_path,
        indices=indices,
        class_mode=str(config["class_mode"]),
        normalization=str(config["normalization"]),
        include_parameters=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=(torch.Generator().manual_seed(int(seed)) if shuffle else None),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def run_one(args, variant_name, seed):
    config = load_config(args.config)
    variant = VARIANTS[variant_name]
    plan = build_run_plan(config, "oracle_parameter_only", int(seed))
    data_path = _data_path(config)
    labels = read_labels(data_path)
    parameter_values, parameter_mask = read_parameters(data_path)
    role_limits = {
        "train_indices": int(args.max_train_positive),
        "early_stop_indices": int(args.max_eval_positive),
    }
    for offset, role_name in enumerate(("train_indices", "early_stop_indices")):
        positive = _positive_valid_indices(
            plan[role_name],
            labels,
            parameter_mask,
            variant["class_index"],
            variant["parameter_column"],
        )
        selected = _target_stratified_limit(
            positive,
            parameter_values[:, variant["parameter_column"]],
            role_limits[role_name],
            int(seed) + 301 + offset,
            variant["target_kind"],
        )
        plan[role_name] = selected
        plan[role_name.replace("_indices", "_positive_count")] = int(len(selected))
        plan[role_name.replace("_indices", "_sample_sha256")] = _indices_sha256(
            selected
        )

    train_target = parameter_values[
        plan["train_indices"], variant["parameter_column"]
    ].astype(np.float64)
    training_median = float(np.median(train_target))
    normalizer = (
        ScalarTargetNormalizer.fit(train_target)
        if variant["target_kind"] == "regression"
        else None
    )
    output_root = Path(args.output_root)
    output_dir = output_root / variant_name / "seed_{}".format(seed)
    stable_arguments = dict(vars(args))
    stable_arguments.update({"variant": variant_name, "seed": int(seed)})
    printable_plan = {
        key: value for key, value in plan.items() if not key.endswith("_indices")
    }
    config_hash = configuration_sha256(
        {
            "config": config,
            "arguments": stable_run_arguments(stable_arguments),
            "variant": variant,
            "plan": printable_plan,
            "training_median": training_median,
            "normalizer": (
                None
                if normalizer is None
                else {"mean": normalizer.mean, "scale": normalizer.scale}
            ),
        }
    )
    decision = decide_run(output_dir, config_hash, resume=args.resume)
    if decision.action == "skip":
        print("Skipping completed run: {}".format(output_dir))
        return decision.receipt
    if args.dry_run:
        payload = {
            "variant": variant_name,
            "seed": int(seed),
            "variant_flags": variant,
            "plan": printable_plan,
            "training_median": training_median,
            "output_dir": str(output_dir),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    train_dataset, train_loader = _build_loader(
        data_path,
        config,
        plan["train_indices"],
        args.batch_size,
        args.workers,
        True,
        int(seed) + 101,
    )
    validation_dataset, validation_loader = _build_loader(
        data_path,
        config,
        plan["early_stop_indices"],
        args.batch_size,
        args.workers,
        False,
        int(seed),
    )
    model = build_oracle_specialist(variant_name, width=args.width)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training = train_oracle_specialist(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        parameter_column=variant["parameter_column"],
        target_kind=variant["target_kind"],
        normalizer=normalizer,
        training_median=training_median,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=output_dir / "best_model.pth",
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
    )
    prediction = specialist_predictions(
        model,
        validation_loader,
        device,
        variant["parameter_column"],
        variant["target_kind"],
        normalizer=normalizer,
        use_amp=args.use_amp,
    )
    metrics = compute_specialist_metrics(
        prediction["predictions"],
        prediction["targets"],
        training_median,
        variant["target_kind"],
    )
    _write_history(output_dir / "history.csv", training["history"])
    _write_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(str(output_dir / "predictions.npz"), **prediction)
    receipt = {
        "status": "completed",
        "variant": variant_name,
        "target": variant["target"],
        "target_kind": variant["target_kind"],
        "representation": variant["representation"],
        "seed": int(seed),
        "config_sha256": config_hash,
        "split_sha256": plan["split_sha256"],
        "train_positive_count": int(len(plan["train_indices"])),
        "validation_positive_count": int(len(plan["early_stop_indices"])),
        "train_sample_sha256": plan["train_sample_sha256"],
        "validation_sample_sha256": plan["early_stop_sample_sha256"],
        "best_epoch": training["best_epoch"],
        "epochs_ran": training["epochs_ran"],
        "mae": metrics["mae"],
        "pearson_correlation": metrics["pearson_correlation"],
        "constant_median_mae": metrics["constant_median_mae"],
        "relative_mae_reduction_vs_constant": metrics[
            "relative_mae_reduction_vs_constant"
        ],
        "prediction_std": metrics["prediction_std"],
        "target_std": metrics["target_std"],
        "rounded_mae": metrics.get("rounded_mae"),
        "rounded_accuracy": metrics.get("rounded_accuracy"),
        "train_seconds": training["train_seconds"],
        "training_parameters": count_trainable_parameters(model),
        "device": str(device),
        "classification_loss_used": False,
        "oracle_positive_only": True,
        "locked_test_used": False,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    train_dataset.close()
    validation_dataset.close()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def _aggregate(receipts):
    rows = []
    gates = []
    for variant_name in VARIANTS:
        selected = [
            row
            for row in receipts
            if row.get("variant") == variant_name
            and row.get("status") == "completed"
        ]
        if not selected:
            continue
        entry = {
            "variant": variant_name,
            "target": VARIANTS[variant_name]["target"],
            "seed_count": len(selected),
        }
        for key in (
            "mae",
            "constant_median_mae",
            "relative_mae_reduction_vs_constant",
            "pearson_correlation",
            "prediction_std",
            "target_std",
            "rounded_mae",
            "rounded_accuracy",
            "train_seconds",
        ):
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            entry[key] = (
                {"mean": float(np.mean(values)), "std": float(np.std(values))}
                if values
                else None
            )
        reductions = [
            float(row["relative_mae_reduction_vs_constant"]) for row in selected
        ]
        correlations = [float(row["pearson_correlation"]) for row in selected]
        gate = {
            "variant": variant_name,
            "target": VARIANTS[variant_name]["target"],
            "mean_relative_mae_reduction_vs_constant": float(np.mean(reductions)),
            "mean_pearson_correlation": float(np.mean(correlations)),
            "positive_mae_direction_seed_count": int(
                sum(value > 0.0 for value in reductions)
            ),
            "positive_correlation_seed_count": int(
                sum(value > 0.0 for value in correlations)
            ),
            "reattach_to_c2_passed": bool(
                len(selected) == 3
                and float(np.mean(reductions)) >= 0.10
                and float(np.mean(correlations)) >= 0.30
                and all(value > 0.0 for value in reductions)
                and all(value > 0.0 for value in correlations)
            ),
        }
        rows.append(entry)
        gates.append(gate)
    return {"aggregate": rows, "gates": gates, "runs": receipts}


def run_experiment(args):
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    receipts = []
    for seed in args.seeds:
        for variant_name in variants:
            receipts.append(run_one(args, variant_name, int(seed)))
    if args.dry_run:
        return receipts
    summary = _aggregate(receipts)
    summary.update(
        {
            "status": "completed",
            "locked_test_used": False,
            "classification_loss_used": False,
            "protocol": {
                "variants": variants,
                "seeds": [int(seed) for seed in args.seeds],
                "epochs": int(args.epochs),
                "max_train_positive": int(args.max_train_positive),
                "max_eval_positive": int(args.max_eval_positive),
                "gate": {
                    "mean_mae_reduction_vs_constant_minimum": 0.10,
                    "mean_correlation_minimum": 0.30,
                    "positive_direction_required": "3/3 seeds for both metrics",
                },
            },
        }
    )
    _write_json(Path(args.output_root) / "oracle_specialist_summary.json", summary)
    print(
        json.dumps(
            {"aggregate": summary["aggregate"], "gates": summary["gates"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "student_b"
        / "supervised_joint_six_class.yaml",
    )
    parser.add_argument("--variant", choices=["all"] + list(VARIANTS), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--max-train-positive", type=int, default=2048)
    parser.add_argument("--max-eval-positive", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "student_b"
        / "oracle_parameter_specialists_20260723",
    )
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
