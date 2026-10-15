"""Screen pre-pooling temporal evidence for the strongest C2 parameter targets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import get_class_names, select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.supervised_joint import (  # noqa: E402
    SupervisedJointParameterModel,
    temporal_local_distribution,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


TARGETS = (
    ("FMB", "interference_bandwidth"),
    ("ISRJ", "sample_width"),
    ("SinFM", "frequency_deviation"),
)
REPRESENTATIONS = ("global_avg", "local_distribution")
DEFAULT_CHECKPOINTS = {
    2041: PROJECT_ROOT
    / "outputs/student_b/supervised_joint_parameter_confirm_seed2041/C2/seed_2041",
    2042: PROJECT_ROOT
    / "outputs/student_b/supervised_joint_parameter_confirm_final/C2/seed_2042",
    2043: PROJECT_ROOT
    / "outputs/student_b/supervised_joint_parameter_confirm_final/C2/seed_2043",
}


class ScalarParameterProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features):
        return self.regressor(features).squeeze(-1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _write_history(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_model(run_dir, device):
    backbone = build_backbone(
        "complex_1d_true", num_classes=6, width=16, dropout=0.2
    )
    model = SupervisedJointParameterModel(
        backbone,
        relation_dim=64,
        parameters_per_label=2,
        enable_parameter_heads=True,
        relation_dropout=0.0,
    )
    checkpoint_path = Path(run_dir) / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_path


def _normalizer_lookup(run_dir):
    rows = json.loads(
        (Path(run_dir) / "parameter_normalizer.json").read_text(encoding="utf-8")
    )
    return {
        (row["class"], row["parameter"]): {
            "mean": float(row["mean"]),
            "scale": float(row["scale"]),
        }
        for row in rows
    }


def _extract_representations(model, data_path, indices, normalization, device, batch_size):
    dataset = V73SignalDataset(
        path=data_path,
        indices=np.asarray(indices, dtype=np.int64),
        class_mode="six",
        normalization=normalization,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    global_parts = []
    local_parts = []
    sample_parts = []
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type="cuda", enabled=device.type == "cuda"
                ):
                    temporal = model.backbone.forward_temporal_features(signal)
                    global_features = temporal.mean(dim=-1)
                    local_features = temporal_local_distribution(temporal)
                global_parts.append(global_features.float().cpu().numpy())
                local_parts.append(local_features.float().cpu().numpy())
                sample_parts.append(
                    torch.as_tensor(batch["sample_index"], dtype=torch.int64).numpy()
                )
    finally:
        dataset.close()
    return {
        "global_avg": np.concatenate(global_parts).astype(np.float32),
        "local_distribution": np.concatenate(local_parts).astype(np.float32),
        "sample_ids": np.concatenate(sample_parts).astype(np.int64),
    }


def _quantile_split(values, seed, validation_fraction=0.2, bins=10):
    target = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    edges = np.unique(np.quantile(target, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) <= 2:
        strata = np.zeros(len(target), dtype=np.int64)
    else:
        strata = np.digitize(target, edges[1:-1], right=True)
    fit = []
    validation = []
    for stratum in sorted(np.unique(strata)):
        group = np.flatnonzero(strata == stratum)
        rng.shuffle(group)
        count = max(1, int(round(len(group) * float(validation_fraction))))
        if len(group) > 1:
            count = min(count, len(group) - 1)
        validation.extend(group[:count].tolist())
        fit.extend(group[count:].tolist())
    return np.asarray(fit, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def _metrics(prediction, target, training_median, scale):
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    error = np.abs(predicted - expected)
    constant_error = np.abs(float(training_median) - expected)
    correlation = (
        float(np.corrcoef(predicted, expected)[0, 1])
        if len(expected) > 2 and predicted.std() > 0.0 and expected.std() > 0.0
        else 0.0
    )
    return {
        "count": int(len(expected)),
        "mae": float(error.mean()),
        "normalized_mae": float(error.mean() / float(scale)),
        "pearson_correlation": correlation,
        "prediction_std": float(predicted.std()),
        "constant_median_mae": float(constant_error.mean()),
        "relative_mae_reduction_vs_constant": float(
            1.0 - error.mean() / max(constant_error.mean(), 1.0e-12)
        ),
    }


def _predict(model, features, mean, std, target_mean, target_scale, device, batch_size):
    model.eval()
    values = []
    normalized = (features - mean) / std
    with torch.no_grad():
        for start in range(0, len(normalized), int(batch_size)):
            batch = torch.from_numpy(
                normalized[start : start + int(batch_size)].astype(np.float32)
            ).to(device)
            values.append(model(batch).cpu().numpy())
    prediction = np.concatenate(values)
    return prediction * float(target_scale) + float(target_mean)


def _train_probe(
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    target_mean,
    target_scale,
    seed,
    device,
    args,
):
    mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train_features.std(axis=0, keepdims=True), 1.0e-5).astype(
        np.float32
    )
    train_x = ((train_features - mean) / std).astype(np.float32)
    train_y = ((train_targets - target_mean) / target_scale).astype(np.float32)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = ScalarParameterProbe(
        input_dim=train_x.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = F.smooth_l1_loss(prediction, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = _predict(
            model,
            validation_features,
            mean,
            std,
            target_mean,
            target_scale,
            device,
            args.batch_size,
        )
        validation_mae = float(
            np.abs(validation_prediction - validation_targets).mean()
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_mae": validation_mae,
            }
        )
        if validation_mae < best_mae - 1.0e-12:
            best_mae = validation_mae
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, history


def _target_info(class_names, class_name, parameter_name):
    class_index = list(class_names).index(class_name)
    specs = get_parameter_specs(class_name)
    parameter_index = [spec.name for spec in specs].index(parameter_name)
    return class_index, parameter_index, specs[parameter_index].column_index


def _run_seed(args, seed, config, data_path, labels, values79, mask79, device):
    run_dir = (
        DEFAULT_CHECKPOINTS[int(seed)]
        if args.checkpoint_root is None
        else Path(args.checkpoint_root) / "seed_{}".format(seed)
    )
    output_dir = args.output_root / "seed_{}".format(seed)
    receipt_path = output_dir / "run_receipt.json"
    if receipt_path.exists() and args.resume:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    plan = build_run_plan(config, "bce", int(seed))
    model, checkpoint_path = _load_model(run_dir, device)
    normalizers = _normalizer_lookup(run_dir)
    train_ids = np.asarray(plan["train_indices"], dtype=np.int64)
    eval_ids = np.asarray(plan["early_stop_indices"], dtype=np.int64)
    started = time.perf_counter()
    train_representations = _extract_representations(
        model,
        data_path,
        train_ids,
        config["normalization"],
        device,
        args.feature_batch_size,
    )
    eval_representations = _extract_representations(
        model,
        data_path,
        eval_ids,
        config["normalization"],
        device,
        args.feature_batch_size,
    )
    if not np.array_equal(train_representations["sample_ids"], train_ids):
        raise ValueError("training feature/sample alignment failed")
    if not np.array_equal(eval_representations["sample_ids"], eval_ids):
        raise ValueError("evaluation feature/sample alignment failed")
    class_names = get_class_names("six")
    rows = []
    for target_offset, (class_name, parameter_name) in enumerate(TARGETS):
        class_index, _, column_index = _target_info(
            class_names, class_name, parameter_name
        )
        train_valid = (
            (labels[train_ids, class_index] == 1)
            & mask79[train_ids, column_index]
            & np.isfinite(values79[train_ids, column_index])
        )
        eval_valid = (
            (labels[eval_ids, class_index] == 1)
            & mask79[eval_ids, column_index]
            & np.isfinite(values79[eval_ids, column_index])
        )
        target_train = values79[train_ids[train_valid], column_index].astype(np.float32)
        target_eval = values79[eval_ids[eval_valid], column_index].astype(np.float32)
        fit, validation = _quantile_split(
            target_train, seed=71000 + int(seed) + target_offset * 100
        )
        normalizer = normalizers[(class_name, parameter_name)]
        training_median = float(np.median(target_train[fit]))
        target_payload = {
            "class": class_name,
            "parameter": parameter_name,
            "parameter_column": int(column_index),
            "train_positive_valid": int(len(target_train)),
            "fit_samples": int(len(fit)),
            "validation_samples": int(len(validation)),
            "eval_positive_valid": int(len(target_eval)),
            "constant": _metrics(
                np.full_like(target_eval, training_median),
                target_eval,
                training_median,
                normalizer["scale"],
            ),
            "representations": {},
        }
        for representation_offset, representation in enumerate(REPRESENTATIONS):
            features_train = train_representations[representation][train_valid]
            features_eval = eval_representations[representation][eval_valid]
            probe_seed = (
                72000
                + int(seed)
                + target_offset * 1000
                + representation_offset * 100
            )
            probe, mean, std, best_epoch, history = _train_probe(
                features_train[fit],
                target_train[fit],
                features_train[validation],
                target_train[validation],
                normalizer["mean"],
                normalizer["scale"],
                probe_seed,
                device,
                args,
            )
            prediction = _predict(
                probe,
                features_eval,
                mean,
                std,
                normalizer["mean"],
                normalizer["scale"],
                device,
                args.batch_size,
            )
            metrics = _metrics(
                prediction,
                target_eval,
                training_median,
                normalizer["scale"],
            )
            metrics.update(
                {
                    "feature_dim": int(features_train.shape[1]),
                    "best_epoch": int(best_epoch),
                    "probe_seed": int(probe_seed),
                }
            )
            target_payload["representations"][representation] = metrics
            _write_history(
                output_dir
                / "{}_{}_{}_history.csv".format(
                    class_name, parameter_name, representation
                ),
                history,
            )
        local = target_payload["representations"]["local_distribution"]
        global_value = target_payload["representations"]["global_avg"]
        target_payload["local_relative_nmae_reduction_vs_global"] = float(
            1.0
            - local["normalized_mae"]
            / max(global_value["normalized_mae"], 1.0e-12)
        )
        rows.append(target_payload)
    receipt = {
        "status": "completed",
        "seed": int(seed),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split_sha256": plan["split_sha256"],
        "train_samples": int(len(train_ids)),
        "eval_samples": int(len(eval_ids)),
        "targets": rows,
        "feature_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "locked_test_used": False,
        "output_dir": str(output_dir),
    }
    _write_json(receipt_path, receipt)
    return receipt


def _aggregate(receipts):
    rows = []
    for class_name, parameter_name in TARGETS:
        selected = [
            target
            for receipt in receipts
            for target in receipt["targets"]
            if target["class"] == class_name
            and target["parameter"] == parameter_name
        ]
        reductions = np.asarray(
            [row["local_relative_nmae_reduction_vs_global"] for row in selected],
            dtype=np.float64,
        )
        entry = {
            "class": class_name,
            "parameter": parameter_name,
            "seed_count": int(len(selected)),
            "local_relative_nmae_reduction_vs_global": {
                "mean": float(reductions.mean()),
                "std": float(reductions.std()),
                "values": reductions.tolist(),
            },
            "positive_seed_count": int((reductions > 0.0).sum()),
            "five_percent_seed_count": int((reductions >= 0.05).sum()),
            "promotion_passed": bool(
                len(reductions) == 3
                and bool((reductions > 0.0).all())
                and float(reductions.mean()) >= 0.05
            ),
            "representations": {},
        }
        for representation in REPRESENTATIONS:
            values = np.asarray(
                [
                    row["representations"][representation]["normalized_mae"]
                    for row in selected
                ],
                dtype=np.float64,
            )
            correlations = np.asarray(
                [
                    row["representations"][representation]["pearson_correlation"]
                    for row in selected
                ],
                dtype=np.float64,
            )
            entry["representations"][representation] = {
                "normalized_mae_mean": float(values.mean()),
                "normalized_mae_std": float(values.std()),
                "correlation_mean": float(correlations.mean()),
            }
        rows.append(entry)
    return rows


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    labels6 = read_labels(data_path)
    labels = select_labels(labels6, "six")
    values79, mask79 = read_parameters(data_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    receipts = []
    for seed in args.seeds:
        print("local parameter probe seed={}".format(seed), flush=True)
        receipts.append(
            _run_seed(
                args,
                int(seed),
                config,
                data_path,
                labels,
                values79,
                mask79,
                device,
            )
        )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "targets": [list(target) for target in TARGETS],
            "representations": list(REPRESENTATIONS),
            "promotion_rule": "mean local NMAE reduction >=5% and 3/3 positive",
            "encoder": "frozen full-data C2 checkpoint",
            "evaluation_role": "early_stop development role",
        },
        "aggregate": _aggregate(receipts),
        "runs": receipts,
    }
    _write_json(args.output_root / "route_b_c2_local_parameter_probe_summary.json", payload)
    print(json.dumps({"aggregate": payload["aggregate"]}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/student_b/supervised_joint_six_class.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/student_b/route_b_c2_local_parameter_probe_20260725",
    )
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
