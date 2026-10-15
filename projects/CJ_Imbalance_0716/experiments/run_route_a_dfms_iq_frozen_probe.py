"""Frozen linear/MLP probe for the DFMS-IQ structure screen."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import (  # noqa: E402
    V73SignalDataset,
    read_labels,
    read_parameters,
)
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_structure_screen import (  # noqa: E402
    CLASS_NAMES,
    MODELS,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


PROBE_TYPES = ("linear", "mlp")


class LabelWiseProbe(nn.Module):
    def __init__(self, input_dim, num_labels, probe_type, hidden_dim, dropout):
        super().__init__()
        self.probe_type = str(probe_type)
        if self.probe_type == "linear":
            self.head = nn.Linear(int(input_dim), int(num_labels))
        elif self.probe_type == "mlp":
            self.head = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(input_dim), int(hidden_dim)),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(int(hidden_dim), 1),
                    )
                    for _ in range(int(num_labels))
                ]
            )
        else:
            raise ValueError("probe_type must be linear or mlp")

    def forward(self, features):
        if self.probe_type == "linear":
            return self.head(features)
        return torch.cat([head(features) for head in self.head], dim=1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_power(values79):
    values = np.asarray(values79, dtype=np.float32)
    return np.stack(
        [values[:, get_parameter_specs(name)[0].column_index] for name in CLASS_NAMES],
        axis=1,
    )


def _load_backbone(run_dir, model_name, device):
    spec = MODELS[model_name]
    model = build_backbone(
        spec["backbone"], num_classes=6, width=spec["width"], dropout=0.2
    )
    checkpoint_path = run_dir / "best_model.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    full_state = checkpoint["model_state_dict"]
    prefix = "backbone."
    state = {
        key[len(prefix) :]: value
        for key, value in full_state.items()
        if key.startswith(prefix)
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_path


def _extract_features(model, data_path, indices, normalization, device, batch_size):
    dataset = V73SignalDataset(
        path=data_path,
        indices=indices,
        class_mode="six",
        normalization=normalization,
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0
    )
    features = []
    targets = []
    sample_ids = []
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = batch["signal"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    output = model.forward_features(inputs)
                features.append(output.detach().cpu().numpy().astype(np.float32))
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(
                    np.asarray(batch["sample_index"], dtype=np.int64)
                )
    finally:
        dataset.close()
    return (
        np.concatenate(features, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(sample_ids, axis=0),
    )


def _macro_aupr(targets, probabilities):
    return float(
        np.mean(
            [
                average_precision_score(targets[:, index], probabilities[:, index])
                for index in range(targets.shape[1])
            ]
        )
    )


def _evaluate(targets, probabilities, weak_mask):
    positive = targets.astype(bool)
    weak_positive = positive & weak_mask
    negative = ~positive
    weak_scores = probabilities[weak_positive]
    negative_scores = probabilities[negative]
    binary_scores = np.concatenate([weak_scores, negative_scores])
    binary_targets = np.concatenate(
        [np.ones(weak_scores.size), np.zeros(negative_scores.size)]
    )
    per_label = {}
    for index, name in enumerate(CLASS_NAMES):
        label_weak = weak_positive[:, index]
        label_negative = negative[:, index]
        label_scores = np.concatenate(
            [probabilities[label_weak, index], probabilities[label_negative, index]]
        )
        label_targets = np.concatenate(
            [np.ones(label_weak.sum()), np.zeros(label_negative.sum())]
        )
        per_label[name] = {
            "aupr": float(
                average_precision_score(targets[:, index], probabilities[:, index])
            ),
            "weak_vs_negative_auroc": float(
                roc_auc_score(label_targets, label_scores)
            ),
            "weak_vs_negative_aupr": float(
                average_precision_score(label_targets, label_scores)
            ),
        }
    return {
        "macro_aupr": _macro_aupr(targets, probabilities),
        "weak_vs_negative_auroc": float(
            roc_auc_score(binary_targets, binary_scores)
        ),
        "weak_vs_negative_aupr": float(
            average_precision_score(binary_targets, binary_scores)
        ),
        "per_label": per_label,
    }


def _train_probe(
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    probe_type,
    seed,
    device,
    args,
):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = LabelWiseProbe(
        train_features.shape[1],
        train_targets.shape[1],
        probe_type=probe_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    positives = train_targets.sum(axis=0).astype(np.float64)
    negatives = len(train_targets) - positives
    pos_weight = torch.as_tensor(
        negatives / np.maximum(positives, 1.0),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32)),
        torch.from_numpy(train_targets.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_x = torch.from_numpy(validation_features.astype(np.float32)).to(device)
    best_state = None
    best_epoch = 0
    best_macro = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(val_x)).cpu().numpy()
        val_macro = _macro_aupr(validation_targets, probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_aupr": val_macro,
            }
        )
        if val_macro > best_macro + 1e-8:
            best_macro = val_macro
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
    model.load_state_dict(best_state)
    return model, best_epoch, history


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(args, context, model_name, seed, device):
    run_dir = args.screen_root / "runs" / model_name / "seed_{}".format(seed)
    output_dir = args.output_root / model_name / "seed_{}".format(seed)
    metrics_path = output_dir / "probe_metrics.json"
    if metrics_path.exists():
        return _read(metrics_path)
    manifest = _read(run_dir / "longtail_manifest.json")
    train_ids = np.asarray(manifest["selected_indices"], dtype=np.int64)
    saved_predictions = np.load(run_dir / "predictions.npz")
    eval_ids = saved_predictions["sample_ids"].astype(np.int64)
    model, checkpoint_path = _load_backbone(run_dir, model_name, device)
    train_features, train_targets, extracted_train_ids = _extract_features(
        model,
        context["data_path"],
        train_ids,
        context["normalization"],
        device,
        args.feature_batch_size,
    )
    eval_features, eval_targets, extracted_eval_ids = _extract_features(
        model,
        context["data_path"],
        eval_ids,
        context["normalization"],
        device,
        args.feature_batch_size,
    )
    if not np.array_equal(train_ids, extracted_train_ids):
        raise ValueError("Train feature/sample alignment failed")
    if not np.array_equal(eval_ids, extracted_eval_ids):
        raise ValueError("Eval feature/sample alignment failed")
    if not np.array_equal(eval_targets, context["labels"][eval_ids]):
        raise ValueError("Eval label/sample alignment failed")

    rng = np.random.default_rng(57000 + int(seed))
    order = rng.permutation(len(train_features))
    split = int(round(len(order) * 0.8))
    fit_indices = order[:split]
    validation_indices = order[split:]
    mean = train_features[fit_indices].mean(axis=0, keepdims=True)
    std = train_features[fit_indices].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-5)
    normalized_train = (train_features - mean) / std
    normalized_eval = (eval_features - mean) / std

    payload = {
        "model": model_name,
        "seed": int(seed),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "feature_dim": int(train_features.shape[1]),
        "train_samples": int(len(fit_indices)),
        "validation_samples": int(len(validation_indices)),
        "eval_samples": int(len(eval_features)),
        "locked_test_used": False,
        "probes": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for probe_index, probe_type in enumerate(PROBE_TYPES):
        probe_seed = 61000 + int(seed) + probe_index * 1000
        probe, best_epoch, history = _train_probe(
            normalized_train[fit_indices],
            train_targets[fit_indices],
            normalized_train[validation_indices],
            train_targets[validation_indices],
            probe_type,
            probe_seed,
            device,
            args,
        )
        probe.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(
                probe(torch.from_numpy(normalized_eval.astype(np.float32)).to(device))
            ).cpu().numpy()
        metrics = _evaluate(
            eval_targets,
            probabilities,
            context["weak_mask"][eval_ids],
        )
        metrics["best_epoch"] = int(best_epoch)
        metrics["probe_seed"] = int(probe_seed)
        payload["probes"][probe_type] = metrics
        _write_history(output_dir / "{}_history.csv".format(probe_type), history)
    _write(metrics_path, payload)
    return payload


def _aggregate(rows, probe_type):
    aggregate = {}
    for model_name in MODELS:
        model_rows = [row["probes"][probe_type] for row in rows if row["model"] == model_name]
        aggregate[model_name] = {
            metric: _mean_ci(
                [row[metric] for row in model_rows],
                65000 + list(MODELS).index(model_name) * 100 + index,
            )
            for index, metric in enumerate(
                ("macro_aupr", "weak_vs_negative_auroc", "weak_vs_negative_aupr")
            )
        }
        aggregate[model_name]["per_label_weak_auroc"] = {
            name: _mean_ci(
                [row["per_label"][name]["weak_vs_negative_auroc"] for row in model_rows],
                66000 + list(MODELS).index(model_name) * 100 + index,
            )
            for index, name in enumerate(CLASS_NAMES)
        }
    return aggregate


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    labels = select_labels(read_labels(data_path), "six")
    values79, _ = read_parameters(data_path)
    context = {
        "data_path": data_path,
        "normalization": config["normalization"],
        "labels": labels,
        "weak_mask": compute_weak_component_mask(
            labels, _component_power(values79), args.weak_threshold_db
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    total = len(MODELS) * len(args.seeds)
    current = 0
    for seed in args.seeds:
        for model_name in MODELS:
            current += 1
            print(
                "[{}/{}] frozen probe model={} seed={}".format(
                    current, total, model_name, seed
                ),
                flush=True,
            )
            rows.append(_run_one(args, context, model_name, seed, device))
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "device": str(device),
        "protocol": {
            "probe_types": list(PROBE_TYPES),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "weak_threshold_db": args.weak_threshold_db,
        },
        "per_run": rows,
        "aggregate": {
            probe_type: _aggregate(rows, probe_type) for probe_type in PROBE_TYPES
        },
    }
    _write(args.output_root / "route_a_dfms_iq_frozen_probe_summary.json", payload)
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
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_dfms_iq_structure_screen"
        / "frozen_probe",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
