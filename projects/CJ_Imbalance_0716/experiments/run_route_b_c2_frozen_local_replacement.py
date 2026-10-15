"""Train frozen local experts and replace only validated C2 parameter outputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.registry import get_class_names, select_labels  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402
from experiments.run_route_b_c2_local_parameter_probe import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    TARGETS,
    _extract_representations,
    _load_model,
    _metrics,
    _predict,
    _quantile_split,
    _sha256,
    _target_info,
    _train_probe,
    _write_history,
    _write_json,
)


PROMOTED_TARGETS = (
    ("FMB", "interference_bandwidth"),
    ("SinFM", "frequency_deviation"),
)


def _normalizers(run_dir):
    rows = json.loads(
        (Path(run_dir) / "parameter_normalizer.json").read_text(encoding="utf-8")
    )
    class_names = get_class_names("six")
    lookup = {(row["class"], row["parameter"]): row for row in rows}
    mean = np.zeros((len(class_names), 2), dtype=np.float64)
    scale = np.ones((len(class_names), 2), dtype=np.float64)
    for class_index, class_name in enumerate(class_names):
        for parameter_index, spec in enumerate(get_parameter_specs(class_name)):
            row = lookup[(class_name, spec.name)]
            mean[class_index, parameter_index] = float(row["mean"])
            scale[class_index, parameter_index] = float(row["scale"])
    return lookup, mean, scale


def _joint_metrics(prediction, target, parameter_mask, labels, class_prediction, scale):
    valid = np.asarray(parameter_mask, dtype=bool) & (
        np.asarray(labels, dtype=np.uint8)[:, :, None] == 1
    )
    normalized_error = np.abs(
        np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    ) / np.asarray(scale, dtype=np.float64)[None, :, :]
    per_target = []
    for class_index, class_name in enumerate(get_class_names("six")):
        specs = get_parameter_specs(class_name)
        for parameter_index, spec in enumerate(specs):
            selected = valid[:, class_index, parameter_index]
            per_target.append(
                {
                    "class": class_name,
                    "parameter": spec.name,
                    "count": int(selected.sum()),
                    "normalized_mae": float(
                        normalized_error[selected, class_index, parameter_index].mean()
                    ),
                    "mae": float(
                        np.abs(
                            prediction[selected, class_index, parameter_index]
                            - target[selected, class_index, parameter_index]
                        ).mean()
                    ),
                }
            )
    detected = valid & (np.asarray(class_prediction, dtype=np.uint8)[:, :, None] == 1)
    return {
        "oracle_macro_target_nmae": float(
            np.mean([row["normalized_mae"] for row in per_target])
        ),
        "oracle_micro_nmae": float(normalized_error[valid].mean()),
        "system_parameter_coverage": float(detected.sum() / max(valid.sum(), 1)),
        "system_detected_micro_nmae": float(normalized_error[detected].mean()),
        "per_target": per_target,
    }


def _save_expert(path, model, feature_mean, feature_std, metadata):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "feature_mean": torch.from_numpy(feature_mean.astype(np.float32)),
            "feature_std": torch.from_numpy(feature_std.astype(np.float32)),
            **metadata,
        },
        str(target),
    )


def _run_seed(args, seed, config, data_path, labels, values79, mask79, device):
    run_dir = (
        DEFAULT_CHECKPOINTS[int(seed)]
        if args.checkpoint_root is None
        else Path(args.checkpoint_root) / "seed_{}".format(seed)
    )
    output_dir = args.output_root / "seed_{}".format(seed)
    receipt_path = output_dir / "run_receipt.json"
    if args.resume and receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    plan = build_run_plan(config, "bce", int(seed))
    train_ids = np.asarray(plan["train_indices"], dtype=np.int64)
    eval_ids = np.asarray(plan["early_stop_indices"], dtype=np.int64)
    model, checkpoint_path = _load_model(run_dir, device)
    representations = _extract_representations(
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
    if not np.array_equal(representations["sample_ids"], train_ids):
        raise ValueError("training sample alignment failed")
    if not np.array_equal(eval_representations["sample_ids"], eval_ids):
        raise ValueError("evaluation sample alignment failed")

    base_path = Path(run_dir) / "predictions.npz"
    with np.load(base_path) as saved:
        base = {key: saved[key].copy() for key in saved.files}
    if not np.array_equal(base["sample_ids"].astype(np.int64), eval_ids):
        raise ValueError("base C2 prediction/sample alignment failed")
    fused_prediction = base["parameter_predictions"].astype(np.float32).copy()
    normalizer_lookup, _, scales = _normalizers(run_dir)
    class_names = get_class_names("six")
    target_rows = []
    for target_offset, (class_name, parameter_name) in enumerate(TARGETS):
        if (class_name, parameter_name) not in PROMOTED_TARGETS:
            continue
        class_index, parameter_index, column_index = _target_info(
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
        train_target = values79[train_ids[train_valid], column_index].astype(np.float32)
        eval_target = values79[eval_ids[eval_valid], column_index].astype(np.float32)
        fit, validation = _quantile_split(
            train_target, seed=71000 + int(seed) + target_offset * 100
        )
        normalizer = normalizer_lookup[(class_name, parameter_name)]
        training_median = float(np.median(train_target[fit]))
        feature_train = representations["local_distribution"][train_valid]
        feature_eval = eval_representations["local_distribution"][eval_valid]
        probe_seed = 72000 + int(seed) + target_offset * 1000 + 100
        probe, feature_mean, feature_std, best_epoch, history = _train_probe(
            feature_train[fit],
            train_target[fit],
            feature_train[validation],
            train_target[validation],
            float(normalizer["mean"]),
            float(normalizer["scale"]),
            probe_seed,
            device,
            args,
        )
        local_prediction = _predict(
            probe,
            feature_eval,
            feature_mean,
            feature_std,
            float(normalizer["mean"]),
            float(normalizer["scale"]),
            device,
            args.batch_size,
        )
        c2_prediction = base["parameter_predictions"][
            eval_valid, class_index, parameter_index
        ]
        local_metrics = _metrics(
            local_prediction,
            eval_target,
            training_median,
            float(normalizer["scale"]),
        )
        c2_metrics = _metrics(
            c2_prediction,
            eval_target,
            training_median,
            float(normalizer["scale"]),
        )
        fused_prediction[eval_valid, class_index, parameter_index] = local_prediction
        expert_path = output_dir / "{}_{}_local_expert.pth".format(
            class_name, parameter_name
        )
        _save_expert(
            expert_path,
            probe,
            feature_mean,
            feature_std,
            {
                "class": class_name,
                "parameter": parameter_name,
                "class_index": int(class_index),
                "parameter_index": int(parameter_index),
                "target_mean": float(normalizer["mean"]),
                "target_scale": float(normalizer["scale"]),
                "probe_seed": int(probe_seed),
                "best_epoch": int(best_epoch),
                "source_c2_checkpoint_sha256": _sha256(checkpoint_path),
            },
        )
        _write_history(
            output_dir / "{}_{}_history.csv".format(class_name, parameter_name),
            history,
        )
        target_rows.append(
            {
                "class": class_name,
                "parameter": parameter_name,
                "eval_positive_valid": int(eval_valid.sum()),
                "c2": c2_metrics,
                "frozen_local": local_metrics,
                "relative_nmae_reduction_vs_c2": float(
                    1.0
                    - local_metrics["normalized_mae"]
                    / max(c2_metrics["normalized_mae"], 1.0e-12)
                ),
                "expert_path": str(expert_path),
                "expert_sha256": _sha256(expert_path),
            }
        )

    base_metrics = _joint_metrics(
        base["parameter_predictions"],
        base["parameter_targets"],
        base["parameter_masks"],
        base["targets"],
        base["predictions"],
        scales,
    )
    fused_metrics = _joint_metrics(
        fused_prediction,
        base["parameter_targets"],
        base["parameter_masks"],
        base["targets"],
        base["predictions"],
        scales,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fused_payload = dict(base)
    fused_payload["base_parameter_predictions"] = base[
        "parameter_predictions"
    ]
    fused_payload["parameter_predictions"] = fused_prediction
    np.savez_compressed(
        str(output_dir / "fused_predictions.npz"), **fused_payload
    )
    receipt = {
        "status": "completed",
        "seed": int(seed),
        "source_c2_checkpoint": str(checkpoint_path),
        "source_c2_checkpoint_sha256": _sha256(checkpoint_path),
        "source_predictions_sha256": _sha256(base_path),
        "split_sha256": plan["split_sha256"],
        "targets": target_rows,
        "base_c2": base_metrics,
        "frozen_local_replacement": fused_metrics,
        "overall_macro_nmae_reduction": float(
            base_metrics["oracle_macro_target_nmae"]
            - fused_metrics["oracle_macro_target_nmae"]
        ),
        "overall_relative_macro_nmae_reduction": float(
            1.0
            - fused_metrics["oracle_macro_target_nmae"]
            / base_metrics["oracle_macro_target_nmae"]
        ),
        "classification_outputs_unchanged": True,
        "unchanged_parameter_target_count": 10,
        "device": str(device),
        "locked_test_used": False,
        "output_dir": str(output_dir),
    }
    _write_json(receipt_path, receipt)
    return receipt


def _aggregate(receipts):
    targets = []
    for class_name, parameter_name in PROMOTED_TARGETS:
        selected = [
            row
            for receipt in receipts
            for row in receipt["targets"]
            if row["class"] == class_name and row["parameter"] == parameter_name
        ]
        reductions = np.asarray(
            [row["relative_nmae_reduction_vs_c2"] for row in selected],
            dtype=np.float64,
        )
        targets.append(
            {
                "class": class_name,
                "parameter": parameter_name,
                "relative_nmae_reduction_vs_c2_mean": float(reductions.mean()),
                "relative_nmae_reduction_vs_c2_std": float(reductions.std()),
                "values": reductions.tolist(),
                "positive_seed_count": int((reductions > 0.0).sum()),
                "promotion_passed": bool(
                    len(reductions) == 3
                    and bool((reductions > 0.0).all())
                    and float(reductions.mean()) >= 0.05
                ),
            }
        )
    base = np.asarray(
        [row["base_c2"]["oracle_macro_target_nmae"] for row in receipts],
        dtype=np.float64,
    )
    fused = np.asarray(
        [
            row["frozen_local_replacement"]["oracle_macro_target_nmae"]
            for row in receipts
        ],
        dtype=np.float64,
    )
    detected_base = np.asarray(
        [row["base_c2"]["system_detected_micro_nmae"] for row in receipts],
        dtype=np.float64,
    )
    detected_fused = np.asarray(
        [
            row["frozen_local_replacement"]["system_detected_micro_nmae"]
            for row in receipts
        ],
        dtype=np.float64,
    )
    return {
        "targets": targets,
        "base_c2_macro_target_nmae": {
            "mean": float(base.mean()),
            "std": float(base.std()),
        },
        "frozen_local_macro_target_nmae": {
            "mean": float(fused.mean()),
            "std": float(fused.std()),
        },
        "overall_relative_macro_nmae_reduction": float(
            1.0 - fused.mean() / base.mean()
        ),
        "positive_overall_seed_count": int((fused < base).sum()),
        "base_system_detected_micro_nmae": float(detected_base.mean()),
        "frozen_local_system_detected_micro_nmae": float(detected_fused.mean()),
        "system_detected_relative_nmae_reduction": float(
            1.0 - detected_fused.mean() / detected_base.mean()
        ),
    }


def run(args):
    config = load_config(args.config)
    data_path = _data_path(config)
    labels = select_labels(read_labels(data_path), "six")
    values79, mask79 = read_parameters(data_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    receipts = []
    for seed in args.seeds:
        print("frozen local replacement seed={}".format(seed), flush=True)
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
            "promoted_targets": [list(value) for value in PROMOTED_TARGETS],
            "source": "frozen full-data C2 checkpoints",
            "classification_outputs_unchanged": True,
            "unchanged_parameter_target_count": 10,
            "evaluation_role": "early_stop development role",
        },
        "aggregate": _aggregate(receipts),
        "runs": receipts,
    }
    _write_json(
        args.output_root / "route_b_c2_frozen_local_replacement_summary.json",
        payload,
    )
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
        / "outputs/student_b/route_b_c2_frozen_local_replacement_20260725",
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
