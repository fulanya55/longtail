"""Student B entrypoint for low-label DAPS-CJNet-v2 experiments."""

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

from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import get_class_names, select_labels  # noqa: E402
from cjlt.data.semi_supervised_split import (  # noqa: E402
    HiddenLabelDataset,
    build_nested_labeled_splits,
)
from cjlt.data.v73_dataset import (  # noqa: E402
    V73SignalDataset,
    read_labels,
    read_parameters,
)
from cjlt.engine.student_b_trainer import train_student_b  # noqa: E402
from cjlt.engine.run_manager import (  # noqa: E402
    configuration_sha256,
    decide_run,
    stable_run_arguments,
)
from cjlt.losses.classification import build_classification_loss  # noqa: E402
from cjlt.losses.parameter import ParameterNormalizer, select_parameter_targets  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone, count_trainable_parameters  # noqa: E402
from cjlt.models.student_b import (  # noqa: E402
    StudentBModel,
    count_student_b_deployment_parameters,
)
from experiments.run_shared import (  # noqa: E402
    _data_path,
    _limit,
    _write_history,
    _write_json,
    build_run_plan,
    load_config,
)


def _deterministic_stratified_limit(indices, labels, maximum, seed):
    values = np.asarray(indices, dtype=np.int64)
    count = int(maximum)
    if count <= 0 or len(values) <= count:
        return values.copy()
    target = np.asarray(labels, dtype=np.uint8)
    powers = np.left_shift(1, np.arange(target.shape[1], dtype=np.int64))
    codes = (target[values].astype(np.int64) * powers[None, :]).sum(axis=1)
    rng = np.random.RandomState(int(seed))
    groups = []
    for code in sorted(np.unique(codes)):
        group = values[codes == code].copy()
        rng.shuffle(group)
        groups.append(group)
    selected = []
    offsets = np.zeros(len(groups), dtype=np.int64)
    while len(selected) < count:
        progressed = False
        for index, group in enumerate(groups):
            if offsets[index] < len(group):
                selected.append(int(group[offsets[index]]))
                offsets[index] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _indices_sha256(indices):
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def student_b_variant_matrix():
    return {
        "B0": {"teacher": False, "dynamic_threshold": False, "prototype_gate": False, "feature_consistency": False, "distribution_alignment": False, "parameter_heads": False},
        "B1": {"teacher": True, "dynamic_threshold": False, "prototype_gate": False, "feature_consistency": False, "distribution_alignment": False, "parameter_heads": False},
        "B2": {"teacher": True, "dynamic_threshold": False, "prototype_gate": False, "feature_consistency": True, "distribution_alignment": False, "parameter_heads": False},
        "B3": {"teacher": True, "dynamic_threshold": True, "prototype_gate": False, "feature_consistency": True, "distribution_alignment": False, "parameter_heads": False},
        "B4": {"teacher": True, "dynamic_threshold": True, "prototype_gate": True, "feature_consistency": True, "distribution_alignment": False, "parameter_heads": False},
        "B5": {"teacher": True, "dynamic_threshold": True, "prototype_gate": True, "feature_consistency": True, "distribution_alignment": True, "parameter_heads": False},
        "B6": {"teacher": True, "dynamic_threshold": True, "prototype_gate": True, "feature_consistency": True, "distribution_alignment": True, "parameter_heads": True},
    }


def build_student_b_plan(
    config,
    labeled_ratio,
    seed,
    max_labeled_samples=0,
    max_unlabeled_samples=0,
    max_eval_samples=0,
):
    base = build_run_plan(
        config,
        loss_name="cb_asl",
        seed=seed,
        max_train_samples=0,
        max_eval_samples=max_eval_samples,
    )
    labels6 = read_labels(_data_path(config))
    labels = select_labels(labels6, str(config["class_mode"]))
    nested = build_nested_labeled_splits(
        labels,
        base["train_indices"],
        ratios=[0.05, 0.10, 0.20, 0.50, 1.00],
        seed=seed,
    )
    ratio = float(labeled_ratio)
    if ratio not in nested.labeled:
        raise ValueError("labeled_ratio must be one of {}".format(nested.ratios))
    labeled_indices = _limit(nested.labeled[ratio], max_labeled_samples)
    unlabeled_indices = _deterministic_stratified_limit(
        nested.unlabeled[ratio], labels, max_unlabeled_samples, int(seed) + 303
    )
    base.update(
        {
            "labeled_ratio": ratio,
            "labeled_samples": len(labeled_indices),
            "unlabeled_samples": len(unlabeled_indices),
            "semi_split_sha256": nested.sha256,
            "labeled_indices": labeled_indices,
            "unlabeled_indices": unlabeled_indices,
            "labeled_sample_sha256": _indices_sha256(labeled_indices),
            "unlabeled_sample_sha256": _indices_sha256(unlabeled_indices),
        }
    )
    return base


def _parameter_metadata(class_names):
    names = []
    for class_name in class_names:
        names.extend(spec.name for spec in get_parameter_specs(class_name))
    return names


def run_experiment(args):
    config = load_config(args.config)
    variants = student_b_variant_matrix()
    variant = variants[args.variant]
    plan = build_student_b_plan(
        config,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
        max_labeled_samples=args.max_labeled_samples,
        max_unlabeled_samples=args.max_unlabeled_samples,
        max_eval_samples=args.max_eval_samples,
    )
    printable = {
        key: value for key, value in plan.items() if not key.endswith("_indices")
    }
    printable.update({"variant": args.variant, "variant_flags": variant})
    if args.dry_run:
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return printable

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data_path = _data_path(config)
    common = {
        "path": data_path,
        "class_mode": plan["class_mode"],
        "normalization": config["normalization"],
    }
    labeled_dataset = V73SignalDataset(
        indices=plan["labeled_indices"], include_parameters=True, **common
    )
    hidden_base = V73SignalDataset(
        indices=plan["unlabeled_indices"], include_parameters=False, **common
    )
    unlabeled_dataset = HiddenLabelDataset(hidden_base)
    early_dataset = V73SignalDataset(
        indices=plan["early_stop_indices"], include_parameters=False, **common
    )
    calibration_dataset = V73SignalDataset(
        indices=plan["calibration_indices"], include_parameters=False, **common
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    labeled_loader = DataLoader(
        labeled_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed) + 101),
        **loader_args,
    )
    unlabeled_loader = (
        DataLoader(
            unlabeled_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(args.seed) + 202),
            **loader_args,
        )
        if variant["teacher"] and len(unlabeled_dataset)
        else None
    )
    early_loader = DataLoader(early_dataset, shuffle=False, **loader_args)
    calibration_loader = DataLoader(
        calibration_dataset, shuffle=False, **loader_args
    )
    class_names = get_class_names(plan["class_mode"])
    positive_counts = labeled_dataset.labels6[plan["labeled_indices"]][
        :, labeled_dataset.selected_indices
    ].sum(axis=0)
    classification_loss = build_classification_loss(
        args.loss,
        positive_counts=positive_counts,
        sample_count=len(plan["labeled_indices"]),
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.negative_clip,
        beta=args.effective_beta,
    )
    parameter_normalizer = None
    parameter_names = _parameter_metadata(class_names)
    if variant["parameter_heads"]:
        values79, mask79 = read_parameters(data_path)
        target, target_mask = select_parameter_targets(
            values79[plan["labeled_indices"]],
            mask79[plan["labeled_indices"]],
            class_names,
        )
        labels = torch.from_numpy(
            select_labels(
                read_labels(data_path)[plan["labeled_indices"]],
                plan["class_mode"],
            )
        ).float()
        parameter_normalizer = ParameterNormalizer().fit(
            target, torch.logical_and(target_mask, labels[:, :, None] > 0.5)
        )
    backbone = build_backbone(
        args.backbone,
        num_classes=plan["num_classes"],
        width=args.width,
        dropout=args.dropout,
    )
    model = StudentBModel(
        backbone,
        relation_dim=args.relation_dim,
        parameters_per_label=2,
        enable_parameter_heads=variant["parameter_heads"],
        relation_dropout=args.relation_dropout,
    )
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(str(args.initial_checkpoint), map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / "student_b"
        / str(config["experiment_name"])
        / args.variant
        / "ratio_{:.2f}".format(args.labeled_ratio)
        / "seed_{}".format(args.seed)
    )
    output_dir = Path(output_dir)
    config_hash = configuration_sha256(
        {
            "config": config,
            "arguments": stable_run_arguments(vars(args)),
            "plan": printable,
            "variant_flags": variant,
        }
    )
    decision = decide_run(output_dir, config_hash, resume=args.resume)
    if decision.action == "skip":
        print("Skipping completed run: {}".format(output_dir))
        return decision.receipt
    labeled_prior = positive_counts.astype(np.float32) / float(
        max(len(plan["labeled_indices"]), 1)
    )
    tau_pos = args.tau_pos
    tau_neg = args.tau_neg
    if args.thresholds_file is not None:
        with Path(args.thresholds_file).open("r", encoding="utf-8") as handle:
            threshold_payload = json.load(handle)
        tau_pos = threshold_payload["tau_pos"]
        tau_neg = threshold_payload["tau_neg"]
    result = train_student_b(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        validation_loader=early_loader,
        calibration_loader=calibration_loader,
        classification_loss=classification_loss,
        parameter_normalizer=parameter_normalizer,
        parameter_names=parameter_names,
        class_names=class_names,
        labeled_prior=labeled_prior,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=output_dir / "best_model.pth",
        use_ema_teacher=variant["teacher"],
        use_dynamic_threshold=variant["dynamic_threshold"],
        use_prototype_gate=variant["prototype_gate"],
        use_feature_consistency=variant["feature_consistency"],
        use_distribution_alignment=variant["distribution_alignment"],
        use_parameter_heads=variant["parameter_heads"],
        parameter_stop_gradient=args.parameter_stop_gradient,
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        reliability_threshold=args.reliability_threshold,
        stability_ceiling=args.stability_ceiling,
        lambda_u=args.lambda_u,
        lambda_feature=args.lambda_feature,
        lambda_supervised_parameter=args.lambda_supervised_parameter,
        lambda_unsupervised_parameter=args.lambda_unsupervised_parameter,
        lambda_marginal=args.lambda_marginal,
        unsupervised_warmup_fraction=args.unsupervised_warmup_fraction,
        teacher_warmup_epochs=args.teacher_warmup_epochs,
        threshold_update_interval=args.threshold_update_interval,
        balance_pseudo_sides=args.balance_pseudo_sides,
        max_pseudo_side_fraction=args.max_pseudo_side_fraction,
        use_amp=args.use_amp,
        augmentation_seed=args.seed,
        selection_metric=args.selection_metric,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_history(output_dir / "history.csv", result.history)
    _write_json(output_dir / "pseudo_label_statistics.json", result.pseudo_label_statistics)
    _write_json(output_dir / "thresholds.json", result.final_thresholds)
    _write_json(output_dir / "prototype_diagnostics.json", result.prototype_diagnostics)
    receipt = {
        "status": "completed",
        "config_sha256": config_hash,
        "variant": args.variant,
        "labeled_ratio": args.labeled_ratio,
        "labeled_samples": plan["labeled_samples"],
        "unlabeled_samples": plan["unlabeled_samples"],
        "best_epoch": result.best_epoch,
        "epochs_ran": result.epochs_ran,
        "best_subset_acc": result.best_subset_acc,
        "best_hamming_loss": result.best_hamming_loss,
        "selection_metric": result.selection_metric,
        "best_selection_value": result.best_selection_value,
        "train_seconds": result.train_seconds,
        "training_parameters": count_trainable_parameters(model),
        "deployment_parameters": count_student_b_deployment_parameters(model),
        "device": str(device),
        "split_sha256": plan["split_sha256"],
        "semi_split_sha256": plan["semi_split_sha256"],
        "labeled_sample_sha256": plan["labeled_sample_sha256"],
        "unlabeled_sample_sha256": plan["unlabeled_sample_sha256"],
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    labeled_dataset.close()
    unlabeled_dataset.close()
    early_dataset.close()
    calibration_dataset.close()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_b" / "daps_param.yaml")
    parser.add_argument("--variant", choices=list(student_b_variant_matrix()), default="B6")
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--loss", choices=["bce", "focal", "asl", "cb_bce", "cb_asl", "db"], default="cb_asl")
    parser.add_argument("--backbone", default="complex_1d_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--relation-dropout", type=float, default=0.1)
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=3.0)
    parser.add_argument("--negative-clip", type=float, default=0.05)
    parser.add_argument("--effective-beta", type=float, default=0.99)
    parser.add_argument("--tau-pos", type=float, default=0.95)
    parser.add_argument("--tau-neg", type=float, default=0.05)
    parser.add_argument("--reliability-threshold", type=float, default=0.60)
    parser.add_argument("--stability-ceiling", type=float, default=0.10)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--lambda-feature", type=float, default=0.10)
    parser.add_argument("--lambda-supervised-parameter", type=float, default=0.10)
    parser.add_argument("--lambda-unsupervised-parameter", type=float, default=0.20)
    parser.add_argument("--lambda-marginal", type=float, default=0.02)
    parser.add_argument("--unsupervised-warmup-fraction", type=float, default=0.20)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=0)
    parser.add_argument("--threshold-update-interval", type=int, default=5)
    parser.add_argument(
        "--max-pseudo-side-fraction",
        type=float,
        default=1.0,
        help="Per minibatch and per label cap applied independently to positive and negative pseudo labels.",
    )
    parser.add_argument(
        "--balance-pseudo-sides",
        dest="balance_pseudo_sides",
        action="store_true",
    )
    parser.add_argument(
        "--no-balance-pseudo-sides",
        dest="balance_pseudo_sides",
        action="store_false",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["subset_acc", "macro_f1", "macro_aupr"],
        default="subset_acc",
    )
    parser.add_argument("--max-labeled-samples", type=int, default=0)
    parser.add_argument("--max-unlabeled-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--parameter-stop-gradient", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--thresholds-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True, balance_pseudo_sides=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
