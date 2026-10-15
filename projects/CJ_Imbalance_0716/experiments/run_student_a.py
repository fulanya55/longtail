"""Student A entrypoint: class-balanced prototypes and paired power views."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pair_dataset import PowerPairDataset  # noqa: E402
from cjlt.data.power_pairs import build_role_bound_power_pair_index  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from cjlt.engine.student_a_trainer import train_student_a  # noqa: E402
from cjlt.engine.run_manager import (  # noqa: E402
    configuration_sha256,
    decide_run,
    stable_run_arguments,
)
from cjlt.losses.classification import (  # noqa: E402
    build_classification_loss,
    effective_number_weights,
)
from cjlt.models.backbone_adapter import build_backbone, count_trainable_parameters  # noqa: E402
from cjlt.models.student_a import (  # noqa: E402
    StudentAModel,
    count_deployment_parameters,
)
from experiments.run_shared import (  # noqa: E402
    _data_path,
    _write_history,
    _write_json,
    build_run_plan,
    load_config,
)


def student_a_variant_matrix():
    return {
        "A0": {"prototype": False, "power": False},
        "A2": {"prototype": True, "power": False},
        "A3": {"prototype": False, "power": True},
        "A4": {"prototype": True, "power": True},
    }


def run_experiment(args):
    config = load_config(args.config)
    variants = student_a_variant_matrix()
    if args.variant not in variants:
        raise ValueError("Unknown Student A variant: {}".format(args.variant))
    variant = variants[args.variant]
    plan = build_run_plan(
        config,
        loss_name=args.loss,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    printable = {
        key: value for key, value in plan.items() if not key.endswith("_indices")
    }
    printable["variant"] = args.variant
    printable.update(variant)
    if args.dry_run:
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return printable

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    data_path = _data_path(config)
    dataset_common = {
        "path": data_path,
        "class_mode": plan["class_mode"],
        "normalization": config["normalization"],
    }
    validation_dataset = V73SignalDataset(
        indices=plan["early_stop_indices"], **dataset_common
    )
    pair_dataset = None
    pair_index = None
    if variant["power"]:
        power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
        pair_index = build_role_bound_power_pair_index(
            data_path,
            power_root,
            base_indices=plan["train_indices"],
            levels=args.power_levels,
            reference_level=args.reference_level,
        )
        pair_dataset = PowerPairDataset(
            power_root,
            pair_index.records,
            class_mode=plan["class_mode"],
            normalization=config["normalization"],
        )
        train_dataset = pair_dataset
    else:
        train_dataset = V73SignalDataset(
            indices=plan["train_indices"], **dataset_common
        )
    loader_common = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.workers),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_common)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_common
    )
    selected = validation_dataset.selected_indices
    positive_counts = validation_dataset.labels6[plan["train_indices"]][
        :, selected
    ].sum(axis=0)
    classification_loss = build_classification_loss(
        args.loss,
        positive_counts=positive_counts,
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.negative_clip,
        beta=args.effective_beta,
    )
    relation_weights = effective_number_weights(
        positive_counts, beta=args.effective_beta
    )
    backbone = build_backbone(
        args.backbone,
        num_classes=plan["num_classes"],
        width=args.width,
        dropout=args.dropout,
    )
    model = StudentAModel(
        backbone,
        relation_dim=args.relation_dim,
        relation_dropout=args.relation_dropout,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / "student_a"
        / str(plan["experiment_name"])
        / args.variant
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
    result = train_student_a(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        classification_loss=classification_loss,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=output_dir / "best_model.pth",
        use_prototype=variant["prototype"],
        use_power_consistency=variant["power"],
        prototype_weight=args.prototype_weight,
        power_weight=args.power_weight,
        prototype_temperature=args.prototype_temperature,
        prototype_momentum=args.prototype_momentum,
        label_weights=relation_weights,
        warmup_fraction=args.warmup_fraction,
        ramp_end_fraction=args.ramp_end_fraction,
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_history(output_dir / "history.csv", result.history)
    _write_json(output_dir / "prototype_diagnostics.json", result.prototype_diagnostics)
    receipt = {
        "status": "completed",
        "config_sha256": config_hash,
        "variant": args.variant,
        "use_prototype": variant["prototype"],
        "use_power_consistency": variant["power"],
        "best_epoch": result.best_epoch,
        "epochs_ran": result.epochs_ran,
        "best_subset_acc": result.best_subset_acc,
        "best_hamming_loss": result.best_hamming_loss,
        "train_seconds": result.train_seconds,
        "training_parameters": count_trainable_parameters(model),
        "deployment_parameters": count_deployment_parameters(model),
        "device": str(device),
        "split_sha256": plan["split_sha256"],
        "pair_sha256": None if pair_index is None else pair_index.sha256,
        "pair_records": 0 if pair_index is None else len(pair_index.records),
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    validation_dataset.close()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "shared" / "four_class.yaml",
    )
    parser.add_argument("--variant", choices=list(student_a_variant_matrix()), default="A4")
    parser.add_argument("--loss", choices=["bce", "focal", "asl", "cb_bce", "cb_asl"], default="cb_asl")
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
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--relation-dropout", type=float, default=0.1)
    parser.add_argument("--prototype-weight", type=float, default=0.2)
    parser.add_argument("--power-weight", type=float, default=0.2)
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument("--prototype-momentum", type=float, default=0.99)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--ramp-end-fraction", type=float, default=0.3)
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=3.0)
    parser.add_argument("--negative-clip", type=float, default=0.05)
    parser.add_argument("--effective-beta", type=float, default=0.99)
    parser.add_argument("--power-levels", type=int, nargs="+", default=[10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--reference-level", type=int, default=30)
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
