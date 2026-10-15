"""Post-hoc pseudo-label audit on hidden unlabeled truth; never used for training."""

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

from cjlt.data.augmentations import weak_complex_augmentation  # noqa: E402
from cjlt.data.registry import get_class_names, select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from cjlt.losses.pseudo_label import build_pseudo_labels  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.student_b import StudentBModel  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402
from experiments.run_student_b import build_student_b_plan  # noqa: E402


def _load_model(args, num_classes):
    backbone = build_backbone(
        args.backbone,
        num_classes=num_classes,
        width=args.width,
        dropout=args.dropout,
    )
    model = StudentBModel(
        backbone,
        relation_dim=args.relation_dim,
        parameters_per_label=2,
        enable_parameter_heads=False,
        relation_dropout=args.relation_dropout,
    )
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    if args.state == "teacher":
        raw = checkpoint["teacher_state_dict"]
        state = {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in raw.items()
        }
    else:
        state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def audit(args):
    config = load_config(args.config)
    plan = build_student_b_plan(
        config,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
        max_labeled_samples=0,
        max_unlabeled_samples=args.max_unlabeled_samples,
        max_eval_samples=args.max_eval_samples,
    )
    dataset = V73SignalDataset(
        path=_data_path(config),
        indices=plan["unlabeled_indices"],
        class_mode=plan["class_mode"],
        normalization=config["normalization"],
        include_parameters=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    with Path(args.thresholds_file).open("r", encoding="utf-8") as handle:
        thresholds = json.load(handle)
    model = _load_model(args, plan["num_classes"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    generator = torch.Generator().manual_seed(int(args.augmentation_seed))
    positive_count = np.zeros(plan["num_classes"], dtype=np.int64)
    negative_count = np.zeros(plan["num_classes"], dtype=np.int64)
    true_positive = np.zeros(plan["num_classes"], dtype=np.int64)
    true_negative = np.zeros(plan["num_classes"], dtype=np.int64)
    sample_count = 0
    for batch in loader:
        raw = batch["signal"]
        weak_one = weak_complex_augmentation(raw, [], generator=generator).signal
        weak_two = weak_complex_augmentation(raw, [], generator=generator).signal
        first = torch.sigmoid(model(weak_one.to(device)))
        second = torch.sigmoid(model(weak_two.to(device)))
        pseudo = build_pseudo_labels(
            first,
            thresholds["tau_pos"],
            thresholds["tau_neg"],
            stability=torch.abs(first - second),
            stability_ceiling=args.stability_ceiling,
            max_positive_fraction=args.max_pseudo_side_fraction,
            max_negative_fraction=args.max_pseudo_side_fraction,
        )
        truth = batch["label"].to(device).bool()
        positive_count += pseudo.positive_mask.sum(dim=0).cpu().numpy()
        negative_count += pseudo.negative_mask.sum(dim=0).cpu().numpy()
        true_positive += torch.logical_and(pseudo.positive_mask, truth).sum(dim=0).cpu().numpy()
        true_negative += torch.logical_and(pseudo.negative_mask, ~truth).sum(dim=0).cpu().numpy()
        sample_count += len(truth)
    class_names = get_class_names(plan["class_mode"])
    labels = select_labels(read_labels(_data_path(config)), plan["class_mode"])
    unlabeled_truth = labels[plan["unlabeled_indices"]]
    positive_precision = np.divide(true_positive, np.maximum(positive_count, 1))
    negative_precision = np.divide(true_negative, np.maximum(negative_count, 1))
    payload = {
        "status": "completed",
        "audit_only_truth_used": True,
        "truth_influenced_training": False,
        "locked_test_used": False,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_state": args.state,
        "unlabeled_samples": int(sample_count),
        "unlabeled_combination_count": int(len(np.unique(unlabeled_truth, axis=0))),
        "unlabeled_positive_prior": unlabeled_truth.mean(axis=0).tolist(),
        "class_names": list(class_names),
        "positive_count": positive_count.tolist(),
        "negative_count": negative_count.tolist(),
        "positive_precision": positive_precision.tolist(),
        "negative_precision": negative_precision.tolist(),
        "positive_coverage": (positive_count / max(sample_count, 1)).tolist(),
        "negative_coverage": (negative_count / max(sample_count, 1)).tolist(),
        "accepted_coverage_total": float(
            (positive_count.sum() + negative_count.sum())
            / max(sample_count * plan["num_classes"], 1)
        ),
        "stability_ceiling": float(args.stability_ceiling),
        "max_pseudo_side_fraction": float(args.max_pseudo_side_fraction),
        "augmentation_seed": int(args.augmentation_seed),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    dataset.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_b" / "daps_no_param.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=["model", "teacher"], default="model")
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-unlabeled-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stability-ceiling", type=float, default=0.10)
    parser.add_argument("--max-pseudo-side-fraction", type=float, default=0.30)
    parser.add_argument("--augmentation-seed", type=int, default=2026)
    parser.add_argument("--backbone", default="complex_1d_true")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--relation-dropout", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
