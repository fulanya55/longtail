"""Build per-label pseudo-label thresholds from a pretrained B0 checkpoint."""

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

from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from cjlt.data.augmentations import weak_complex_augmentation  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.student_b import StudentBModel  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402
from experiments.run_student_b import build_student_b_plan  # noqa: E402


@torch.no_grad()
def build_thresholds(args):
    config = load_config(args.config)
    plan = build_student_b_plan(
        config,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
        max_labeled_samples=0,
        max_unlabeled_samples=0,
        max_eval_samples=args.max_eval_samples,
    )
    dataset = V73SignalDataset(
        path=_data_path(config),
        indices=plan["calibration_indices"],
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
        enable_parameter_heads=False,
        relation_dropout=args.relation_dropout,
    )
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    probabilities = []
    targets = []
    generator = torch.Generator().manual_seed(int(args.augmentation_seed))
    for batch in loader:
        signal = batch["signal"]
        if args.teacher_view == "weak":
            signal = weak_complex_augmentation(
                signal, [], generator=generator
            ).signal
        logits = model(signal.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["label"].numpy())
    probability = np.concatenate(probabilities, axis=0)
    target = np.concatenate(targets, axis=0).astype(np.uint8)
    tau_pos = np.quantile(probability, float(args.positive_quantile), axis=0)
    tau_neg = np.quantile(probability, float(args.negative_quantile), axis=0)
    positive = probability >= tau_pos[None, :]
    negative = probability <= tau_neg[None, :]
    accepted = np.logical_or(positive, negative)
    positive_precision = np.divide(
        np.logical_and(positive, target > 0).sum(axis=0),
        np.maximum(positive.sum(axis=0), 1),
    )
    negative_precision = np.divide(
        np.logical_and(negative, target == 0).sum(axis=0),
        np.maximum(negative.sum(axis=0), 1),
    )
    payload = {
        "tau_pos": tau_pos.tolist(),
        "tau_neg": tau_neg.tolist(),
        "positive_quantile": float(args.positive_quantile),
        "negative_quantile": float(args.negative_quantile),
        "candidate_coverage_per_label": accepted.mean(axis=0).tolist(),
        "candidate_coverage_total": float(accepted.mean()),
        "positive_precision_per_label": positive_precision.tolist(),
        "negative_precision_per_label": negative_precision.tolist(),
        "calibration_samples": int(len(probability)),
        "source_role": "calibration_{}_view".format(args.teacher_view),
        "teacher_view": args.teacher_view,
        "augmentation_seed": int(args.augmentation_seed),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "locked_test_used": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    dataset.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_b" / "daps_no_param.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labeled-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--positive-quantile", type=float, default=0.90)
    parser.add_argument("--negative-quantile", type=float, default=0.10)
    parser.add_argument("--teacher-view", choices=["clean", "weak"], default="weak")
    parser.add_argument("--augmentation-seed", type=int, default=2026)
    parser.add_argument("--backbone", default="complex_1d_true")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--relation-dropout", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    build_thresholds(parse_args())
