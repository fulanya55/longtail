"""Repeat the seed-2043 S2 expert with fixed data and changed initialization only."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files, load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_label_local_saw import F2_MODEL_NAME, _load_h2_head  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import (  # noqa: E402
    _binary_metrics,
    _development_pairs,
    _extract_seed_arrays,
    _indices_for_identities,
    _level_lookup,
    _model_scores,
    _parameter_counts,
    _stratified_triplets,
    _train_variant,
)
from experiments.run_route_a_sawfm_expert_routing import _extract_power_identities, _load_expert  # noqa: E402
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


VARIANT = "S2_TeacherHardNeg"
SAW_INDEX = 1


def _sample_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base identity labels are misaligned")
    seed = int(args.data_seed)
    plan = build_run_plan(config, "db", seed, 0, args.max_eval_samples)
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    receipt = _read(source_dir / "run_receipt.json")
    if receipt.get("locked_test_used") or receipt["split_sha256"] != plan["split_sha256"]:
        raise ValueError("F2 source invalid or split drifted")
    selection = _stratified_triplets(
        plan["train_indices"], labels, identities, args.pairs_per_combo, 161000 + seed
    )
    development = _development_pairs(
        plan["early_stop_indices"], labels, identities, 162000 + seed
    )
    train_ids = set(
        selection["positive_identities"].tolist()
        + selection["hard_negative_identities"].tolist()
        + selection["random_negative_identities"].tolist()
    )
    development_ids = set(
        development["positive_identities"].tolist()
        + development["hard_negative_identities"].tolist()
    )
    overlap = train_ids.intersection(development_ids)
    if overlap:
        raise ValueError("Training and development identities overlap")
    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    extraction_levels = sorted(set([10, 40, 45, 50] + list(args.eval_levels)))
    level_info = _level_lookup(power_files, extraction_levels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, f2_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    base_head, h2_checkpoint, h2_path, _ = _load_h2_head(seed, args.frozen_head_root, device)
    print("extracting fixed seed-2043 training and development features", flush=True)
    arrays = _extract_seed_arrays(
        args,
        config,
        power_files,
        level_info,
        backbone,
        base_head,
        h2_checkpoint,
        selection,
        development,
        device,
    )
    development_features = {}
    for level in args.eval_levels:
        development_features[str(level)] = {
            "positive": _extract_power_identities(
                args, config, backbone, base_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)],
                development["positive_identities"], device
            )["features"],
            "negative": _extract_power_identities(
                args, config, backbone, base_head, h2_checkpoint,
                power_files[int(level)], level_info[int(level)],
                development["hard_negative_identities"], device
            )["features"],
        }
    original_expert_path = (
        args.expert_root
        / "seed_{}".format(seed)
        / "S2_TeacherHardNeg"
        / "best_saw_stage2_evidence.pth"
    )
    original_expert, _ = _load_expert(original_expert_path, device)
    original_power = {}
    for level in args.eval_levels:
        groups = development_features[str(level)]
        positive = _model_scores(original_expert, groups["positive"], device)
        negative = _model_scores(original_expert, groups["negative"], device)
        original_power[str(level)] = _binary_metrics(positive, negative)
    original_cross_power_mean = float(
        np.mean([original_power[str(level)]["auroc"] for level in args.eval_levels])
    )
    order = np.random.default_rng(163000 + seed).permutation(len(selection["positive_identities"]))
    split = int(round(0.8 * len(order)))
    fit_indices, validation_indices = order[:split], order[split:]
    positive_labels = labels[development["positive_base_indices"]]
    six_component = positive_labels.sum(axis=1) == 6
    rows = []
    for initialization_seed in args.initialization_seeds:
        print("training initialization_seed={}".format(initialization_seed), flush=True)
        model, best_epoch, history, train_seed = _train_variant(
            VARIANT,
            arrays,
            fit_indices,
            validation_indices,
            seed,
            device,
            args,
            train_seed_override=int(initialization_seed),
        )
        output_dir = args.output_root / "init_{}".format(initialization_seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best_saw_stage2_evidence.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "variant": VARIANT,
                "data_seed": seed,
                "train_seed": int(train_seed),
                "best_epoch": int(best_epoch),
                "selection_sha256": selection["selection_sha256"],
                "split_sha256": plan["split_sha256"],
            },
            checkpoint_path,
        )
        _write_history(output_dir / "history.csv", history)
        power_metrics = {}
        for level in args.eval_levels:
            groups = development_features[str(level)]
            positive = _model_scores(model, groups["positive"], device)
            negative = _model_scores(model, groups["negative"], device)
            overall = _binary_metrics(positive, negative)
            six = (
                _binary_metrics(positive[six_component], negative[six_component])
                if int(six_component.sum()) >= 2
                else None
            )
            power_metrics[str(level)] = {"overall": overall, "six_component": six}
            np.savez_compressed(
                str(output_dir / "power_{}db_predictions.npz".format(level)),
                positive_scores=positive,
                negative_scores=negative,
                six_component_mask=six_component,
                positive_identities=development["positive_identities"],
                negative_identities=development["hard_negative_identities"],
            )
        overall_aurocs = np.asarray(
            [power_metrics[str(level)]["overall"]["auroc"] for level in args.eval_levels]
        )
        six_aurocs = np.asarray(
            [power_metrics[str(level)]["six_component"]["auroc"] for level in args.eval_levels]
        )
        row = {
            "initialization_seed": int(initialization_seed),
            "data_seed": seed,
            "best_epoch": int(best_epoch),
            "power": power_metrics,
            "cross_power_auroc_mean": float(overall_aurocs.mean()),
            "cross_power_auroc_std": float(overall_aurocs.std(ddof=1)),
            "cross_power_auroc_worst": float(overall_aurocs.min()),
            "six_component_auroc_mean": float(six_aurocs.mean()),
            "six_component_pair_count": int(six_component.sum()),
            "parameter_counts": _parameter_counts(model),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "f2_checkpoint_sha256": _sha256(f2_path),
            "h2_checkpoint_sha256": _sha256(h2_path),
            "selection_sha256": selection["selection_sha256"],
            "development_selection_sha256": development["selection_sha256"],
            "split_sha256": plan["split_sha256"],
            "identity_overlap": 0,
            "locked_test_used": False,
        }
        _write(output_dir / "metrics.json", row)
        _write(output_dir / "run_receipt.json", {
            "status": "completed", "initialization_seed": int(initialization_seed),
            "data_seed": seed, "checkpoint_sha256": row["checkpoint_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "identity_overlap": 0, "locked_test_used": False,
        })
        rows.append(row)
    baseline_mean = original_cross_power_mean
    for row in rows:
        row["gain_vs_original_2043_s2"] = row["cross_power_auroc_mean"] - baseline_mean
    aggregate = {
        metric: _mean_ci([row[metric] for row in rows], 261000 + index)
        for index, metric in enumerate(("cross_power_auroc_mean", "cross_power_auroc_std", "cross_power_auroc_worst", "six_component_auroc_mean", "gain_vs_original_2043_s2"))
    }
    checks = {
        "two_mean_auroc_0p65": sum(row["cross_power_auroc_mean"] >= 0.65 for row in rows) >= 2,
        "two_worst_auroc_0p58": sum(row["cross_power_auroc_worst"] >= 0.58 for row in rows) >= 2,
        "initialization_mean_std_0p10": _sample_std(
            [row["cross_power_auroc_mean"] for row in rows]
        ) <= 0.10,
        "two_six_component_0p55": sum(row["six_component_auroc_mean"] >= 0.55 for row in rows) >= 2,
        "two_gain_vs_original_0p10": sum(row["gain_vs_original_2043_s2"] >= 0.10 for row in rows) >= 2,
        "identity_and_locked_test_audit": len(overlap) == 0 and all(not row["locked_test_used"] for row in rows),
    }
    gate = {"checks": checks, "passed": all(checks.values())}
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {"data_seed": seed, "initialization_seeds": list(args.initialization_seeds), "eval_levels": list(args.eval_levels), "pairs_per_combo": args.pairs_per_combo, "epochs": args.epochs, "patience": args.patience},
        "sampling": {"fit_pairs": int(len(fit_indices)), "validation_pairs": int(len(validation_indices)), "development_pairs": int(len(development["positive_identities"])), "six_component_pairs": int(six_component.sum()), "identity_overlap": 0, "selection_sha256": selection["selection_sha256"], "development_selection_sha256": development["selection_sha256"], "split_sha256": plan["split_sha256"]},
        "original_2043_s2": {
            "cross_power_auroc_mean": baseline_mean,
            "power": original_power,
            "checkpoint_sha256": _sha256(original_expert_path),
        },
        "per_run": rows, "aggregate": aggregate,
        "initialization_mean_auroc_std": _sample_std(
            [row["cross_power_auroc_mean"] for row in rows]
        ),
        "promotion_gate": gate,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_sawfm_s2_initialization_stability_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--cross-power-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_fixed_fusion_cross_power")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_s2_initialization_stability")
    parser.add_argument("--data-seed", type=int, default=2043)
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--eval-levels", type=int, nargs="+", default=[35, 40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=40)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--adapter-limit", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    _run(parse_args())
