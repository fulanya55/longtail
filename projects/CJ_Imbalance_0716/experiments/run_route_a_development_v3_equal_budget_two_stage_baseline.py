"""Equal-total-budget two-stage Focal24 -> frozen Linear-BCE12 baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import _extract, _predict_head, _train_head  # noqa: E402
from experiments.run_route_a_development_v3_mainstream_loss_baselines import (  # noqa: E402
    EXPECTED_V3_SHA256, _load_rawsaw, _reference_result, _route_rawsaw, _train_loss_model, _value,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _evaluate, _load_roles, _raw_scores, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


REFERENCE = "R0_Focal36_RawSaw"
STAGE1 = "T1_Focal24_RawSaw"
TWOSTAGE = "T2_Focal24_FrozenLinearBCE12_RawSaw"
METRICS = ("macro_aupr", "macro_f1", "target_weak_recall", "target_weak_auroc", "target_fpr", "isrj_aupr", "isrj_weak_recall", "isrj_weak_auroc")


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _two_stage_args(args, epochs: int):
    derived = copy.copy(args)
    derived.epochs = int(epochs)
    derived.smoke_epochs = int(epochs)
    # Full compute budget is fixed.  This makes `patience` inert without
    # changing the shared implementation's validation/checkpoint receipt.
    derived.patience = int(epochs)
    return derived


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    model, f2_checkpoint = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    model.load_state_dict(initial_state)

    stage1_args = _two_stage_args(args, args.representation_epochs)
    # Reuse the established Focal loss definition; the public experiment name
    # remains T1 so it cannot be mistaken for the 36-epoch single-stage run.
    model, stage1_training = _train_loss_model(
        model, "L2_Focal_RawSaw", roles, weak6, seed, device, stage1_args, root / STAGE1
    )
    features = {
        role: _extract(model, values["signals"], device, args.feature_batch_size)
        for role, values in roles.items()
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mean = features["fit"].mean(axis=0, keepdims=True)
    std = np.maximum(features["fit"].std(axis=0, keepdims=True), 1e-5)
    features = {role: ((value - mean) / std).astype(np.float32) for role, value in features.items()}

    head_args = _two_stage_args(args, args.head_epochs)
    head, head_training = _train_head(
        features, roles, weak6, "linear", "bce", seed + 91000, device, head_args, root / TWOSTAGE
    )
    two_stage_logits = {
        role: _predict_head(head, values, device, args.eval_batch_size)
        for role, values in features.items()
    }
    del head
    raw, raw_path = _load_rawsaw(args, seed, device)
    raw_scores = {
        role: _raw_scores(raw, values["signals"], device, args.raw_batch_size)
        for role, values in roles.items()
    }
    del raw
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage-1 is evaluated with its preserved end-to-end classifier before the
    # representation is discarded; load the saved checkpoint only for this
    # read-only comparison.
    stage1_payload = torch.load(root / STAGE1 / "best_model.pth", map_location="cpu", weights_only=False)
    stage1_model, _ = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    stage1_model.load_state_dict(stage1_payload["model_state_dict"])
    from experiments.run_route_a_development_v2_shared_gradient import _predict  # local import avoids a broad module dependency at import time
    stage1_logits = {
        role: _predict(stage1_model, values["signals"], device, args.eval_batch_size)
        for role, values in roles.items()
    }
    del stage1_model
    results = [
        _reference_result(args.convergence_root, seed, "E2_Focal_36", REFERENCE),
        _evaluate(STAGE1, _route_rawsaw(stage1_logits, raw_scores), roles, weak6, root / STAGE1, args),
        _evaluate(TWOSTAGE, _route_rawsaw(two_stage_logits, raw_scores), roles, weak6, root / TWOSTAGE, args),
    ]
    row = {
        "status": "completed",
        "seed": int(seed),
        "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_path),
        "f2_checkpoint_sha256": _sha256(f2_checkpoint),
        "rawsaw_checkpoint_sha256": _sha256(raw_path),
        "stage1_checkpoint_sha256": _sha256(root / STAGE1 / "best_model.pth"),
        "stage2_checkpoint_sha256": _sha256(root / TWOSTAGE / "best_model.pth"),
        "feature_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "feature_std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
        "training": {
            "representation": stage1_training,
            "head": head_training,
            "representation_epochs_budget": int(args.representation_epochs),
            "head_epochs_budget": int(args.head_epochs),
            "total_epochs_budget": int(args.representation_epochs + args.head_epochs),
            "total_epochs_executed": int(stage1_training["epochs_ran"] + args.head_epochs),
        },
        "variants": results,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    _write_json(receipt, row)
    return row


def run(args):
    if args.representation_epochs + args.head_epochs != args.total_epochs:
        raise ValueError("representation_epochs + head_epochs must equal total_epochs")
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if seal["access_authorized"] or seal["analysis_allowed"] or seal["used"]:
        raise ValueError("test-v3 is not sealed")
    labels = read_labels(data_path)
    values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    if args.smoke:
        args = copy.copy(args)
        args.representation_epochs = args.smoke_representation_epochs
        args.head_epochs = args.smoke_head_epochs
        args.total_epochs = args.representation_epochs + args.head_epochs
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {
        variant: {metric: _mean_ci([_value(_variant_map(row)[variant], metric) for row in rows], 20331000 + vi * 100 + mi)
                  for mi, metric in enumerate(METRICS)}
        for vi, variant in enumerate((REFERENCE, STAGE1, TWOSTAGE))
    }
    deltas = {
        metric: _mean_ci([_value(_variant_map(row)[TWOSTAGE], metric) - _value(_variant_map(row)[REFERENCE], metric) for row in rows], 20332000 + mi)
        for mi, metric in enumerate(METRICS)
    }
    checks = {
        "exact_total_budget": all(row["training"]["total_epochs_executed"] == args.total_epochs for row in rows),
        "all_stage1_full_budget": all(row["training"]["representation"]["epochs_ran"] == args.representation_epochs for row in rows),
        "test_v3_unused": all(not row["test_v3_used"] for row in rows),
    }
    payload = {
        "status": "completed",
        "experiment": "RouteA_DevelopmentV3_EqualBudgetTwoStageBaseline",
        "smoke": bool(args.smoke),
        "seeds": list(seeds),
        "data_sha256": data_sha,
        "protocol": {"representation_loss": "focal", "representation_epochs": args.representation_epochs, "head_loss": "bce", "head_epochs": args.head_epochs, "total_epochs": args.total_epochs},
        "aggregate": aggregate,
        "two_stage_minus_focal36": deltas,
        "execution_gate": {"passed": bool(all(checks.values())), "checks": checks},
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_development_v3_equal_budget_two_stage_baseline_summary.json"
    _write_json(output, payload)
    print(json.dumps({"summary": str(output), "execution_gate": payload["execution_gate"], "test_v3_used": False}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--convergence-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_loss_convergence_audit")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_equal_budget_two_stage_baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--representation-epochs", type=int, default=24)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--total-epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=128); parser.add_argument("--eval-batch-size", type=int, default=128); parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4); parser.add_argument("--representation-lr-scale", type=float, default=.1)
    parser.add_argument("--minimum-lr-ratio", type=float, default=.01); parser.add_argument("--saw-loss-weight", type=float, default=.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument("--max-negative-fpr", type=float, default=.20); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--raw-hidden-channels", type=int, default=16); parser.add_argument("--raw-topk", type=int, default=16); parser.add_argument("--raw-dropout", type=float, default=.1)
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-representation-epochs", type=int, default=2); parser.add_argument("--smoke-head-epochs", type=int, default=2)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
