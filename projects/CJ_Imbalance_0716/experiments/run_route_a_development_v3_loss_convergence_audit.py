"""Audit whether the strongest development-v3 loss baselines were limited by 12 epochs."""

from __future__ import annotations

import argparse
import copy
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
from experiments.run_route_a_development_v2_shared_gradient import _component_power, _predict  # noqa: E402
from experiments.run_route_a_development_v3_mainstream_loss_baselines import (  # noqa: E402
    EXPECTED_V3_SHA256, METRICS, _comparison, _load_model_from_state, _load_rawsaw,
    _reference_result, _route_rawsaw, _train_loss_model, _value,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    _evaluate, _load_roles, _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


LONG_VARIANTS = {
    "E2_Focal_36": "L2_Focal_RawSaw",
    "E3_ASL_36": "L3_ASL_RawSaw",
    "E4_DB_36": "L4_DB_RawSaw",
}
ORIGINAL_VARIANTS = {
    "O2_Focal_12": "L2_Focal_RawSaw",
    "O3_ASL_12": "L3_ASL_RawSaw",
    "O4_DB_12": "L4_DB_RawSaw",
}
F3_NAME = "R1_F3_Reference"
F2_MODEL_NAME = "F2_DFMS_Full"


def _load_existing_loss(root, seed, source_name, output_name):
    source = root / f"seed_{seed}" / source_name
    result = {
        "variant": output_name,
        "metrics": json.loads((source / "metrics.json").read_text(encoding="utf-8")),
        "calibration": json.loads((source / "calibration.json").read_text(encoding="utf-8")),
        "thresholds": json.loads((source / "thresholds.json").read_text(encoding="utf-8")),
        "predictions_sha256": _sha256(source / "predictions.npz"),
    }
    return result


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config),
                                   config["normalization"], args.io_batch_size)
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    template, f2_checkpoint = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial_state = copy.deepcopy({key: value.detach().cpu()
                                   for key, value in template.state_dict().items()})
    del template
    raw, raw_checkpoint = _load_rawsaw(args, seed, device)
    from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _raw_scores
    raw_scores = {role: _raw_scores(raw, values["signals"], device, args.raw_batch_size)
                  for role, values in roles.items()}
    del raw

    results = []
    for output_name, source_name in ORIGINAL_VARIANTS.items():
        results.append(_load_existing_loss(args.baseline_root, seed, source_name, output_name))
    results.append(_reference_result(args.f3_root, seed, "F3_AlignedPeriodicityFusionHead", F3_NAME))

    training = {}
    hashes = {}
    for output_name, train_name in LONG_VARIANTS.items():
        model = _load_model_from_state(f2_dir, device, initial_state)
        model, training[output_name] = _train_loss_model(
            model, train_name, roles, weak6, seed, device, args, root / output_name
        )
        checkpoint = root / output_name / "best_model.pth"
        hashes[output_name] = _sha256(checkpoint)
        logits = {role: _predict(model, values["signals"], device, args.eval_batch_size)
                  for role, values in roles.items()}
        result = _evaluate(output_name, _route_rawsaw(logits, raw_scores), roles, weak6,
                           root / output_name, args)
        results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "seed": int(seed), "status": "completed", "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_path),
        "f2_checkpoint_sha256": _sha256(f2_checkpoint),
        "rawsaw_checkpoint_sha256": _sha256(raw_checkpoint),
        "training": training, "model_sha256": hashes, "variants": results,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _aggregate(rows, names):
    return {name: {metric: _mean_ci([
        _value(_variant_map(row)[name], metric) for row in rows
    ], 20320000 + ni * 100 + mi) for mi, metric in enumerate(METRICS)}
        for ni, name in enumerate(names)}


def _paired_epoch_delta(rows, extended, original, offset):
    pseudo_rows = []
    for row in rows:
        variants = _variant_map(row)
        pseudo_rows.append({"seed": row["seed"], "variants": [variants[extended], variants[original]]})
    return _comparison(pseudo_rows, extended, original, offset)


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")

    labels6 = read_labels(data_path)
    values79, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values79), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_DevelopmentV3_LossConvergenceAudit",
        "seeds": args.seeds, "long_variants": LONG_VARIANTS,
        "epochs": args.epochs, "patience": args.patience,
        "learning_rate": args.learning_rate,
        "representation_lr_scale": args.representation_lr_scale,
        "same_first_12_epoch_batch_order": True,
        "locked_test_used": False, "test_v3_used": False,
    })
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in args.seeds]
    names = tuple(ORIGINAL_VARIANTS) + (F3_NAME,) + tuple(LONG_VARIANTS)
    aggregate = _aggregate(rows, names)
    strongest = max(LONG_VARIANTS, key=lambda name: (
        aggregate[name]["macro_aupr"]["mean"], aggregate[name]["target_weak_auroc"]["mean"]))
    f3_comparison = _comparison(rows, F3_NAME, strongest, 20321000)
    original_for_extended = {extended: original for extended, original in zip(LONG_VARIANTS, ORIGINAL_VARIANTS)}
    budget_deltas = {extended: _paired_epoch_delta(
        rows, extended, original_for_extended[extended], 20322000 + index * 100
    ) for index, extended in enumerate(LONG_VARIANTS)}
    best_epochs = {name: [int(row["training"][name]["best_epoch"]) for row in rows]
                   for name in LONG_VARIANTS}
    cap_hits = {name: int(sum(epoch == args.epochs for epoch in epochs))
                for name, epochs in best_epochs.items()}
    delta = f3_comparison["aggregate"]
    checks = {
        "f3_isrj_aupr_positive": delta["isrj_aupr"]["mean"] > 0,
        "f3_isrj_weak_recall_ge_0p030": delta["isrj_weak_recall"]["mean"] >= .030,
        "f3_isrj_weak_auroc_ge_0p010": delta["isrj_weak_auroc"]["mean"] >= .010,
        "four_of_five_positive_weak_recall": delta["positive_isrj_weak_recall_seeds"] >= 4,
        "four_of_five_positive_weak_auroc": delta["positive_isrj_weak_auroc_seeds"] >= 4,
        "f3_isrj_fpr_le_0p20": aggregate[F3_NAME]["isrj_fpr"]["mean"] <= .20,
        "test_boundary_safe": all(not row["test_v3_used"] for row in rows),
    }
    convergence = {
        "best_epochs": best_epochs, "epoch_cap_hits": cap_hits,
        "all_methods_platformed_before_cap": all(value == 0 for value in cap_hits.values()),
    }
    payload = {
        "status": "completed", "experiment": "RouteA_DevelopmentV3_LossConvergenceAudit",
        "seeds": args.seeds, "data_sha256": data_sha, "aggregate": aggregate,
        "strongest_long_budget_baseline": strongest,
        "f3_vs_strongest_long_budget": f3_comparison,
        "extended_vs_12_epoch": budget_deltas, "convergence": convergence,
        "f3_evidence_gate": {"passed": bool(all(checks.values())), "checks": checks},
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    path = args.output_root / "route_a_development_v3_loss_convergence_audit_summary.json"
    _write_json(path, payload)
    print(json.dumps({"status": "completed", "strongest": strongest,
                      "convergence": convergence, "f3_gate": payload["f3_evidence_gate"],
                      "summary": str(path)}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_mainstream_loss_baselines")
    parser.add_argument("--f3-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_loss_convergence_audit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--representation-lr-scale", type=float, default=.1)
    parser.add_argument("--saw-loss-weight", type=float, default=.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-fpr", type=float, default=.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.smoke = False
    args.smoke_epochs = args.epochs
    return args


if __name__ == "__main__":
    run(parse_args())
