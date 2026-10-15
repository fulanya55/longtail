"""Final development-v3 saturation audit for the strongest Focal loss baseline."""

from __future__ import annotations

import argparse
import copy
import csv
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
from experiments.run_route_a_development_v3_loss_convergence_audit import _load_existing_loss  # noqa: E402
from experiments.run_route_a_development_v3_mainstream_loss_baselines import (  # noqa: E402
    EXPECTED_V3_SHA256, METRICS, _comparison, _load_model_from_state, _load_rawsaw,
    _reference_result, _route_rawsaw, _train_loss_model, _value,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    _evaluate, _load_roles, _raw_scores, _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


F2_MODEL_NAME = "F2_DFMS_Full"
FOCAL_TRAIN_NAME = "L2_Focal_RawSaw"
FOCAL36_NAME = "B2_Focal_36"
FOCAL72_NAME = "S2_Focal_Cosine72"
F3_NAME = "R1_F3_Reference"


def _plateau_from_history(path, gain_limit=.0005, span_limit=.0010, lr_limit=5.5e-6):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    if len(rows) < 12:
        raise ValueError("at least 12 epochs are required for plateau audit")
    tail = rows[-12:]
    macro = np.asarray([float(row["validation_common_macro_aupr"]) for row in tail])
    final_lr = float(tail[-1]["classifier_lr"])
    gain = float(macro[-1] - macro[0])
    span = float(np.max(macro) - np.min(macro))
    passed = abs(gain) <= gain_limit and span <= span_limit and final_lr <= lr_limit
    return {"passed": bool(passed), "last12_endpoint_gain": gain, "last12_span": span,
            "final_classifier_lr": final_lr, "epochs_ran": len(rows)}


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
    raw_scores = {role: _raw_scores(raw, values["signals"], device, args.raw_batch_size)
                  for role, values in roles.items()}
    del raw

    results = [
        _load_existing_loss(args.convergence_root, seed, "E2_Focal_36", FOCAL36_NAME),
        _reference_result(args.f3_root, seed, "F3_AlignedPeriodicityFusionHead", F3_NAME),
    ]
    model = _load_model_from_state(f2_dir, device, initial_state)
    model, training = _train_loss_model(
        model, FOCAL_TRAIN_NAME, roles, weak6, seed, device, args, root / FOCAL72_NAME
    )
    logits = {role: _predict(model, values["signals"], device, args.eval_batch_size)
              for role, values in roles.items()}
    results.append(_evaluate(FOCAL72_NAME, _route_rawsaw(logits, raw_scores), roles, weak6,
                             root / FOCAL72_NAME, args))
    checkpoint = root / FOCAL72_NAME / "best_model.pth"
    plateau = _plateau_from_history(root / FOCAL72_NAME / "history.csv")
    payload = {
        "seed": int(seed), "status": "completed", "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_path),
        "f2_checkpoint_sha256": _sha256(f2_checkpoint),
        "rawsaw_checkpoint_sha256": _sha256(raw_checkpoint),
        "training": training, "plateau": plateau,
        "model_sha256": _sha256(checkpoint), "variants": results,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


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
        "experiment": "RouteA_DevelopmentV3_FocalCosineSaturation",
        "seeds": args.seeds, "epochs": args.epochs, "lr_schedule": args.lr_schedule,
        "minimum_lr_ratio": args.minimum_lr_ratio, "full_schedule_no_early_stop": True,
        "locked_test_used": False, "test_v3_used": False,
    })
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in args.seeds]
    names = (FOCAL36_NAME, FOCAL72_NAME, F3_NAME)
    aggregate = {name: {metric: _mean_ci([
        _value(_variant_map(row)[name], metric) for row in rows
    ], 20323000 + ni * 100 + mi) for mi, metric in enumerate(METRICS)}
        for ni, name in enumerate(names)}
    strongest = max((FOCAL36_NAME, FOCAL72_NAME), key=lambda name: (
        aggregate[name]["macro_aupr"]["mean"], aggregate[name]["target_weak_auroc"]["mean"]))
    f3_comparison = _comparison(rows, F3_NAME, strongest, 20324000)
    budget_comparison = _comparison(rows, FOCAL72_NAME, FOCAL36_NAME, 20325000)
    plateau_count = int(sum(row["plateau"]["passed"] for row in rows))
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
    payload = {
        "status": "completed", "experiment": "RouteA_DevelopmentV3_FocalCosineSaturation",
        "seeds": args.seeds, "data_sha256": data_sha, "aggregate": aggregate,
        "strongest_traditional_baseline": strongest,
        "focal72_vs_focal36": budget_comparison,
        "f3_vs_strongest": f3_comparison,
        "plateau": {"passed_seed_count": plateau_count,
                    "passed": plateau_count >= 4,
                    "per_seed": [{"seed": row["seed"], **row["plateau"]} for row in rows]},
        "f3_evidence_gate": {"passed": bool(all(checks.values())), "checks": checks},
        "joint_gate": bool(plateau_count >= 4 and all(checks.values())),
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    path = args.output_root / "route_a_development_v3_focal_cosine_saturation_summary.json"
    _write_json(path, payload)
    print(json.dumps({"status": "completed", "strongest": strongest,
                      "plateau": payload["plateau"], "f3_gate": payload["f3_evidence_gate"],
                      "joint_gate": payload["joint_gate"], "summary": str(path)},
                     ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--convergence-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_loss_convergence_audit")
    parser.add_argument("--f3-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_focal_cosine_saturation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--patience", type=int, default=73)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--representation-lr-scale", type=float, default=.1)
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine"])
    parser.add_argument("--minimum-lr-ratio", type=float, default=.01)
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
