"""Retrained RawSaw local-pooling top-k 4/8/16 ablation on development-v3."""

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
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256, SAW_INDEX, _evaluate, _extract_backbone, _load_model_from_state,
    _load_roles, _raw_scores, _train_raw, _validate_config, _write_history,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


TOPKS = (4, 8, 16)
NON_SAW_INDICES = (0, 2, 3, 4, 5)
STATE_TOLERANCE = 1e-6


def _metric(metrics: dict, key: str) -> float:
    saw = metrics["per_label"]["SawFM"]
    values = {
        "macro_aupr": metrics["macro_aupr"], "macro_f1": metrics["macro_f1"],
        "saw_aupr": saw["aupr"], "saw_weak_recall": saw["weak_recall"],
        "saw_strong_recall": saw["strong_recall"], "saw_fpr": saw["negative_fpr"],
        "saw_weak_auroc": saw["weak_vs_negative_auroc"],
    }
    return float(values[key])


def _state_max_abs(model: torch.nn.Module, reference: dict) -> float:
    values = []
    for name, tensor in model.state_dict().items():
        if name not in reference:
            raise ValueError(f"source RawSaw state lacks {name}")
        values.append(float((tensor.detach().cpu().float() - reference[name].detach().cpu().float()).abs().max()))
    return max(values, default=0.0)


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    c6_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    c6_payload = torch.load(c6_path, map_location="cpu", weights_only=False)
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    backbone = _load_model_from_state(f2_dir, device, c6_payload["model_state_dict"])
    extracted = _extract_backbone(backbone, roles, device, args.feature_batch_size, topk=8)
    del backbone
    source_path = args.reference_root / f"seed_{seed}" / "checkpoints" / "rawsaw.pth"
    source_state = torch.load(source_path, map_location="cpu", weights_only=False)["model_state_dict"]
    variants, training, state_delta = {}, {}, {}
    for topk in TOPKS:
        arm = f"RawSaw_topk{topk}"
        arm_root = root / arm
        arm_args = copy.copy(args); arm_args.raw_topk = int(topk)
        model, history, info = _train_raw(roles, weak6, seed, device, arm_args)
        raw_scores = {role: _raw_scores(model, values["signals"], device, args.raw_batch_size) for role, values in roles.items()}
        logits = {role: np.asarray(values["base_logits"], dtype=np.float32).copy() for role, values in extracted.items()}
        for role in logits:
            logits[role][:, SAW_INDEX] = raw_scores[role]
        result = _evaluate(arm, logits, roles, weak6, arm_root, args)
        torch.save({"model_state_dict": model.state_dict(), **info, "topk": int(topk)}, arm_root / "rawsaw.pth")
        _write_history(arm_root / "history.csv", history)
        variants[arm] = result
        training[arm] = {**info, "topk": int(topk), "trainable_parameters": int(sum(value.numel() for value in model.parameters()))}
        state_delta[arm] = _state_max_abs(model, source_state) if topk == 16 else None
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    safety = {}
    reference = np.load(root / "RawSaw_topk16" / "predictions.npz")
    for topk in (4, 8):
        current = np.load(root / f"RawSaw_topk{topk}" / "predictions.npz")
        safety[f"topk{topk}"] = {
            "non_saw_probability_max_abs": float(np.max(np.abs(current["probabilities"][:, NON_SAW_INDICES] - reference["probabilities"][:, NON_SAW_INDICES]))),
            "non_saw_prediction_mismatch": int((current["predictions"][:, NON_SAW_INDICES] != reference["predictions"][:, NON_SAW_INDICES]).sum()),
        }
    payload = {
        "status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path),
        "c6_checkpoint_sha256": _sha256(c6_path), "reference_rawsaw_sha256": _sha256(source_path),
        "training": training, "variants": variants, "topk16_reference_state_max_abs": state_delta["RawSaw_topk16"],
        "non_saw_safety": safety, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(receipt, payload)
    return payload


def _summary(rows):
    metrics = ("macro_aupr", "macro_f1", "saw_aupr", "saw_weak_recall", "saw_strong_recall", "saw_fpr", "saw_weak_auroc")
    aggregate = {name: {metric: _mean_ci([_metric(row["variants"][name]["metrics"], metric) for row in rows], 2026081200 + 100 * index + offset) for offset, metric in enumerate(metrics)} for index, name in enumerate(f"RawSaw_topk{value}" for value in TOPKS)}
    comparisons = {}
    for topk in (4, 8):
        name = f"RawSaw_topk{topk}"; reference = "RawSaw_topk16"
        values = []
        for row in rows:
            values.append({"seed": row["seed"], **{metric: _metric(row["variants"][name]["metrics"], metric) - _metric(row["variants"][reference]["metrics"], metric) for metric in metrics}})
        comparisons[f"topk{topk}_minus_topk16"] = {"per_seed": values, "aggregate": {metric: _mean_ci([row[metric] for row in values], 2026081600 + 100 * topk + offset) for offset, metric in enumerate(metrics)}}
    return aggregate, comparisons


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 boundary is not sealed")
    labels = read_labels(data_path); parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in seeds]
    aggregate, comparisons = _summary(rows)
    checks = {} if args.smoke else {
        "same_trainable_parameter_count": len({row["training"][name]["trainable_parameters"] for row in rows for name in row["training"]}) == 1,
        "topk16_reproduces_frozen_rawsaw": max(row["topk16_reference_state_max_abs"] for row in rows) <= STATE_TOLERANCE,
        "non_saw_probabilities_exact": max(item["non_saw_probability_max_abs"] for row in rows for item in row["non_saw_safety"].values()) == 0.0,
        "non_saw_predictions_exact": sum(item["non_saw_prediction_mismatch"] for row in rows for item in row["non_saw_safety"].values()) == 0,
        "test_boundary_safe": all(not row["test_v3_used"] for row in rows),
    }
    payload = {
        "status": "completed", "experiment": "RouteA_RawSawTopK_4_8_16_Ablation", "smoke": bool(args.smoke), "seeds": list(seeds),
        "data_sha256": data_sha, "topks": list(TOPKS), "aggregate": aggregate, "comparisons_vs_topk16": comparisons,
        "construction_checks": {"checks": checks, "passed": bool(checks and all(checks.values()))},
        "interpretation": "core_ablation_only_no_candidate_or_threshold_selection", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_rawsaw_topk_4_8_16_ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--reference-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_rawsaw_topk_4_8_16_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=128); parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16); parser.add_argument("--raw-topk", type=int, default=16); parser.add_argument("--raw-dropout", type=float, default=.1); parser.add_argument("--raw-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--raw-epochs", type=int, default=30); parser.add_argument("--raw-patience", type=int, default=5); parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-negative-fpr", type=float, default=.20); parser.add_argument("--weak-threshold-db", type=float, default=10.0); parser.add_argument("--smoke-epochs", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
