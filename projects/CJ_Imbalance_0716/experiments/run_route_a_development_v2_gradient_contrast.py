"""Select paired C6-to-M5 checkpoint extrapolation on validation roles."""

import argparse
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
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    COMMON_INDICES,
    COMMON_NAMES,
    TARGET_NAMES,
    _comparison,
    _evaluate_variant,
    _sha256,
    _target_weak_rank,
    _validate_config,
    _write_json,
)
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _aggregate_variants as _unused_two_variant_aggregate,
    _component_power,
    _load_signal_array,
    _macro_aupr,
    _predict,
    _promotion_gate as _unused_two_variant_gate,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


GRID = (1.0, 1.5, 2.0, 3.0)
VARIANTS = ("C6_Joint", "GradientContrast")


def _interpolate_state(c6_state, m5_state, alpha):
    if c6_state.keys() != m5_state.keys():
        raise ValueError("paired checkpoints have different state keys")
    result = {}
    for key in c6_state:
        left, right = c6_state[key], m5_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("paired checkpoint tensor contract differs at {}".format(key))
        if torch.is_floating_point(left):
            result[key] = left + float(alpha) * (right - left)
        else:
            result[key] = right.clone()
    return result


def _load_checkpoint_state(path):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def _metric(row, path):
    value = row
    for key in path:
        value = value[key]
    return float(value)


def _variant_map(seed_row):
    return {row["variant"]: row for row in seed_row["variants"]}


def _aggregate(seed_rows):
    paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "hamming_loss": ("metrics", "hamming_loss"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "target_weak_vs_negative_auroc": ("metrics", "target_weak_vs_negative_auroc"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        rows = [_variant_map(seed_row)[variant] for seed_row in seed_rows]
        result[variant] = {
            name: _mean_ci([_metric(row, path) for row in rows], 20261300 + 100 * variant_index + offset)
            for offset, (name, path) in enumerate(paths.items())
        }
        result[variant]["per_label_aupr"] = {
            name: _mean_ci(
                [row["metrics"]["per_label_diagnostics"][name]["aupr"] for row in rows],
                20261400 + 100 * variant_index + index,
            )
            for index, name in enumerate(COMMON_NAMES)
        }
    return result


def _gate(comparison, aggregate):
    delta = comparison["aggregate"]
    candidate = aggregate["GradientContrast"]
    checks = {
        "macro_aupr_noninferior_minus_0p005": delta["macro_aupr"]["mean"] >= -0.005,
        "macro_f1_gain_0p010": delta["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_0p030": delta["target_mean_weak_recall"]["mean"] >= 0.030,
        "target_label_aupr_noninferior_minus_0p010": all(
            delta["per_label_aupr"][name]["mean"] >= -0.010 for name in TARGET_NAMES
        ),
        "target_mean_fpr_0p20": candidate["target_mean_negative_fpr"]["mean"] <= 0.20,
        "positive_target_weak_recall_at_least_four": delta["positive_target_weak_recall_seeds"] >= 4,
        "sawfm_present_macro_f1_positive": delta["sawfm_present_macro_f1"]["mean"] > 0.0,
        "sawfm_absent_macro_f1_noninferior_minus_0p005": delta["sawfm_absent_macro_f1"]["mean"] >= -0.005,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _run_seed(args, config, labels6, weak6, seed, device):
    source = Path(args.source_root) / "seed_{}".format(seed)
    source_summary = json.loads((source / "seed_summary.json").read_text(encoding="utf-8"))
    role_ids = np.load(source / "role_indices.npz")
    roles = {}
    for role in ("validation", "early", "calibration", "threshold"):
        ids = role_ids[role].astype(np.int64)
        values = _load_signal_array(_data_path(config), ids, str(config["normalization"]), args.io_batch_size)
        roles[role] = values
    c6_path = source / "C6_Joint" / "best_model.pth"
    m5_path = source / "M5_GradientIsolated" / "best_model.pth"
    c6_state = _load_checkpoint_state(c6_path)
    m5_state = _load_checkpoint_state(m5_path)
    f2_dir = Path(args.f2_root) / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    model, _ = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    validation_targets = roles["validation"]["targets6"][:, COMMON_INDICES]
    validation_weak = weak6[roles["validation"]["sample_ids"]][:, COMMON_INDICES]
    candidates = []
    for alpha in GRID:
        model.load_state_dict(_interpolate_state(c6_state, m5_state, alpha), strict=True)
        logits = _predict(model, roles["validation"]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES]
        if not np.isfinite(logits).all():
            candidates.append({
                "alpha": float(alpha),
                "validation_macro_aupr": None,
                "validation_target_weak_auroc": None,
                "eligible": False,
                "failure_reason": "nonfinite_validation_logits",
            })
            continue
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        candidates.append({
            "alpha": float(alpha),
            "validation_macro_aupr": _macro_aupr(validation_targets, probability),
            "validation_target_weak_auroc": _target_weak_rank(validation_targets, probability, validation_weak),
            "eligible": True,
        })
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("all gradient-contrast candidates produced nonfinite logits")
    selected = max(eligible, key=lambda row: (row["validation_macro_aupr"], row["validation_target_weak_auroc"]))
    model.load_state_dict(_interpolate_state(c6_state, m5_state, selected["alpha"]), strict=True)
    logits = {
        role: _predict(model, roles[role]["signals"], device, args.eval_batch_size)[:, COMMON_INDICES]
        for role in ("early", "calibration", "threshold")
    }
    targets = {role: roles[role]["targets6"][:, COMMON_INDICES] for role in logits}
    weak = {role: weak6[roles[role]["sample_ids"]][:, COMMON_INDICES] for role in logits}
    saw = {role: roles[role]["targets6"][:, 1].astype(bool) for role in logits}
    saw["early_sample_ids"] = roles["early"]["sample_ids"]
    output_dir = Path(args.output_root) / "seed_{}".format(seed) / "GradientContrast"
    candidate = _evaluate_variant(
        "GradientContrast", logits, targets, weak, saw, targets["calibration"], output_dir, args
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "selected_alpha": selected["alpha"],
        "source_c6": str(c6_path),
        "source_m5": str(m5_path),
    }, output_dir / "best_model.pth")
    seed_row = {
        "seed": int(seed),
        "variants": [
            next(row for row in source_summary["variants"] if row["variant"] == "C6_Joint"),
            candidate,
        ],
        "selection": {"grid": list(GRID), "candidates": candidates, "selected": selected},
        "source_split_sha256": source_summary["split_sha256"],
        "source_manifest_sha256": source_summary["manifest_sha256"],
        "source_c6_sha256": _sha256(c6_path),
        "source_m5_sha256": _sha256(m5_path),
        "locked_test_used": False,
        "test_v2_used": False,
    }
    _write_json(Path(args.output_root) / "seed_{}".format(seed) / "seed_summary.json", seed_row)
    return seed_row


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    labels6 = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameters), float(args.weak_threshold_db))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_rows = [_run_seed(args, config, labels6, weak6, int(seed), device) for seed in args.seeds]
    aggregate = _aggregate(seed_rows)
    comparison = _comparison(seed_rows, "GradientContrast", "C6_Joint")
    gate = _gate(comparison, aggregate)
    selected_alpha = {str(row["seed"]): row["selection"]["selected"]["alpha"] for row in seed_rows}
    payload = {
        "status": "completed",
        "experiment": "route_a_development_v2_gradient_contrast",
        "protocol": {"seeds": [int(seed) for seed in args.seeds], "grid": list(GRID), "locked_test_used": False, "test_v2_used": False},
        "data": {"path": str(data_path), "sha256": _sha256(data_path)},
        "selected_alpha": selected_alpha,
        "per_seed": seed_rows,
        "aggregate": aggregate,
        "comparison_gradient_contrast_minus_c6": comparison,
        "promotion_gate": gate,
        "next_decision": "preregister_one_time_test_v2_confirmation" if gate["passed"] else "stop_contrast_extrapolation",
        "locked_test_used": False,
        "test_v2_used": False,
    }
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "route_a_development_v2_gradient_contrast_summary.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_alpha": selected_alpha,
        "comparison": comparison["aggregate"],
        "promotion_gate": gate,
        "next_decision": payload["next_decision"],
        "locked_test_used": False,
        "test_v2_used": False,
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v2_shared_gradient")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v2_gradient_contrast")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
