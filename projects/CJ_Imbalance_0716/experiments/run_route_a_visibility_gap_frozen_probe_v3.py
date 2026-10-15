"""Frozen final-temporal probe for continuous physical visibility on development-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _component_power,
)
from experiments.run_route_a_development_v3_label_selective_projection import (  # noqa: E402
    _load_model_from_state,
)
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES, TARGET_NAMES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _extract_backbone,
    _load_roles,
    _standardize,
    _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


def _relative_power_gap(labels6, component_power6, minimum_db=-40.0):
    """Power of each positive component relative to the strongest other positive."""
    labels = np.asarray(labels6).astype(bool)
    power = np.asarray(component_power6, dtype=np.float64)
    if labels.shape != power.shape:
        raise ValueError("labels and component powers must have the same shape")
    gap = np.full(labels.shape, np.nan, dtype=np.float32)
    for row in range(len(labels)):
        positive = np.flatnonzero(labels[row])
        if len(positive) == 1:
            index = int(positive[0])
            if np.isfinite(power[row, index]):
                gap[row, index] = 0.0
            continue
        for index in positive:
            other = positive[positive != index]
            value = power[row, index] - np.nanmax(power[row, other])
            if np.isfinite(value):
                gap[row, index] = np.float32(np.clip(value, float(minimum_db), 0.0))
    return gap


def _safe_correlation(function, truth, prediction):
    if len(truth) < 3 or np.std(truth) <= 1e-12 or np.std(prediction) <= 1e-12:
        return float("nan")
    return float(function(truth, prediction).statistic)


def _role_metrics(truth, prediction, weak_threshold_db):
    weak = truth <= -float(weak_threshold_db)
    strong = ~weak
    auroc = (float(roc_auc_score(weak.astype(np.uint8), -prediction))
             if weak.any() and strong.any() else float("nan"))
    return {
        "count": int(len(truth)),
        "weak_count": int(weak.sum()),
        "pearson": _safe_correlation(pearsonr, truth, prediction),
        "spearman": _safe_correlation(spearmanr, truth, prediction),
        "mae_db": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)),
        "weak_vs_strong_auroc": auroc,
    }


def _run_seed(args, config, labels6, gap6, seed, device, data_sha):
    output = args.output_root / f"seed_{seed}"
    summary_path = output / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    roles, role_indices_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    checkpoint_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    model = _load_model_from_state(f2_dir, device, state)
    extracted = _extract_backbone(model, roles, device, args.feature_batch_size, args.topk)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    stats = {role: values["temporal_stats"] for role, values in extracted.items()}
    standardized, mean, std = _standardize(stats["fit"], stats)
    labels_result = {}
    for index, name in zip(TARGET_INDICES, TARGET_NAMES):
        fit_ids = roles["fit"]["sample_ids"]
        fit_valid = (labels6[fit_ids, index] == 1) & np.isfinite(gap6[fit_ids, index])
        probe = Ridge(alpha=float(args.ridge_alpha), fit_intercept=True)
        probe.fit(standardized["fit"][fit_valid], gap6[fit_ids[fit_valid], index])
        label_result = {
            "fit_count": int(fit_valid.sum()),
            "coefficient_l2": float(np.linalg.norm(probe.coef_)),
            "intercept": float(probe.intercept_),
        }
        for role in ("validation", "early"):
            ids = roles[role]["sample_ids"]
            valid = (labels6[ids, index] == 1) & np.isfinite(gap6[ids, index])
            truth = gap6[ids[valid], index].astype(np.float64)
            prediction = probe.predict(standardized[role][valid]).astype(np.float64)
            label_result[role] = _role_metrics(truth, prediction, args.weak_threshold_db)
        labels_result[name] = label_result
    early_rows = [labels_result[name]["early"] for name in TARGET_NAMES]
    mean_spearman = float(np.nanmean([row["spearman"] for row in early_rows]))
    mean_auroc = float(np.nanmean([row["weak_vs_strong_auroc"] for row in early_rows]))
    passing_labels = int(sum(
        row["spearman"] >= 0.25 and row["weak_vs_strong_auroc"] >= 0.62
        for row in early_rows
    ))
    seed_gate = {
        "mean_spearman_ge_0p30": mean_spearman >= 0.30,
        "mean_weak_vs_strong_auroc_ge_0p65": mean_auroc >= 0.65,
        "two_of_three_labels_pass": passing_labels >= 2,
    }
    payload = {
        "seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
        "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
        "ridge_alpha": float(args.ridge_alpha), "gap_clip_db": float(args.gap_clip_db),
        "labels": labels_result,
        "early_mean_spearman": mean_spearman,
        "early_mean_weak_vs_strong_auroc": mean_auroc,
        "early_passing_labels": passing_labels,
        "seed_gate": {"checks": seed_gate, "passed": bool(all(seed_gate.values()))},
        "feature_standardization": {"mean_shape": list(mean.shape), "std_min": float(std.min())},
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def _aggregate(seed_rows):
    aggregate = {}
    for label_offset, name in enumerate(TARGET_NAMES):
        aggregate[name] = {}
        for metric_offset, metric in enumerate(("pearson", "spearman", "mae_db", "r2", "weak_vs_strong_auroc")):
            aggregate[name][metric] = _mean_ci(
                [row["labels"][name]["early"][metric] for row in seed_rows],
                20266100 + 100 * label_offset + metric_offset,
            )
    aggregate["mean_across_labels"] = {
        "spearman": _mean_ci([row["early_mean_spearman"] for row in seed_rows], 20266501),
        "weak_vs_strong_auroc": _mean_ci(
            [row["early_mean_weak_vs_strong_auroc"] for row in seed_rows], 20266502
        ),
    }
    return aggregate


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    sealed = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if sealed["test_v3"]["access_authorized"] or sealed["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    gap6 = _relative_power_gap(labels6, _component_power(parameter_values), -args.gap_clip_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    seed_rows = [_run_seed(args, config, labels6, gap6, seed, device, data_sha) for seed in seeds]
    aggregate = _aggregate(seed_rows)
    mean_spearman = aggregate["mean_across_labels"]["spearman"]["mean"]
    mean_auroc = aggregate["mean_across_labels"]["weak_vs_strong_auroc"]["mean"]
    passed_seeds = int(sum(row["seed_gate"]["passed"] for row in seed_rows))
    checks = {} if args.smoke else {
        "mean_spearman_ge_0p30": mean_spearman >= 0.30,
        "mean_weak_vs_strong_auroc_ge_0p65": mean_auroc >= 0.65,
        "four_of_five_seeds_pass": passed_seeds >= 4,
        "identity_and_test_safe": all(not row["test_v3_used"] for row in seed_rows),
    }
    gate = {"checks": checks, "passed": bool(checks and all(checks.values())),
            "passed_seeds": passed_seeds}
    payload = {
        "status": "completed", "experiment": "RouteA_VisibilityGap_FrozenProbe_v3",
        "smoke": bool(args.smoke), "seeds": [int(value) for value in seeds],
        "data_sha256": data_sha, "aggregate": aggregate, "promotion_gate": gate,
        "promoted": bool(gate["passed"]), "locked_test_used": False,
        "test_v2_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_visibility_gap_frozen_probe_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_visibility_gap_frozen_probe_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--gap-clip-db", type=float, default=40.0)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

