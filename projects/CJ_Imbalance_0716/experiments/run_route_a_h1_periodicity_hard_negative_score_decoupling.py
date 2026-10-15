"""Frozen-score hard-negative decoupling for the H1+B2 ISRJ increment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_h1_periodicity_context_threshold import _value
from experiments.run_route_a_h1_periodicity_subgroup_safe_threshold import _worst_fpr
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import (
    _fit_head,
    _head_scores,
    _replace_isrj,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _evaluate

ISRJ = 5
ALL_ROLES = ("fit", "validation", "early", "calibration", "threshold")
VARIANTS = ("D0_WeakOnlyStackedScore", "D1_HardNegativeDecoupledScore")


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _score_features(b0_logits, b1_logits, b2_logits):
    b1 = np.asarray(b1_logits, dtype=np.float32)[:, ISRJ]
    delta = np.asarray(b2_logits, dtype=np.float32)[:, ISRJ] - b1
    other_probability = _sigmoid(np.asarray(b0_logits, dtype=np.float32)[:, :ISRJ])
    fmb_probability = other_probability[:, 0]
    maximum_context = other_probability.max(axis=1)
    return np.stack(
        [b1, delta, delta * fmb_probability, delta * maximum_context], axis=1
    ).astype(np.float32)


def _weights(targets, weak, delta, weak_positive_weight=4.0, hard_negative_weight=4.0):
    positive = targets[:, ISRJ] == 1
    weak_positive = positive & weak[:, ISRJ]
    negative = ~positive
    cutoff = float(np.quantile(delta[negative], 0.75))
    hard_negative = negative & (delta >= cutoff)
    control = np.where(weak_positive, weak_positive_weight, 1.0).astype(np.float64)
    treatment = control.copy()
    treatment[hard_negative] = hard_negative_weight
    return control, treatment, cutoff, hard_negative


def _metric_rows(rows, variant, key):
    return [_value(row["variants"][variant]["metrics"], key) for row in rows]


def _run_seed(args, labels, weak, seed):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    role_path = args.c6_root / f"seed_{seed}" / "role_indices.npz"
    indices = np.load(role_path)
    role_data = {
        role: {
            "sample_ids": indices[role].astype(np.int64),
            "targets6": labels[indices[role].astype(np.int64)],
        }
        for role in ALL_ROLES
    }
    source_path = args.increment_root / f"seed_{seed}" / "all_role_logits.npz"
    source = np.load(source_path)
    b0 = {role: source[f"B0_H1__{role}"] for role in ALL_ROLES}
    b1 = {role: source[f"B1_H1OnlyWeightedHead__{role}"] for role in ALL_ROLES}
    b2 = {role: source[f"B2_H1AlignedPeriodicity__{role}"] for role in ALL_ROLES}
    features = {
        role: _score_features(b0[role], b1[role], b2[role]) for role in ALL_ROLES
    }
    fit_ids = role_data["fit"]["sample_ids"]
    fit_targets = role_data["fit"]["targets6"]
    delta = features["fit"][:, 1]
    control_weight, treatment_weight, cutoff, hard_negative = _weights(
        fit_targets,
        weak[fit_ids],
        delta,
        args.weak_positive_weight,
        args.hard_negative_weight,
    )
    variants = {}
    for offset, (name, sample_weight) in enumerate(
        zip(VARIANTS, (control_weight, treatment_weight))
    ):
        model, mean, std = _fit_head(
            features["fit"],
            fit_targets[:, ISRJ].astype(np.uint8),
            sample_weight,
            20338000 + seed + offset,
            args.c_value,
        )
        head_dir = root / f"head_{name}"
        head_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, head_dir / "model.joblib")
        np.savez_compressed(head_dir / "normalization.npz", mean=mean, std=std)
        logits = {
            role: _replace_isrj(
                b0[role], _head_scores(model, features[role], mean, std)
            )
            for role in ALL_ROLES
        }
        non_isrj_error = max(
            float(np.max(np.abs(logits[role][:, :ISRJ] - b0[role][:, :ISRJ])))
            for role in ALL_ROLES
        )
        evaluation = _evaluate(name, logits, role_data, weak, root / name, args)
        variants[name] = {
            "metrics": evaluation["metrics"],
            "thresholds": evaluation["thresholds"],
            "head_coefficient": model.coef_[0].astype(float).tolist(),
            "head_intercept": float(model.intercept_[0]),
            "non_isrj_max_abs_error": non_isrj_error,
        }
    b0_metrics = json.loads(
        (args.increment_root / f"seed_{seed}" / "B0_H1" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    b2_metrics = json.loads(
        (
            args.increment_root
            / f"seed_{seed}"
            / "B2_H1AlignedPeriodicity"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    prediction = np.load(root / VARIANTS[1] / "predictions.npz")["predictions"].astype(bool)
    worst, all_fpr = _worst_fpr(labels[role_data["early"]["sample_ids"]], prediction, args.min_negative)
    _write_json(root / VARIANTS[1] / "subgroup_fpr.json", all_fpr)
    row = {
        "seed": seed,
        "hard_negative_delta_q75": cutoff,
        "hard_negative_count": int(hard_negative.sum()),
        "fit_negative_count": int((fit_targets[:, ISRJ] == 0).sum()),
        "variants": variants,
        "b0_metrics": b0_metrics,
        "b2_metrics": b2_metrics,
        "worst_d1_subgroup": {"name": worst[0], "fpr": worst[1]},
        "source_logits_sha256": _sha256(source_path),
        "role_indices_sha256": _sha256(role_path),
        "test_v3_used": False,
    }
    _write_json(root / "seed_summary.json", row)
    return row


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if (
        seal["test_v3"]["access_authorized"]
        or seal["test_v3"]["analysis_allowed"]
        or seal["test_v3"]["used"]
    ):
        raise ValueError("test-v3 is not sealed")
    labels = read_labels(args.data_path)
    values, _ = read_parameters(args.data_path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    rows = [_run_seed(args, labels, weak, seed) for seed in args.seeds]
    keys = (
        "macro_aupr",
        "macro_f1",
        "isrj_aupr",
        "isrj_weak_recall",
        "isrj_strong_recall",
        "isrj_weak_auroc",
        "isrj_fpr",
    )
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        aggregate[variant] = {
            key: _mean_ci(_metric_rows(rows, variant, key), 20339000 + vi * 100 + i)
            for i, key in enumerate(keys)
        }
    comparisons = {}
    for ci, (name, left, right_getter) in enumerate(
        (
            ("D1_vs_D0", VARIANTS[1], lambda row: row["variants"][VARIANTS[0]]["metrics"]),
            ("D1_vs_B2", VARIANTS[1], lambda row: row["b2_metrics"]),
            ("D1_vs_B0", VARIANTS[1], lambda row: row["b0_metrics"]),
        )
    ):
        comparisons[name] = {
            key: _mean_ci(
                [
                    _value(row["variants"][left]["metrics"], key)
                    - _value(right_getter(row), key)
                    for row in rows
                ],
                20340000 + ci * 100 + i,
            )
            for i, key in enumerate(keys)
        }
    d1_b2 = comparisons["D1_vs_B2"]
    d1_d0 = comparisons["D1_vs_D0"]
    checks = {
        "weak_auroc_vs_b2_ge_0p005": d1_b2["isrj_weak_auroc"]["mean"] >= 0.005,
        "weak_auroc_vs_d0_ge_0p003": d1_d0["isrj_weak_auroc"]["mean"] >= 0.003,
        "isrj_aupr_vs_b2_nonnegative": d1_b2["isrj_aupr"]["mean"] >= 0.0,
        "weak_recall_vs_b2_nonnegative": d1_b2["isrj_weak_recall"]["mean"] >= 0.0,
        "mean_isrj_fpr_le_0p20": aggregate[VARIANTS[1]]["isrj_fpr"]["mean"] <= 0.20,
        "worst_subgroup_fpr_le_0p25": max(
            row["worst_d1_subgroup"]["fpr"] for row in rows
        )
        <= 0.25,
        "macro_f1_vs_b2_ge_minus_0p005": d1_b2["macro_f1"]["mean"] >= -0.005,
        "four_of_five_positive_weak_auroc_vs_b2": sum(
            _value(row["variants"][VARIANTS[1]]["metrics"], "isrj_weak_auroc")
            > _value(row["b2_metrics"], "isrj_weak_auroc")
            for row in rows
        )
        >= 4,
        "non_isrj_logits_unchanged": all(
            row["variants"][variant]["non_isrj_max_abs_error"] == 0.0
            for row in rows
            for variant in VARIANTS
        ),
        "test_boundary_safe": all(not row["test_v3_used"] for row in rows),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "status": "completed",
        "experiment": "RouteA_H1Periodicity_HardNegativeScoreDecoupling",
        "seeds": args.seeds,
        "rows": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "development_gate": {"passed": bool(all(checks.values())), "checks": checks},
        "locked_test_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = args.output_root / "route_a_h1_periodicity_hard_negative_score_decoupling_summary.json"
    _write_json(summary, result)
    print(json.dumps({"status": "completed", "gate": result["development_gate"], "summary": str(summary)}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat"))
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--increment-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_h1_periodicity_increment")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_periodicity_hard_negative_score_decoupling")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--weak-positive-weight", type=float, default=4.0)
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--min-negative", type=int, default=50)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    run(parser.parse_args())

