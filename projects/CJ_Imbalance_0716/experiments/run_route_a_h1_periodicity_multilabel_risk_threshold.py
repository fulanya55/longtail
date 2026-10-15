"""Frozen multi-label probability risk calibrator for H1+B2 ISRJ decisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_h1_periodicity_context_threshold import (
    _select_context,
    _value,
)
from experiments.run_route_a_h1_periodicity_subgroup_safe_threshold import _worst_fpr
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import (
    _apply_frozen_calibration,
    _metrics,
)

ISRJ = 5
NON_ISRJ = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)


def _fit_risk_model(probabilities, targets, base_threshold):
    negatives = targets[:, ISRJ] == 0
    x = probabilities[negatives][:, NON_ISRJ]
    y = (probabilities[negatives, ISRJ] >= base_threshold).astype(np.uint8)
    if np.unique(y).size != 2:
        raise RuntimeError("risk calibration target must contain both classes")
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    model.fit(x, y)
    score = model.predict_proba(x)[:, 1]
    return model, {
        "negative_count": int(negatives.sum()),
        "false_positive_count": int(y.sum()),
        "false_positive_rate": float(y.mean()),
        "fit_auroc": float(roc_auc_score(y, score)),
        "coef": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "feature_labels": ["FMB", "MFT", "SawFM", "SinFM", "VFT"],
    }


def _risk_scores(model, probabilities):
    return model.predict_proba(probabilities[:, NON_ISRJ])[:, 1]


def _run_seed(args, labels, weak, seed):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    roles = np.load(args.c6_root / f"seed_{seed}" / "role_indices.npz")
    logits = np.load(args.increment_root / f"seed_{seed}" / "all_role_logits.npz")
    source = args.increment_root / f"seed_{seed}" / "B2_H1AlignedPeriodicity"
    calibration = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
    base_thresholds = np.asarray(
        json.loads((source / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    )

    calibration_ids = roles["calibration"].astype(np.int64)
    calibration_probabilities = _apply_frozen_calibration(
        logits["B2_H1AlignedPeriodicity__calibration"], calibration
    )
    risk_model, risk_fit = _fit_risk_model(
        calibration_probabilities,
        labels[calibration_ids],
        float(base_thresholds[ISRJ]),
    )

    threshold_ids = roles["threshold"].astype(np.int64)
    threshold_probabilities = _apply_frozen_calibration(
        logits["B2_H1AlignedPeriodicity__threshold"], calibration
    )
    threshold_risk_score = _risk_scores(risk_model, threshold_probabilities)
    threshold_risk_flag = threshold_risk_score >= 0.5
    threshold_negative = labels[threshold_ids, ISRJ] == 0
    threshold_fp = (
        threshold_probabilities[:, ISRJ] >= base_thresholds[ISRJ]
    ).astype(np.uint8)
    threshold_risk_auc = float(
        roc_auc_score(
            threshold_fp[threshold_negative],
            threshold_risk_score[threshold_negative],
        )
    )
    selection = _select_context(
        labels[threshold_ids, ISRJ],
        weak[threshold_ids, ISRJ],
        threshold_probabilities[:, ISRJ],
        threshold_risk_flag,
        float(base_thresholds[ISRJ]),
        args.overall_fpr_cap,
        args.risk_fpr_cap,
    )
    selection["threshold_risk_auroc"] = threshold_risk_auc
    selection["risk_probability_cutoff"] = 0.5

    early_ids = roles["early"].astype(np.int64)
    early_logits = logits["B2_H1AlignedPeriodicity__early"]
    early_probabilities = _apply_frozen_calibration(early_logits, calibration)
    early_risk_score = _risk_scores(risk_model, early_probabilities)
    early_risk_flag = early_risk_score >= 0.5
    base_predictions = early_probabilities >= base_thresholds[None, :]
    safe_predictions = base_predictions.copy()
    safe_predictions[early_risk_flag, ISRJ] = (
        early_probabilities[early_risk_flag, ISRJ] >= selection["risk_threshold"]
    )
    safe = _metrics(
        labels[early_ids], weak[early_ids], early_probabilities, safe_predictions
    )
    base = _metrics(
        labels[early_ids], weak[early_ids], early_probabilities, base_predictions
    )
    h1 = json.loads(
        (args.increment_root / f"seed_{seed}" / "B0_H1" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    worst, all_fpr = _worst_fpr(
        labels[early_ids], safe_predictions, args.min_negative
    )

    output = root / "C2_MultilabelProbabilityRiskThreshold"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "risk_model.json",
        {
            "source_role": "calibration",
            "model": "LogisticRegression",
            "C": 0.1,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 0,
            **risk_fit,
        },
    )
    _write_json(
        output / "selection.json",
        {
            "source_role": "threshold",
            "base_isrj_threshold": float(base_thresholds[ISRJ]),
            **selection,
        },
    )
    _write_json(output / "metrics.json", safe)
    _write_json(output / "subgroup_fpr.json", all_fpr)
    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=early_ids,
        targets=labels[early_ids],
        weak_mask=weak[early_ids].astype(np.uint8),
        logits=early_logits,
        probabilities=early_probabilities,
        predictions=safe_predictions.astype(np.uint8),
        risk_probability=early_risk_score,
        risk_flag=early_risk_flag.astype(np.uint8),
    )
    return {
        "seed": seed,
        "risk_fit": risk_fit,
        "selection": selection,
        "early_risk_fraction": float(early_risk_flag.mean()),
        "worst_early_subgroup": {"name": worst[0], "fpr": worst[1]},
        "safe_metrics": safe,
        "base_b2_metrics": base,
        "h1_metrics": h1,
        "probability_sha256": _sha256(source / "predictions.npz"),
        "test_v3_used": False,
    }


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
    weak = compute_weak_component_mask(
        labels, _component_power(values), args.weak_threshold_db
    )
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
    aggregate = {
        key: {
            "safe": _mean_ci(
                [_value(row["safe_metrics"], key) for row in rows], 20337000 + i
            ),
            "safe_vs_b2": _mean_ci(
                [
                    _value(row["safe_metrics"], key)
                    - _value(row["base_b2_metrics"], key)
                    for row in rows
                ],
                20337100 + i,
            ),
            "safe_vs_h1": _mean_ci(
                [
                    _value(row["safe_metrics"], key)
                    - _value(row["h1_metrics"], key)
                    for row in rows
                ],
                20337200 + i,
            ),
        }
        for i, key in enumerate(keys)
    }
    b2_gain = np.mean(
        [
            _value(row["base_b2_metrics"], "isrj_weak_recall")
            - _value(row["h1_metrics"], "isrj_weak_recall")
            for row in rows
        ]
    )
    retention = float(
        aggregate["isrj_weak_recall"]["safe_vs_h1"]["mean"] / b2_gain
    )
    mean_risk_auc = float(
        np.mean([row["selection"]["threshold_risk_auroc"] for row in rows])
    )
    checks = {
        "worst_subgroup_fpr_le_0p25": max(
            row["worst_early_subgroup"]["fpr"] for row in rows
        )
        <= 0.25,
        "mean_isrj_fpr_le_0p20": aggregate["isrj_fpr"]["safe"]["mean"] <= 0.20,
        "retain_80pct_b2_weak_gain": retention >= 0.80,
        "strong_recall_drop_le_0p03": aggregate["isrj_strong_recall"]["safe_vs_b2"]["mean"]
        >= -0.03,
        "macro_f1_drop_le_0p005": aggregate["macro_f1"]["safe_vs_b2"]["mean"] >= -0.005,
        "aupr_unchanged": abs(aggregate["isrj_aupr"]["safe_vs_b2"]["mean"]) < 1e-10,
        "weak_auroc_unchanged": abs(
            aggregate["isrj_weak_auroc"]["safe_vs_b2"]["mean"]
        )
        < 1e-10,
        "mean_threshold_risk_auroc_ge_0p60": mean_risk_auc >= 0.60,
        "four_of_five_risk_threshold_ge_base": sum(
            row["selection"]["risk_threshold"]
            >= json.loads(
                (
                    args.increment_root
                    / f"seed_{row['seed']}"
                    / "B2_H1AlignedPeriodicity"
                    / "thresholds.json"
                ).read_text(encoding="utf-8")
            )["thresholds"][ISRJ]
            for row in rows
        )
        >= 4,
        "test_boundary_safe": all(not row["test_v3_used"] for row in rows),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "status": "completed",
        "experiment": "RouteA_H1_Periodicity_MultilabelProbabilityRiskThreshold",
        "seeds": args.seeds,
        "rows": rows,
        "aggregate": aggregate,
        "weak_recall_gain_retention": retention,
        "mean_threshold_risk_auroc": mean_risk_auc,
        "development_gate": {"passed": bool(all(checks.values())), "checks": checks},
        "locked_test_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = args.output_root / "route_a_h1_periodicity_multilabel_risk_threshold_summary.json"
    _write_json(summary, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "retention": retention,
                "mean_threshold_risk_auroc": mean_risk_auc,
                "gate": result["development_gate"],
                "summary": str(summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat"
        ),
    )
    parser.add_argument(
        "--c6-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/tapc/route_a_development_v3_label_selective_projection",
    )
    parser.add_argument(
        "--increment-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/tapc/route_a_development_v3_h1_periodicity_increment",
    )
    parser.add_argument(
        "--test-v3-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/tapc/route_a_h1_periodicity_multilabel_risk_threshold",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--overall-fpr-cap", type=float, default=0.20)
    parser.add_argument("--risk-fpr-cap", type=float, default=0.22)
    parser.add_argument("--min-negative", type=int, default=50)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    run(parser.parse_args())
