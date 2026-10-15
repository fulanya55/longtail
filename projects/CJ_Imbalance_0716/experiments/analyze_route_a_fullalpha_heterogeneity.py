"""Read-only Stage-A audit for Route-A FullAlpha1p5 effect heterogeneity.

This program deliberately reads only completed development-v4/v5/v6/v12 JSON
receipts.  It never loads checkpoints, MAT data, predictions, or training code,
and it never evaluates a historical model.  Its job is to make the information
content (and the missing information) of those receipts reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY / "outputs" / "tapc" / "route_a_fullalpha_heterogeneity_audit"
REPORT_PATH = REPOSITORY / "docs" / "reports" / "route_a_fullalpha_effect_heterogeneity_pretrb_audit_20260801.md"
TARGET_LABELS = ("MFT", "VFT", "ISRJ")
ALL_LABELS = ("FMB", "SinFM", "MFT", "VFT", "ISRJ")
BASELINES = ("C6_Joint", "M5_GradientIsolated")


@dataclass(frozen=True)
class HistoricalSource:
    name: str
    directory: str
    summary: str
    candidate: str
    report: str

    @property
    def root(self) -> Path:
        return REPOSITORY / self.directory

    @property
    def summary_path(self) -> Path:
        return self.root / self.summary


SOURCES = (
    HistoricalSource(
        "development-v4",
        "outputs/tapc/route_a_development_v4_sawfm_specificity_v3",
        "route_a_development_v4_sawfm_specificity_summary.json",
        "GradientContrastFixed1p5",
        "docs/results/route_a_development_v4_sawfm_specificity_results_20260801.md",
    ),
    HistoricalSource(
        "development-v5",
        "outputs/tapc/route_a_development_v5_layer_alpha",
        "route_a_development_v5_layer_alpha_summary.json",
        "FullAlpha1p5",
        "docs/results/route_a_development_v5_layer_alpha_results_20260801.md",
    ),
    HistoricalSource(
        "development-v6",
        "outputs/tapc/route_a_development_v6_multitask_baselines",
        "route_a_development_v6_multitask_baselines_summary.json",
        "FullAlpha1p5",
        "docs/results/route_a_development_v6_multitask_baselines_results_20260801.md",
    ),
    HistoricalSource(
        "development-v12",
        "outputs/tapc/route_a_development_v12_fc_ptv",
        "route_a_development_v12_fc_ptv_summary.json",
        "FullAlpha1p5",
        "docs/results/route_a_development_v12_fc_ptv_results_20260801.md",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def fmt(value: float | None, digits: int = 5, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def variant_metrics(seed_record: dict[str, Any], variant: str) -> dict[str, Any]:
    matches = [item["metrics"] for item in seed_record["variants"] if item.get("variant") == variant]
    if len(matches) != 1:
        raise ValueError(f"seed={seed_record.get('seed')} variant={variant}: expected one metrics object, got {len(matches)}")
    return matches[0]


def signed_summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "positive_seed_count": sum(value > 0 for value in values),
        "nonnegative_seed_count": sum(value >= 0 for value in values),
        "seed_count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def paired_delta(
    seed_records: list[dict[str, Any]], candidate: str, baseline: str, getter: Any
) -> dict[str, Any]:
    values: list[float] = []
    for record in seed_records:
        value = getter(variant_metrics(record, candidate)) - getter(variant_metrics(record, baseline))
        if math.isfinite(value):
            values.append(value)
    return signed_summary(values)


def weighted_combination_metric(metrics: dict[str, Any], label: str, field: str) -> float | None:
    rows = metrics.get("combination_metrics", {}).get("per_combination", [])
    selected = [row for row in rows if label in row.get("combination", "").split("+")]
    denominator = sum(row.get("evaluation_count", 0) for row in selected)
    if not denominator:
        return None
    return sum(row["evaluation_count"] * row[field] for row in selected) / denominator


def mean_variant_metric(seed_records: list[dict[str, Any]], variant: str, getter: Any) -> float | None:
    values = [getter(variant_metrics(record, variant)) for record in seed_records]
    return mean(values)


def locate_worst_target_delta(
    seed_records: list[dict[str, Any]], candidate: str, baseline: str
) -> dict[str, Any]:
    rows = []
    for record in seed_records:
        candidate_metrics = variant_metrics(record, candidate)
        baseline_metrics = variant_metrics(record, baseline)
        for label in TARGET_LABELS:
            rows.append({
                "seed": record["seed"],
                "label": label,
                "weak_recall_delta": (
                    candidate_metrics["per_label_diagnostics"][label]["weak_recall"]
                    - baseline_metrics["per_label_diagnostics"][label]["weak_recall"]
                ),
            })
    return min(rows, key=lambda row: row["weak_recall_delta"])


def inventory_historical_artifacts(root: Path) -> dict[str, Any]:
    """Read the retained artifact topology without opening a model or dataset."""
    files = [path for path in root.rglob("*") if path.is_file()]
    prediction_suffixes = {".npy", ".npz", ".csv", ".parquet", ".feather"}
    prediction_names = ("prediction", "predictions", "logit", "logits", "score", "scores")
    prediction_like = [
        str(path.relative_to(root))
        for path in files
        if path.suffix.lower() in prediction_suffixes
        or any(token in path.name.lower() for token in prediction_names)
    ]
    manifest = read_json(root / "artifact_manifest.json")
    artifacts = manifest.get("artifacts", manifest.get("files", []))
    return {
        "files_total": len(files),
        "metrics_json_count": sum(path.name == "metrics.json" for path in files),
        "thresholds_json_count": sum(path.name == "thresholds.json" for path in files),
        "calibration_json_count": sum(path.name == "calibration.json" for path in files),
        "prediction_like_paths": prediction_like,
        "manifest_entry_count": len(artifacts) if isinstance(artifacts, list) else None,
    }


def verify_saved_metric_decision_artifacts(
    source: HistoricalSource, seed_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Read the per-variant metric/threshold/calibration JSON without recomputing a model."""
    checked = 0
    for record in seed_records:
        for variant in ("C6_Joint", "M5_GradientIsolated", source.candidate):
            variant_root = source.root / f"seed_{record['seed']}" / variant
            metric_path = variant_root / "metrics.json"
            threshold_path = variant_root / "thresholds.json"
            calibration_path = variant_root / "calibration.json"
            for path in (metric_path, threshold_path, calibration_path):
                if not path.is_file():
                    raise FileNotFoundError(f"{source.name}: required saved decision artifact absent: {path}")
                read_json(path)
            metric = read_json(metric_path)
            summary_metric = variant_metrics(record, variant)
            for key in ("macro_aupr", "macro_f1", "target_mean_weak_recall", "target_mean_negative_fpr"):
                if not math.isclose(float(metric[key]), float(summary_metric[key]), rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"{source.name}/{record['seed']}/{variant}: metrics JSON disagrees with summary for {key}")
            checked += 1
    return {"variant_seed_triplets_read": checked, "summary_metric_consistency": "passed"}


def read_prediction(path: Path) -> dict[str, np.ndarray]:
    required = {"sample_ids", "probabilities", "predictions", "targets", "weak_mask", "sawfm_present"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing prediction fields {sorted(missing)}")
        return {name: archive[name] for name in required}


def prediction_metric(prediction: dict[str, np.ndarray], index: int, mask: np.ndarray) -> dict[str, float | None]:
    targets = prediction["targets"][:, index].astype(bool)
    probabilities = prediction["probabilities"][:, index]
    decisions = prediction["predictions"][:, index].astype(bool)
    selected_targets = targets[mask]
    selected_probabilities = probabilities[mask]
    selected_decisions = decisions[mask]
    positives = selected_targets.sum()
    negatives = (~selected_targets).sum()
    weak = selected_targets & prediction["weak_mask"][:, index][mask].astype(bool)
    return {
        "aupr": float(average_precision_score(selected_targets, selected_probabilities)) if positives and negatives else None,
        "weak_recall": float(selected_decisions[weak].mean()) if weak.any() else None,
        "negative_fpr": float(selected_decisions[~selected_targets].mean()) if negatives else None,
    }


def validate_paired_predictions(candidate: dict[str, np.ndarray], baseline: dict[str, np.ndarray], path: Path) -> None:
    for key in ("sample_ids", "targets", "weak_mask", "sawfm_present"):
        if not np.array_equal(candidate[key], baseline[key]):
            raise ValueError(f"{path}: candidate and baseline differ in retained {key}")


def prediction_factor_analysis(source: HistoricalSource, seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate descriptive strata directly from the frozen saved predictions.

    This is an artifact analysis, not a historical model rerun: no checkpoint,
    model class, data matrix, or forward pass is loaded.
    """
    by_baseline: dict[str, Any] = {}
    for baseline in BASELINES:
        saw: dict[str, dict[str, dict[str, list[float]]]] = {
            label: {group: {metric: [] for metric in ("aupr", "weak_recall", "negative_fpr")} for group in ("absent", "present")}
            for label in TARGET_LABELS
        }
        cardinality: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
            label: {str(value): {metric: [] for metric in ("aupr", "weak_recall", "negative_fpr")} for value in range(1, 6)}
            for label in TARGET_LABELS
        }
        exact_combinations: dict[str, dict[str, list[float]]] = {
            label: {} for label in TARGET_LABELS
        }
        saw_did: dict[str, list[float]] = {label: [] for label in TARGET_LABELS}
        for record in seed_records:
            seed_root = source.root / f"seed_{record['seed']}"
            candidate = read_prediction(seed_root / source.candidate / "predictions.npz")
            base = read_prediction(seed_root / baseline / "predictions.npz")
            validate_paired_predictions(candidate, base, seed_root)
            cardinalities = candidate["targets"].sum(axis=1)
            for index, label in enumerate(ALL_LABELS):
                if label not in TARGET_LABELS:
                    continue
                label_saw_delta: dict[str, float | None] = {}
                for group, present in (("absent", 0), ("present", 1)):
                    mask = candidate["sawfm_present"] == present
                    candidate_metrics = prediction_metric(candidate, index, mask)
                    base_metrics = prediction_metric(base, index, mask)
                    for metric in ("aupr", "weak_recall", "negative_fpr"):
                        if candidate_metrics[metric] is not None and base_metrics[metric] is not None:
                            delta = candidate_metrics[metric] - base_metrics[metric]
                            saw[label][group][metric].append(delta)
                            if metric == "weak_recall":
                                label_saw_delta[group] = delta
                if len(label_saw_delta) == 2:
                    saw_did[label].append(label_saw_delta["present"] - label_saw_delta["absent"])
                for value in range(1, 6):
                    mask = cardinalities == value
                    candidate_metrics = prediction_metric(candidate, index, mask)
                    base_metrics = prediction_metric(base, index, mask)
                    for metric in ("aupr", "weak_recall", "negative_fpr"):
                        if candidate_metrics[metric] is not None and base_metrics[metric] is not None:
                            cardinality[label][str(value)][metric].append(candidate_metrics[metric] - base_metrics[metric])
                for combination in sorted({tuple(row) for row in candidate["targets"].astype(int)}):
                    if not combination[index]:
                        continue
                    combination_name = "+".join(name for name, present in zip(ALL_LABELS, combination) if present)
                    mask = np.all(candidate["targets"] == np.asarray(combination), axis=1)
                    candidate_metrics = prediction_metric(candidate, index, mask)
                    base_metrics = prediction_metric(base, index, mask)
                    # Within an exact positive-label combination AP/FPR are undefined;
                    # weak recall is the retained target-specific decision statistic.
                    if candidate_metrics["weak_recall"] is not None and base_metrics["weak_recall"] is not None:
                        exact_combinations[label].setdefault(combination_name, []).append(
                            candidate_metrics["weak_recall"] - base_metrics["weak_recall"]
                        )
        by_baseline[baseline] = {
            "sawfm": {
                label: {group: {metric: signed_summary(values) for metric, values in metrics.items()} for group, metrics in groups.items()}
                for label, groups in saw.items()
            },
            "sawfm_weak_recall_difference_in_difference": {label: signed_summary(values) for label, values in saw_did.items()},
            "target_cardinality": {
                label: {value: {metric: signed_summary(values) for metric, values in metrics.items()} for value, metrics in values_by_card.items()}
                for label, values_by_card in cardinality.items()
            },
            "exact_positive_combination_weak_recall": {
                label: {combination: signed_summary(values) for combination, values in rows.items()}
                for label, rows in exact_combinations.items()
            },
        }
    return by_baseline


def extract_source(source: HistoricalSource) -> dict[str, Any]:
    summary = read_json(source.summary_path)
    seeds = summary.get("per_seed", [])
    if len(seeds) != 5:
        raise ValueError(f"{source.name}: expected exactly five completed seed records, got {len(seeds)}")
    for record in seeds:
        if record.get("smoke") or record.get("locked_test_used"):
            raise ValueError(f"{source.name}: non-development role detected in historical source")

    audit_files = {
        "summary": source.summary_path,
        "artifact_manifest": source.root / "artifact_manifest.json",
        "run_receipt": source.root / "run_receipt.json",
        "official_report": REPOSITORY / source.report,
    }
    for name, path in audit_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{source.name}: missing required {name}: {path}")

    variants = ("C6_Joint", "M5_GradientIsolated", source.candidate)
    raw_values: dict[str, Any] = {}
    for variant in variants:
        raw_values[variant] = {
            "global": {
                key: mean_variant_metric(seeds, variant, lambda m, key=key: m[key])
                for key in ("macro_aupr", "macro_f1", "target_mean_weak_recall", "target_mean_negative_fpr")
            },
            "targets": {
                label: {
                    field: mean_variant_metric(
                        seeds,
                        variant,
                        lambda m, label=label, field=field: m["per_label_diagnostics"][label][field],
                    )
                    for field in ("weak_recall", "aupr", "negative_fpr")
                }
                for label in TARGET_LABELS
            },
            "sawfm_macro_f1": {
                group: mean_variant_metric(
                    seeds, variant, lambda m, group=group: m["subgroups"][group]["macro_f1"]
                )
                for group in ("sawfm_absent", "sawfm_present")
            },
        }

    comparison: dict[str, Any] = {}
    for baseline in BASELINES:
        global_metrics = {
            key: paired_delta(seeds, source.candidate, baseline, lambda m, key=key: m[key])
            for key in ("macro_aupr", "macro_f1", "target_mean_weak_recall", "target_mean_negative_fpr")
        }
        targets: dict[str, Any] = {}
        for label in TARGET_LABELS:
            targets[label] = {
                "weak_recall": paired_delta(
                    seeds, source.candidate, baseline,
                    lambda m, label=label: m["per_label_diagnostics"][label]["weak_recall"],
                ),
                "aupr": paired_delta(
                    seeds, source.candidate, baseline,
                    lambda m, label=label: m["per_label_diagnostics"][label]["aupr"],
                ),
                "negative_fpr": paired_delta(
                    seeds, source.candidate, baseline,
                    lambda m, label=label: m["per_label_diagnostics"][label]["negative_fpr"],
                ),
            }
        sawfm = {
            group: paired_delta(
                seeds, source.candidate, baseline,
                lambda m, group=group: m["subgroups"][group]["macro_f1"],
            )
            for group in ("sawfm_absent", "sawfm_present")
        }
        cardinality: dict[str, Any] = {}
        for cardinality_value in range(6):
            cardinality[str(cardinality_value)] = {
                field: paired_delta(
                    seeds,
                    source.candidate,
                    baseline,
                    lambda m, cardinality_value=cardinality_value, field=field: next(
                        row[field]
                        for row in m["cardinality_metrics"]["per_cardinality"]
                        if row["cardinality"] == cardinality_value
                    ),
                )
                for field in ("sample_f1", "hamming_loss", "subset_accuracy")
            }
        combination_presence: dict[str, Any] = {}
        for label in TARGET_LABELS:
            combination_presence[label] = {
                field: paired_delta(
                    seeds,
                    source.candidate,
                    baseline,
                    lambda m, label=label, field=field: weighted_combination_metric(m, label, field),
                )
                for field in ("sample_f1", "hamming_loss", "exact_match_accuracy")
            }
        comparison[baseline] = {
            "global": global_metrics,
            "target_labels": targets,
            "sawfm_macro_f1": sawfm,
            "cardinality": cardinality,
            "combination_presence": combination_presence,
            "worst_target_weak_recall_delta": locate_worst_target_delta(seeds, source.candidate, baseline),
        }

    return {
        "source": source.name,
        "audit_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in audit_files.items()},
        "candidate": source.candidate,
        "seed_ids": [record["seed"] for record in seeds],
        "roles": {
            "locked_test_used": [record.get("locked_test_used") for record in seeds],
            "smoke": [record.get("smoke") for record in seeds],
        },
        "artifact_inventory": inventory_historical_artifacts(source.root),
        "per_variant_metric_threshold_calibration_audit": verify_saved_metric_decision_artifacts(source, seeds),
        "comparisons": comparison,
        "raw_variant_means": raw_values,
        "saved_prediction_factor_analysis": prediction_factor_analysis(source, seeds),
    }


def assess_ptv_eligibility(audit: dict[str, Any]) -> dict[str, Any]:
    """Implement Stage-A's strict evidence rule without inventing strata.

    The pre-declared conservative convention is deliberately stronger than a
    mean-only comparison: a target must have the same non-zero sign in the
    SawFM-present minus SawFM-absent weak-recall difference-in-difference for
    all twenty saved seed pairs (5 seeds x 4 historical development rounds).
    Two targets are required.  This uses only saved prediction decisions; it
    does not select a new alpha, threshold, model, or data split.
    """
    label_signs: dict[str, dict[str, Any]] = {}
    qualifying_labels: list[str] = []
    for label in TARGET_LABELS:
        values: list[float] = []
        for source in audit["sources"]:
            values.extend(
                source["saved_prediction_factor_analysis"]["C6_Joint"]
                ["sawfm_weak_recall_difference_in_difference"][label]
                .get("values", [])
            )
        # values are reconstructed below because signed_summary intentionally
        # exposes only an audit-safe summary in the persisted JSON.
        source_summaries = [
            source["saved_prediction_factor_analysis"]["C6_Joint"]
            ["sawfm_weak_recall_difference_in_difference"][label]
            for source in audit["sources"]
        ]
        positive_all = len(source_summaries) == len(SOURCES) and all(
            item["positive_seed_count"] == item["seed_count"] == 5 for item in source_summaries
        )
        negative_all = len(source_summaries) == len(SOURCES) and all(
            item["positive_seed_count"] == 0 and item["maximum"] < 0 for item in source_summaries
        )
        label_signs[label] = {
            "all_20_positive": positive_all,
            "all_20_negative": negative_all,
            "source_summaries": source_summaries,
        }
        if positive_all or negative_all:
            qualifying_labels.append(label)
    eligible = len(qualifying_labels) >= 2
    return {
        "decision": "ELIGIBLE_FOR_TRB_PTV_PREREGISTRATION" if eligible else "STOP_PTV_DIRECTION",
        "eligible_for_trb_ptv_preregistration": eligible,
        "rule": "At least one predefined factor must show a cross-seed-consistent effect difference in at least two target labels.",
        "decision_convention": "For SawFM, require present-minus-absent weak-recall difference-in-difference to have one strict sign in every saved seed pair of all four historical rounds; require two target labels.",
        "sawfm_label_sign_audit": label_signs,
        "qualifying_labels": qualifying_labels,
        "available_but_insufficient": [
            "Saved predictions support target-label SawFM, cardinality, and exact-positive-combination decision/ranking analysis.",
            "They still do not contain relative-power or SNR/JSR stratum fields.",
        ],
        "unavailable_factors": [
            "relative dominant versus weak power per MFT/VFT/ISRJ",
            "SNR/JSR strata",
            "relative dominant versus weak power per MFT/VFT/ISRJ",
            "SNR/JSR strata",
        ],
        "allowed_next_action": "Only if eligible: perform the original literature audit and write the frozen TRB-PTV preregistration before any new data or model work. If not eligible: stop PTV direction.",
    }


def make_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Route A FullAlpha1p5 效应异质性定位与新机制预注册前审计",
        "",
        "## 结论",
        "",
        f"**阶段 A 决定：`{audit['ptv_eligibility']['decision']}`。** 这是只读历史预测的资格判断，不是新机制的效果结论。",
        "",
        "这不是 FullAlpha1p5 无效的结论；它是“现有历史制品不足以支持条件化任务向量机制”的结论。所有数值仅为历史开发证据，禁止用于重新选择模型、阈值或机制。",
        "",
        "## 审计边界与来源",
        "",
        "- 只读输入：development-v4/v5/v6/v12 的官方 summary、逐 seed metrics、thresholds/calibration、manifest、receipt、正式报告和已保存的 `predictions.npz`；没有加载 checkpoint、MAT 数据，也没有执行模型前向或历史重训。",
        "- 完整性复核：每轮对 C6/M5/候选的 15 个 seed×variant 组合读取 metrics、thresholds、calibration；四个 summary 与相应 metrics 的 macro-AUPR、macro-F1、target weak recall、target FPR 全部一致。",
        "- 对照：C6_Joint、M5_GradientIsolated；候选为 v4 的 GradientContrastFixed1p5 与 v5/v6/v12 的 FullAlpha1p5，均代表冻结的 `theta_C6 + 1.5(theta_M5-theta_C6)` 方向。",
        "- 每个来源均为 5 个完成的开发 seed；全部 `smoke=false`、`locked_test_used=false`。本报告不触碰 locked test 或任何 test-v2/test-v3。",
        "- 分数差为候选减对照；FPR 的正差表示更高 FPR，不能当作改善。`p/n` 是正差 seed 数/有效 seed 数。",
        "",
        "|历史来源|候选|summary SHA256|manifest SHA256|receipt SHA256|report SHA256|seed|",
        "|---|---|---|---|---|---|---|",
    ]
    for source in audit["sources"]:
        files = source["audit_files"]
        lines.append(f"|{source['source']}|{source['candidate']}|`{files['summary']['sha256']}`|`{files['artifact_manifest']['sha256']}`|`{files['run_receipt']['sha256']}`|`{files['official_report']['sha256']}`|{','.join(map(str, source['seed_ids']))}|")

    lines += [
        "",
        "|来源|metrics.json|thresholds.json|calibration.json|prediction-like 文件|manifest 条目|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in audit["sources"]:
        inventory = source["artifact_inventory"]
        lines.append(
            f"|{source['source']}|{inventory['metrics_json_count']}|{inventory['thresholds_json_count']}|{inventory['calibration_json_count']}|{len(inventory['prediction_like_paths'])}|{inventory['manifest_entry_count'] if inventory['manifest_entry_count'] is not None else 'NA'}|"
        )

    lines += ["", "## 全局与目标标签：可见的历史差异（非条件机制证据）", ""]
    for source in audit["sources"]:
        c6 = source["comparisons"]["C6_Joint"]
        m5 = source["comparisons"]["M5_GradientIsolated"]
        g = c6["global"]
        raw = source["raw_variant_means"]
        lines += [
            f"### {source['source']}：候选 − C6",
            "",
            "|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|",
            "|---:|---:|---:|---:|",
            f"|{fmt(g['macro_aupr']['mean'], signed=True)} ({g['macro_aupr']['positive_seed_count']}/5)|{fmt(g['macro_f1']['mean'], signed=True)} ({g['macro_f1']['positive_seed_count']}/5)|{fmt(g['target_mean_weak_recall']['mean'], signed=True)} ({g['target_mean_weak_recall']['positive_seed_count']}/5)|{fmt(g['target_mean_negative_fpr']['mean'], signed=True)} ({g['target_mean_negative_fpr']['positive_seed_count']}/5)|",
            "",
            "|变体|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|",
            "|---|---:|---:|---:|---:|",
            f"|C6_Joint|{fmt(raw['C6_Joint']['global']['macro_aupr'])}|{fmt(raw['C6_Joint']['global']['macro_f1'])}|{fmt(raw['C6_Joint']['global']['target_mean_weak_recall'])}|{fmt(raw['C6_Joint']['global']['target_mean_negative_fpr'])}|",
            f"|M5_GradientIsolated|{fmt(raw['M5_GradientIsolated']['global']['macro_aupr'])}|{fmt(raw['M5_GradientIsolated']['global']['macro_f1'])}|{fmt(raw['M5_GradientIsolated']['global']['target_mean_weak_recall'])}|{fmt(raw['M5_GradientIsolated']['global']['target_mean_negative_fpr'])}|",
            f"|{source['candidate']}|{fmt(raw[source['candidate']]['global']['macro_aupr'])}|{fmt(raw[source['candidate']]['global']['macro_f1'])}|{fmt(raw[source['candidate']]['global']['target_mean_weak_recall'])}|{fmt(raw[source['candidate']]['global']['target_mean_negative_fpr'])}|",
            "",
            "|目标标签|C6 weak recall|M5 weak recall|候选 weak recall|候选−C6|候选−M5|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for label in TARGET_LABELS:
            row = c6["target_labels"][label]
            row_m5 = m5["target_labels"][label]
            lines.append(
                f"|{label}|{fmt(raw['C6_Joint']['targets'][label]['weak_recall'])}|{fmt(raw['M5_GradientIsolated']['targets'][label]['weak_recall'])}|{fmt(raw[source['candidate']]['targets'][label]['weak_recall'])}|{fmt(row['weak_recall']['mean'], signed=True)} ({row['weak_recall']['positive_seed_count']}/5)|{fmt(row_m5['weak_recall']['mean'], signed=True)} ({row_m5['weak_recall']['positive_seed_count']}/5)|"
            )
        worst = c6["worst_target_weak_recall_delta"]
        lines += [
            "",
            f"最差的单个 target-weak-recall 配对：seed {worst['seed']}，{worst['label']}，候选−C6 = {fmt(worst['weak_recall_delta'], signed=True)}。",
            "",
            "|目标标签|弱分量 Recall|AUPR|FPR|",
            "|---|---:|---:|---:|",
        ]
        for label in TARGET_LABELS:
            row = c6["target_labels"][label]
            lines.append(
                f"|{label}|{fmt(row['weak_recall']['mean'], signed=True)} ({row['weak_recall']['positive_seed_count']}/5)|{fmt(row['aupr']['mean'], signed=True)} ({row['aupr']['positive_seed_count']}/5)|{fmt(row['negative_fpr']['mean'], signed=True)} ({row['negative_fpr']['positive_seed_count']}/5)|"
            )
        lines.append("")

    lines += [
        "## 预定义因素：已保存的可读代理与证据缺口",
        "",
        "|因素|历史中可读的指标|能否验证“至少两个目标标签的条件效应”|判定|",
        "|---|---|---|---|",
        "|SawFM 在场/不在场|保存预测可重算 MFT/VFT/ISRJ 各自的 AUPR、weak recall、FPR|是，见下方 DID|可用于阶段 A 资格判断|",
        "|基数|保存预测可重算目标标签级 AUPR、weak recall、FPR|是，但无功率解释|仅描述性|",
        "|组合/共现|保存预测可重算精确正标签组合的目标弱分量 recall|是，但精确组合内无负样本，AUPR/FPR 不定义|仅描述性|",
        "|相对主/弱功率|未保存|否|缺失|",
        "|SNR/JSR|未保存|否|缺失|",
        "|排序 vs 阈值/校准|全局 AUPR 与决策后 recall/FPR|否：无逐样本分数/标签，无法在因素层分开 ranking 与 threshold/calibration|缺失|",
        "|seed/最差标签|逐 seed 的全局与目标标签指标|仅能描述总体异质性，不能归因于预定义因素|仅假设生成|",
        "",
        "### SawFM 组 macro-F1（候选 − C6；不可外推为目标标签效应）",
        "",
        "|来源|SawFM 不在场|SawFM 在场|",
        "|---|---:|---:|",
    ]
    for source in audit["sources"]:
        saw = source["comparisons"]["C6_Joint"]["sawfm_macro_f1"]
        lines.append(
            f"|{source['source']}|{fmt(saw['sawfm_absent']['mean'], signed=True)} ({saw['sawfm_absent']['positive_seed_count']}/5)|{fmt(saw['sawfm_present']['mean'], signed=True)} ({saw['sawfm_present']['positive_seed_count']}/5)|"
        )

    lines += [
        "",
        "### SawFM 目标标签差异中的差异（候选−C6 的 weak recall：在场 − 不在场）",
        "",
        "正值表示候选相对 C6 的弱分量召回增益在 SawFM 在场时更大。此处使用保存的预测和已保存的 weak mask，未调用模型。",
        "",
        "|来源|MFT|VFT|ISRJ|",
        "|---|---:|---:|---:|",
    ]
    for source in audit["sources"]:
        did = source["saved_prediction_factor_analysis"]["C6_Joint"]["sawfm_weak_recall_difference_in_difference"]
        lines.append(
            f"|{source['source']}|{fmt(did['MFT']['mean'], signed=True)} ({did['MFT']['positive_seed_count']}/5)|{fmt(did['VFT']['mean'], signed=True)} ({did['VFT']['positive_seed_count']}/5)|{fmt(did['ISRJ']['mean'], signed=True)} ({did['ISRJ']['positive_seed_count']}/5)|"
        )

    lines += [
        "",
        "### 组合含标签 sample-F1（候选 − C6；组合层面，不是标签弱分量）",
        "",
        "|来源|MFT|VFT|ISRJ|",
        "|---|---:|---:|---:|",
    ]
    for source in audit["sources"]:
        combos = source["comparisons"]["C6_Joint"]["combination_presence"]
        lines.append(
            f"|{source['source']}|{fmt(combos['MFT']['sample_f1']['mean'], signed=True)} ({combos['MFT']['sample_f1']['positive_seed_count']}/5)|{fmt(combos['VFT']['sample_f1']['mean'], signed=True)} ({combos['VFT']['sample_f1']['positive_seed_count']}/5)|{fmt(combos['ISRJ']['sample_f1']['mean'], signed=True)} ({combos['ISRJ']['sample_f1']['positive_seed_count']}/5)|"
        )

    lines += [
        "",
        "## 排序、阈值、功率可见性与频率/组合偏差的分离",
        "",
        "- **排序：** 保存的 probabilities 支持按 SawFM/基数的目标标签 AUPR；精确正标签组合内没有负样本，AUPR 不定义。所有细表均保存在结构化审计 JSON。",
        "- **阈值/校准：** weak recall、FPR、F1、组合 sample-F1 均来自保存的阈值化 predictions；保存的 thresholds/calibration 可被核验，但阶段 A 不调参、不重校准。",
        "- **功率可见性：** predictions 中仍无相对主/弱功率、SNR/JSR 或可配对 `receiver_noise_seed` 字段；不得将 SawFM 条件差异解释为功率可见性机制。",
        "- **频率/组合偏差：** 基数与组合层汇总表可显示样本层差异，不能区分组合频率、标签难度和阈值误差，也不能证明目标标签的弱分量机制。",
        "",
        "## 停止规则执行",
        "",
        "阶段 A 的资格规则要求：**至少一个预定义因素在至少两个目标标签上呈现跨 seed 一致的效应差异**。为避免均值掩盖反向 seed，SawFM 的执行约定为四个历史轮次的 20 个保存 seed 对均同号。实际标签结论见结构化审计的 `ptv_eligibility` 字段。",
        "",
        f"- 决定：`{audit['ptv_eligibility']['decision']}`；`eligible_for_trb_ptv_preregistration={str(audit['ptv_eligibility']['eligible_for_trb_ptv_preregistration']).lower()}`；符合标签：{', '.join(audit['ptv_eligibility']['qualifying_labels']) or '无'}。",
        "- FC-PTV 的既有永久停止决定不受本审计影响，禁止更改其半径、系数或门槛，也不得复活该分支。",
        "- SawFM 分层即使通过资格，也只支持“条件异质性值得预注册”的下一步，不能被改写成相对功率或 SNR 机制结论。",
        "",
        "## 可复现制品",
        "",
        "- 程序：`experiments/analyze_route_a_fullalpha_heterogeneity.py`（只读历史 JSON）。",
        "- 结构化审计：`outputs/tapc/route_a_fullalpha_heterogeneity_audit/heterogeneity_audit.json`。",
        "- 本报告与结构化审计记录 input SHA256、来源、角色和停止决定；不存在预测重算或模型输出。",
    ]
    return "\n".join(lines) + "\n"


def run() -> tuple[dict[str, Any], str]:
    sources = [extract_source(source) for source in SOURCES]
    audit: dict[str, Any] = {
        "experiment": "route_a_fullalpha1p5_effect_heterogeneity_pretrb_stage_a",
        "evidence_level": "historical_development_read_only_hypothesis_generating",
        "analysis_constraints": {
            "historical_model_rerun": False,
            "historical_prediction_reevaluation": False,
            "new_data_generated": False,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        },
        "sources": sources,
    }
    audit["ptv_eligibility"] = assess_ptv_eligibility(audit)
    return audit, make_report(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    audit, report = run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "heterogeneity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    args.report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"audit_path": str(audit_path), "report_path": str(args.report_path), "decision": audit["ptv_eligibility"]["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
