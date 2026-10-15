"""Frozen-logit single-label error contribution decomposition.

This is an attribution/oracle analysis, not a causal intervention.  It reuses
the frozen test logits and threshold parameters from the governed AMA IF3
experiment and never retrains or tunes on test labels.
"""
from __future__ import annotations

import itertools
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


BASE = Path(r"E:\LJQ_code\CJ_Imbalance_0716\outputs\goal_ama_if3_bce_exact_20260905")
OUT = Path(r"E:\LJQ_code\CJ_Imbalance_0716\outputs\goal_ama_if3_single_label_contrib_20260907")
SEEDS = (3050, 3051, 3052, 3053, 3054)
CLASS_NAMES = ("FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ")
METHODS = ("fixed_0.5", "combo_exact_coordinate_threshold")
FACTORS = ("FMB_FP", "SawFM_FN", "MFT_FN", "ISRJ_FN")
FACTOR_COL = {"FMB_FP": 0, "SawFM_FN": 2, "MFT_FN": 4, "ISRJ_FN": 5}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictions(logits: np.ndarray, params: dict) -> np.ndarray:
    z = logits.astype(np.float64) / float(params.get("temperature", 1.0))
    bias = np.asarray(params.get("bias", np.zeros(6)), dtype=np.float64)
    z = z + bias[None, :]
    probs = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    thresholds = np.asarray(params.get("thresholds", np.full(6, 0.5)), dtype=np.float64)
    return (probs >= thresholds[None, :]).astype(np.uint8)


def combo_key(row: np.ndarray) -> str:
    active = [CLASS_NAMES[i] for i in np.flatnonzero(row)]
    return "+".join(active) if active else "<empty>"


def metric_summary(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict:
    y = y.astype(np.uint8); pred = pred.astype(np.uint8)
    exact = np.all(y == pred, axis=1)
    keys = np.asarray([combo_key(row) for row in y], dtype=object)
    per_combo = []
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        per_combo.append({"combination": key, "count": int(mask.sum()), "exact": float(np.mean(np.all(y[mask] == pred[mask], axis=1)))})
    cardinality = {}
    for k in sorted(np.unique(y.sum(axis=1)).tolist()):
        mask = y.sum(axis=1) == k
        cardinality[str(int(k))] = {"count": int(mask.sum()), "exact": float(exact[mask].mean())}
    per_label = {}
    auprs = []; aurocs = []
    for i, name in enumerate(CLASS_NAMES):
        truth = y[:, i]; pos = truth == 1; neg = truth == 0
        aupr = float(average_precision_score(truth, probs[:, i]))
        auroc = float(roc_auc_score(truth, probs[:, i])) if len(np.unique(truth)) == 2 else float("nan")
        per_label[name] = {
            "positive_count": int(pos.sum()),
            "false_positive_count": int(np.logical_and(~pos, pred[:, i] == 1).sum()),
            "false_negative_count": int(np.logical_and(pos, pred[:, i] == 0).sum()),
            "recall": float(pred[pos, i].mean()) if pos.any() else float("nan"),
            "negative_fpr": float(pred[neg, i].mean()) if neg.any() else float("nan"),
            "f1": float(f1_score(truth, pred[:, i], zero_division=0)),
            "aupr": aupr,
            "auroc": auroc,
        }
        auprs.append(aupr); aurocs.append(auroc)
    return {
        "sample_exact_accuracy": float(exact.mean()),
        "frequency_weighted_combination_accuracy": float(exact.mean()),
        "macro_combination_accuracy": float(np.mean([x["exact"] for x in per_combo])),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "hamming_loss": float(np.mean(y != pred)),
        "macro_aupr": float(np.nanmean(auprs)),
        "macro_auroc": float(np.nanmean(aurocs)),
        "cardinality_stratified_exact": cardinality,
        "per_combination": per_combo,
        "per_label": per_label,
        "total_exact_errors": int((~exact).sum()),
        "total_label_errors": int(np.sum(y != pred)),
    }


def factor_masks(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "FMB_FP": (y[:, 0] == 0) & (pred[:, 0] == 1),
        "SawFM_FN": (y[:, 2] == 1) & (pred[:, 2] == 0),
        "MFT_FN": (y[:, 4] == 1) & (pred[:, 4] == 0),
        "ISRJ_FN": (y[:, 5] == 1) & (pred[:, 5] == 0),
    }
    return {k: np.asarray(v, dtype=bool) for k, v in masks.items()}


def apply_oracle(y: np.ndarray, pred: np.ndarray, masks: dict[str, np.ndarray], selected: tuple[str, ...]) -> np.ndarray:
    fixed = pred.copy()
    for factor in selected:
        col = FACTOR_COL[factor]
        fixed[masks[factor], col] = y[masks[factor], col]
    return fixed


def subset_key(selected: tuple[str, ...]) -> str:
    return "none" if not selected else "+".join(selected)


def shapley_from_subsets(subset_scores: dict[str, float], baseline: float) -> dict[str, float]:
    n = len(FACTORS); out = {}
    for factor in FACTORS:
        others = [f for f in FACTORS if f != factor]
        value = 0.0
        for r in range(n):
            for subset in itertools.combinations(others, r):
                s = subset_key(tuple(f for f in FACTORS if f in subset))
                with_factor = subset_key(tuple(f for f in FACTORS if f in subset or f == factor))
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                value += weight * (subset_scores[with_factor] - subset_scores[s])
        out[factor] = float(value)
    return out


def full_label_oracles(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    baseline = float(np.mean(np.all(y == pred, axis=1)))
    for i, name in enumerate(CLASS_NAMES):
        fixed = pred.copy(); wrong = fixed[:, i] != y[:, i]; fixed[wrong, i] = y[wrong, i]
        exact = float(np.mean(np.all(y == fixed, axis=1)))
        out[name] = {"wrong_count": int(wrong.sum()), "oracle_exact": exact, "oracle_gain": exact - baseline}
    return out


def decomposition(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict:
    base_metric = metric_summary(y, pred, probs)
    masks = factor_masks(y, pred)
    baseline = base_metric["sample_exact_accuracy"]
    single = {}
    for factor in FACTORS:
        corrected = apply_oracle(y, pred, masks, (factor,))
        m = metric_summary(y, corrected, probs)
        single[factor] = {
            "error_count": int(masks[factor].sum()),
            "exact_after_oracle": m["sample_exact_accuracy"],
            "exact_gain": m["sample_exact_accuracy"] - baseline,
            "metric": m,
        }
    subset_scores = {}; subset_metrics = {}
    for r in range(len(FACTORS) + 1):
        for subset in itertools.combinations(FACTORS, r):
            key = subset_key(subset)
            corrected = apply_oracle(y, pred, masks, subset)
            m = metric_summary(y, corrected, probs)
            subset_scores[key] = m["sample_exact_accuracy"]
            subset_metrics[key] = m
    shapley = shapley_from_subsets(subset_scores, baseline)
    pair_interactions = {}
    pair_overlaps = {}
    for left, right in itertools.combinations(FACTORS, 2):
        pair = f"{left}__{right}"
        pair_interactions[pair] = float(
            subset_scores[subset_key((left, right))]
            - subset_scores[subset_key((left,))]
            - subset_scores[subset_key((right,))]
            + subset_scores["none"]
        )
        pair_overlaps[pair] = int(np.logical_and(masks[left], masks[right]).sum())
    all_four = subset_scores[subset_key(FACTORS)]
    correctable = all_four - baseline
    full_oracle = full_label_oracles(y, pred)
    four_error_counts = int(np.logical_or.reduce([masks[f] for f in FACTORS]).sum())
    return {
        "baseline": base_metric,
        "factor_error_counts": {f: int(masks[f].sum()) for f in FACTORS},
        "single_factor_oracle": single,
        "subset_exact_scores": subset_scores,
        "subset_metrics": subset_metrics,
        "shapley_exact_gain": shapley,
        "shapley_sum": float(sum(shapley.values())),
        "pairwise_error_overlap_count": pair_overlaps,
        "pairwise_exact_interaction": pair_interactions,
        "all_four_oracle_exact": all_four,
        "all_four_oracle_gain": correctable,
        "four_factor_error_union_count": four_error_counts,
        "correctable_exact_loss_fraction": float(correctable / (1.0 - baseline)) if baseline < 1.0 else float("nan"),
        "residual_exact_loss_after_four": float(1.0 - all_four),
        "full_label_oracle": full_oracle,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_summary = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    goal_prompt = """对已冻结 AMA IF3 六类 BCE logits 做单标签误差贡献分解：固定组合坐标阈值为主、0.5为敏感性对照；定义 FMB_FP、SawFM_FN、MFT_FN、ISRJ_FN 四因素，计算单因素 oracle、16 个子集 exact、Shapley exact gain、四因素可纠正损失占比、四因素后残余损失，并给出六标签完整 oracle。只复用既有冻结 logits 和阈值，不重新训练或用 test 调参。"""
    (OUT / "goal_prompt.md").write_text("# Goal prompt\n\n" + goal_prompt + "\n", encoding="utf-8")
    config = {
        "schema": "goal-ama-if3-single-label-contrib-config-v1",
        "base_output": str(BASE), "seeds": list(SEEDS), "classes": list(CLASS_NAMES),
        "methods": list(METHODS), "factors": list(FACTORS), "factor_columns": FACTOR_COL,
        "attribution": "frozen-prediction oracle and Shapley allocation; not causal intervention",
        "test_tuning": False, "retraining": False, "checkpoint_selection": "reused macro_aupr epoch-8 frozen checkpoints",
    }
    write_json(OUT / "config.json", config)
    all_rows = []
    for seed in SEEDS:
        npz = np.load(BASE / f"seed_{seed}" / "test_logits_macro_aupr.npz")
        y = np.asarray(npz["targets"], dtype=np.uint8)
        logits = np.asarray(npz["logits"], dtype=np.float32)
        frozen_row = next(r for r in base_summary["frozen_rows"] if r["seed"] == seed and r["strategy"] == "macro_aupr")
        for method in METHODS:
            params = frozen_row["threshold_methods"][method]["parameters"]
            pred = predictions(logits, params)
            probs = 1.0 / (1.0 + np.exp(-np.clip((logits / float(params.get("temperature", 1.0))) + np.asarray(params.get("bias", np.zeros(6)))[None, :], -40, 40)))
            result = decomposition(y, pred, probs)
            logits_path = BASE / f"seed_{seed}" / "test_logits_macro_aupr.npz"
            result.update({"seed": int(seed), "method": method, "threshold_parameters": params, "logits_file": str(logits_path.resolve()), "logits_sha256": sha256(logits_path)})
            write_json(OUT / f"seed_{seed}__{method}.json", result)
            all_rows.append(result)
    aggregates = {}
    for method in METHODS:
        rows = [r for r in all_rows if r["method"] == method]
        agg = {"seed_count": len(rows), "metrics": {}, "factors": {}, "full_label_oracle": {}}
        for metric in ("sample_exact_accuracy", "macro_f1", "hamming_loss", "macro_combination_accuracy", "total_exact_errors", "total_label_errors"):
            vals = np.asarray([r["baseline"][metric] for r in rows], dtype=float)
            agg["metrics"][metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)), "per_seed": vals.tolist()}
        for factor in FACTORS:
            gain = np.asarray([r["single_factor_oracle"][factor]["exact_gain"] for r in rows], dtype=float)
            count = np.asarray([r["factor_error_counts"][factor] for r in rows], dtype=float)
            shap = np.asarray([r["shapley_exact_gain"][factor] for r in rows], dtype=float)
            agg["factors"][factor] = {
                "error_count": {"mean": float(count.mean()), "std": float(count.std(ddof=1)), "per_seed": count.tolist()},
                "single_oracle_gain": {"mean": float(gain.mean()), "std": float(gain.std(ddof=1)), "per_seed": gain.tolist(), "positive_seed_count": int((gain > 0).sum())},
                "shapley_exact_gain": {"mean": float(shap.mean()), "std": float(shap.std(ddof=1)), "per_seed": shap.tolist(), "positive_seed_count": int((shap > 0).sum())},
            }
        pair_keys = sorted(rows[0]["pairwise_error_overlap_count"])
        agg["pairwise_overlap_and_interaction"] = {
            key: {
                "overlap_count": {
                    "mean": float(np.mean([r["pairwise_error_overlap_count"][key] for r in rows])),
                    "std": float(np.std([r["pairwise_error_overlap_count"][key] for r in rows], ddof=1)),
                    "per_seed": [r["pairwise_error_overlap_count"][key] for r in rows],
                },
                "exact_interaction": {
                    "mean": float(np.mean([r["pairwise_exact_interaction"][key] for r in rows])),
                    "std": float(np.std([r["pairwise_exact_interaction"][key] for r in rows], ddof=1)),
                    "per_seed": [r["pairwise_exact_interaction"][key] for r in rows],
                },
            }
            for key in pair_keys
        }
        for label in CLASS_NAMES:
            vals = np.asarray([r["full_label_oracle"][label]["oracle_gain"] for r in rows], dtype=float)
            counts = np.asarray([r["full_label_oracle"][label]["wrong_count"] for r in rows], dtype=float)
            agg["full_label_oracle"][label] = {"wrong_count_mean": float(counts.mean()), "wrong_count_std": float(counts.std(ddof=1)), "oracle_gain_mean": float(vals.mean()), "oracle_gain_std": float(vals.std(ddof=1)), "per_seed_gain": vals.tolist()}
        corr = np.asarray([r["all_four_oracle_gain"] for r in rows], dtype=float)
        residual = np.asarray([r["residual_exact_loss_after_four"] for r in rows], dtype=float)
        agg["four_factor"] = {
            "all_four_oracle_gain": {"mean": float(corr.mean()), "std": float(corr.std(ddof=1)), "per_seed": corr.tolist()},
            "residual_exact_loss": {"mean": float(residual.mean()), "std": float(residual.std(ddof=1)), "per_seed": residual.tolist()},
            "correctable_loss_fraction": {
                "mean": float(np.mean([r["correctable_exact_loss_fraction"] for r in rows])),
                "std": float(np.std([r["correctable_exact_loss_fraction"] for r in rows], ddof=1)),
            },
        }
        aggregates[method] = agg
        subset_payload = {str(r["seed"]): r["subset_exact_scores"] for r in rows}
        write_json(OUT / f"subset_matrix_{method}.json", subset_payload)
    summary = {
        "schema": "goal-ama-if3-single-label-contrib-summary-v1", "status": "completed", "base_output": str(BASE.resolve()), "base_summary_sha256": sha256(BASE / "summary.json"), "output": str(OUT.resolve()), "classes": list(CLASS_NAMES), "seeds": list(SEEDS), "methods": list(METHODS), "factors": list(FACTORS), "rows": all_rows, "aggregates": aggregates, "locked_test_used": False, "test_logits_provenance": "reused from frozen development final confirmation; no test tuning or retraining", "evidence_boundary": "frozen-prediction attribution, not causal effect and not locked-test benchmark"
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "terminal_receipt.json", {
        "status": "completed", "summary": str((OUT / "summary.json").resolve()),
        "row_count": len(all_rows), "seed_count": len(SEEDS), "method_count": len(METHODS),
        "subset_count_per_row": 16, "retraining": False, "test_tuning": False,
        "locked_test_used": False,
    })
    print(json.dumps({"status": "completed", "rows": len(all_rows), "output": str((OUT / "summary.json").resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
