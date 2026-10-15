"""Frozen-logit legal-combination MAP decoding on the AMA IF3 experiment.

No model is retrained.  The combination-prior coefficient is selected only on
the threshold-selection role, then applied to the check role and saved final
test logits.  This isolates joint decoding from representation and BCE loss.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.engine.combination_metrics import compute_cardinality_metrics, compute_combination_metrics  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_c1_baseline import initialize_f2  # noqa: E402
from experiments.goal_verify_ama_if3_bce_exact_20260905 import (  # noqa: E402
    CLASS_NAMES,
    DATA_ROOT,
    INIT,
    OUT as BASE_OUT,
    ArrayDataset,
    load_mat,
    named_combo_counts,
    sha256,
)

OUT = ROOT / "outputs" / "goal_ama_if3_map_decode_20260906"
SEEDS = (3050, 3051, 3052, 3053, 3054)
STRATEGY = "macro_aupr"
LAMBDA_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.00)


def code_vectors() -> np.ndarray:
    return ((np.arange(1, 64)[:, None] >> np.arange(6)[None, :]) & 1).astype(np.uint8)


def map_predictions(logits: np.ndarray, combo_counts: dict[str, int], prior_lambda: float) -> np.ndarray:
    """MAP over the 63 non-empty combinations under factorized BCE logits."""
    vectors = code_vectors().astype(np.float64)
    # log P(y|x) under independent Bernoulli logits, omitting only a common term.
    scores = logits.astype(np.float64) @ (2.0 * vectors.T - 1.0)
    if prior_lambda:
        prior = []
        total = float(sum(combo_counts.values()))
        for vector in vectors.astype(np.uint8):
            key = "+".join(CLASS_NAMES[i] for i in np.flatnonzero(vector))
            prior.append(np.log(max(float(combo_counts.get(key, 0)) / total, 1.0e-12)))
        scores = scores + float(prior_lambda) * np.asarray(prior, dtype=np.float64)[None, :]
    return vectors[np.argmax(scores, axis=1)].astype(np.uint8)


def evaluate_predictions(y: np.ndarray, logits: np.ndarray, predictions: np.ndarray, combo_counts: dict[str, int]) -> dict:
    y = y.astype(np.uint8); predictions = predictions.astype(np.uint8)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40.0, 40.0)))
    exact = np.all(y == predictions, axis=1)
    per_label = {}; aupr = []; auroc = []
    for col, name in enumerate(CLASS_NAMES):
        truth = y[:, col]; pos = truth == 1; neg = truth == 0
        a = float(average_precision_score(truth, probabilities[:, col]))
        r = float(roc_auc_score(truth, probabilities[:, col])) if len(np.unique(truth)) == 2 else float("nan")
        per_label[name] = {
            "aupr": a,
            "auroc": r,
            "recall": float(predictions[pos, col].mean()) if pos.any() else float("nan"),
            "negative_fpr": float(predictions[neg, col].mean()) if neg.any() else float("nan"),
            "f1": float(f1_score(truth, predictions[:, col], zero_division=0)),
        }
        aupr.append(a); auroc.append(r)
    combo = compute_combination_metrics(y, predictions, CLASS_NAMES, train_combo_counts=combo_counts)
    cardinality = compute_cardinality_metrics(y, predictions)
    rows = combo["per_combination"]
    frequency_weighted = float(sum(r["exact_match_accuracy"] * r["evaluation_count"] for r in rows) / len(y))
    return {
        "macro_aupr": float(np.nanmean(aupr)),
        "macro_auroc": float(np.nanmean(auroc)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(np.mean(y != predictions)),
        "sample_exact_accuracy": float(exact.mean()),
        "frequency_weighted_combination_accuracy": frequency_weighted,
        "macro_combination_accuracy": float(combo["summary"]["macro_combination_accuracy"]),
        "cardinality_stratified_exact": {str(r["cardinality"]): r["subset_accuracy"] for r in cardinality["per_cardinality"]},
        "seen_combination_accuracy": combo["summary"].get("seen_combination_accuracy"),
        "unseen_combination_accuracy": combo["summary"].get("unseen_combination_accuracy"),
        "combination_count": combo["summary"]["combination_count"],
        "per_label": per_label,
    }


def predict_logits(model, signal, labels, device):
    loader = DataLoader(ArrayDataset(signal, labels), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    outputs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            outputs.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return labels.astype(np.uint8), np.concatenate(outputs).astype(np.float32)


def summarize(rows, split: str):
    summary = {}
    for method in sorted(set(r["method"] for r in rows)):
        values = [r[split]["sample_exact_accuracy"] for r in rows if r["method"] == method]
        summary[method] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "per_seed": values}
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_summary = json.loads((BASE_OUT / "summary.json").read_text(encoding="utf-8"))
    role_manifest = json.loads((BASE_OUT / "role_manifest.json").read_text(encoding="utf-8"))
    train_signal, train_labels, _ = load_mat(DATA_ROOT / "closedset_6class_train.mat")
    threshold_indices = np.asarray(role_manifest["roles"]["threshold_selection"]["indices"], dtype=np.int64)
    check_indices = np.asarray(role_manifest["roles"]["check"]["indices"], dtype=np.int64)
    fit_indices = np.asarray(role_manifest["roles"]["fit"]["indices"], dtype=np.int64)
    combo_counts = named_combo_counts(train_labels[fit_indices])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for replay")
    rows = []
    model_meta = []
    for seed in SEEDS:
        seed_dir = BASE_OUT / f"seed_{seed}"
        model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
        initialize_f2(model, INIT)
        checkpoint = seed_dir / "epoch_08.pth"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
        threshold_signal, threshold_labels, _ = train_signal[threshold_indices], train_labels[threshold_indices], None
        check_signal, check_labels, _ = train_signal[check_indices], train_labels[check_indices], None
        threshold_y, threshold_logits = predict_logits(model, threshold_signal, threshold_labels, device)
        check_y, check_logits = predict_logits(model, check_signal, check_labels, device)
        test_payload = np.load(seed_dir / "test_logits_macro_aupr.npz")
        test_y, test_logits = np.asarray(test_payload["targets"], dtype=np.uint8), np.asarray(test_payload["logits"], dtype=np.float32)
        per_seed = []
        for prior_lambda in LAMBDA_GRID:
            threshold_pred = map_predictions(threshold_logits, combo_counts, prior_lambda)
            check_pred = map_predictions(check_logits, combo_counts, prior_lambda)
            test_pred = map_predictions(test_logits, combo_counts, prior_lambda)
            row = {
                "seed": int(seed),
                "method": "map_prior_lambda_" + str(prior_lambda),
                "prior_lambda": float(prior_lambda),
                "threshold_selection": evaluate_predictions(threshold_y, threshold_logits, threshold_pred, combo_counts),
                "check": evaluate_predictions(check_y, check_logits, check_pred, combo_counts),
                "test": evaluate_predictions(test_y, test_logits, test_pred, combo_counts),
            }
            rows.append(row); per_seed.append(row)
        model_meta.append({"seed": int(seed), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint)})
    # Select one lambda using only threshold-selection mean exact; tie to smaller lambda.
    selection = {}
    for lam in LAMBDA_GRID:
        values = [r["threshold_selection"]["sample_exact_accuracy"] for r in rows if r["prior_lambda"] == lam]
        selection[float(lam)] = float(np.mean(values))
    best_lambda = sorted(LAMBDA_GRID, key=lambda lam: (-selection[float(lam)], float(lam)))[0]
    selected = [r for r in rows if r["prior_lambda"] == best_lambda]
    base_test = [r["test"]["sample_exact_accuracy"] for r in base_summary["test_rows"] if r["strategy"] == STRATEGY and r["method"] == "combo_exact_coordinate_threshold"]
    base_fixed = [r["test"]["sample_exact_accuracy"] for r in base_summary["test_rows"] if r["strategy"] == STRATEGY and r["method"] == "fixed_0.5"]
    summary = {
        "schema": "goal-ama-if3-map-decode-summary-v1",
        "status": "completed",
        "base_experiment": str(BASE_OUT.resolve()),
        "data_sha256": base_summary["test_sha256"],
        "classes": list(CLASS_NAMES),
        "valid_combination_count": 63,
        "prior_source": "fit role combination counts only",
        "lambda_grid": list(LAMBDA_GRID),
        "threshold_selection_mean_exact_by_lambda": {str(k): v for k, v in selection.items()},
        "selected_lambda": float(best_lambda),
        "model_meta": model_meta,
        "all_rows": rows,
        "selected_threshold_selection": summarize(selected, "threshold_selection"),
        "selected_check": summarize(selected, "check"),
        "selected_test": summarize(selected, "test"),
        "baseline_comparison": {
            "fixed_0.5_test_mean": float(np.mean(base_fixed)),
            "combo_coordinate_threshold_test_mean": float(np.mean(base_test)),
            "map_selected_test_mean": float(np.mean([r["test"]["sample_exact_accuracy"] for r in selected])),
            "map_minus_combo_coordinate": float(np.mean([r["test"]["sample_exact_accuracy"] for r in selected]) - np.mean(base_test)),
            "map_minus_fixed_0.5": float(np.mean([r["test"]["sample_exact_accuracy"] for r in selected]) - np.mean(base_fixed)),
        },
        "evidence_boundary": "frozen-logit development diagnostic; lambda selected on threshold_selection; final test logits were already frozen in the base experiment",
        "locked_test_used": False,
    }
    (OUT / "config.json").write_text(json.dumps({"schema": "goal-ama-if3-map-decode-config-v1", "lambda_grid": list(LAMBDA_GRID), "prior_source": "fit role", "selection_role": "threshold_selection", "comparison": ["fixed_0.5", "combo_coordinate_threshold", "map_prior"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "selected_lambda": best_lambda, "selected_threshold_mean": selection[float(best_lambda)], "selected_test_mean": summary["baseline_comparison"]["map_selected_test_mean"], "map_minus_combo": summary["baseline_comparison"]["map_minus_combo_coordinate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
