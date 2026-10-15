"""Aggregate the preregistered H3 T0/T1/T2 development comparison only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


MODES = ("T0-h3", "T1-h3", "T2-h3")
SEEDS = (20261003, 20261011, 20261019, 20261027, 20261104)


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    samples = np.asarray([values[rng.integers(0, values.size, values.size)].mean() for _ in range(draws)])
    return {
        "mean": float(values.mean()), "per_seed": values.tolist(),
        "bootstrap_95_ci": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
        "positive_seeds": int((values > 0).sum()), "negative_seeds": int((values < 0).sum()),
    }


def _load(root: Path, mode: str, seed: int) -> dict[str, object]:
    receipt_path = root / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed_development_unaggregated" or receipt.get("smoke"):
        raise ValueError(f"not a formal H3 development receipt: {receipt_path}")
    if receipt.get("modes") != [mode] or receipt.get("seeds") != [seed] or len(receipt.get("results", [])) != 1:
        raise ValueError(f"run identity mismatch: {receipt_path}")
    if any(receipt.get(key, True) for key in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError(f"test boundary violation: {receipt_path}")
    if receipt["config"].get("calibration") != "none" or receipt["config"].get("threshold_selection") != "none":
        raise ValueError(f"H3 must be ranking-only: {receipt_path}")
    result = receipt["results"][0]
    if result.get("mode") != mode or int(result.get("seed")) != seed:
        raise ValueError(f"result mismatch: {receipt_path}")
    predictions = root / f"{mode}_seed{seed}" / "development_predictions.npz"
    with np.load(predictions, allow_pickle=False) as data:
        probability = np.asarray(data["probabilities"], dtype=np.float64)
        label = np.asarray(data["labels"], dtype=bool)
        arm = np.asarray(data["vft_visibility_arm"], dtype=np.uint8)
    if probability.shape != label.shape or probability.ndim != 2 or probability.shape[1] != 6:
        raise ValueError(f"invalid H3 prediction shape: {predictions}")
    if set(np.unique(arm).tolist()) != {0, 1, 2}:
        raise ValueError(f"missing H3 visibility arm: {predictions}")
    def metric(present: int) -> dict[str, float]:
        chosen = (arm == 0) | (arm == present)
        y, p = label[chosen, 4], probability[chosen, 4]
        return {"auroc": float(roc_auc_score(y, p)), "aupr": float(average_precision_score(y, p))}
    weak, strong = metric(1), metric(2)
    stored = result["metrics"]
    if not (np.isclose(weak["auroc"], stored["weak_vft_vs_absent"]["auroc"], atol=1e-12) and np.isclose(strong["auroc"], stored["strong_vft_vs_absent"]["auroc"], atol=1e-12)):
        raise ValueError(f"immutable-prediction metric mismatch: {predictions}")
    return {"root": str(root), "result": result, "derived": {
        "weak_vft_auroc": weak["auroc"], "weak_vft_aupr": weak["aupr"],
        "strong_vft_auroc": strong["auroc"], "strong_vft_aupr": strong["aupr"],
        "macro_aupr": float(stored["macro_aupr"]),
        "non_vft_macro_aupr": float(np.mean([stored["per_label"][i]["aupr"] for i in (0, 1, 2, 3, 5)])),
    }}


def run(args: argparse.Namespace) -> dict[str, object]:
    table: dict[str, dict[int, dict[str, object]]] = {}
    for mode in MODES:
        table[mode] = {}
        for seed in args.seeds:
            root = Path(args.root_template.format(mode=mode.replace("-", "_"), seed=seed))
            table[mode][seed] = _load(root, mode, seed)
    comparisons: dict[str, dict[str, object]] = {}
    for name, left, right, metric in (
        ("T1_minus_T2_weak_vft_auroc", "T1-h3", "T2-h3", "weak_vft_auroc"),
        ("T1_minus_T2_weak_vft_aupr", "T1-h3", "T2-h3", "weak_vft_aupr"),
        ("T1_minus_T0_strong_vft_auroc", "T1-h3", "T0-h3", "strong_vft_auroc"),
        ("T1_minus_T0_macro_aupr", "T1-h3", "T0-h3", "macro_aupr"),
        ("T1_minus_T0_non_vft_macro_aupr", "T1-h3", "T0-h3", "non_vft_macro_aupr"),
    ):
        delta = np.asarray([table[left][seed]["derived"][metric] - table[right][seed]["derived"][metric] for seed in args.seeds], dtype=float)
        comparisons[name] = _bootstrap(delta, args.bootstrap_draws, args.bootstrap_seed)
    weak_auc, weak_aupr, strong = (comparisons[key] for key in ("T1_minus_T2_weak_vft_auroc", "T1_minus_T2_weak_vft_aupr", "T1_minus_T0_strong_vft_auroc"))
    capacity_equal = all(table["T1-h3"][seed]["result"]["parameter_count"] == table["T2-h3"][seed]["result"]["parameter_count"] for seed in args.seeds)
    gates = {
        "t1_minus_t2_weak_auroc_mean_ge_0p030": weak_auc["mean"] >= .030,
        "t1_minus_t2_weak_auroc_5of5_positive": weak_auc["positive_seeds"] == len(args.seeds),
        "t1_minus_t2_weak_auroc_ci_lower_gt_0": weak_auc["bootstrap_95_ci"][0] > 0,
        "t1_minus_t2_weak_aupr_mean_ge_0p025": weak_aupr["mean"] >= .025,
        "t1_minus_t2_weak_aupr_ci_lower_gt_0": weak_aupr["bootstrap_95_ci"][0] > 0,
        "t1_minus_t0_strong_auroc_mean_ge_minus_0p010": strong["mean"] >= -.010,
        "t1_minus_t0_strong_auroc_every_seed_ge_minus_0p020": min(strong["per_seed"]) >= -.020,
        "t1_t2_equal_parameter_count": capacity_equal,
        "all_runs_ranking_only_and_test_free": True,
    }
    output = {"status": "completed_development_summary", "experiment": "RouteA_H3_VFTThreeStrataIdentifiability", "seeds": list(args.seeds), "per_mode_per_seed": table, "comparison": comparisons, "gate": {"passed": bool(all(gates.values())), "checks": gates}, "evidence_boundary": "Development-only mechanism evidence. No calibration, threshold selection, external source, historical locked test, test-v2, or test-v3 was used.", "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-template", default="outputs/tapc/route_a_h3_vft_{mode}_seed{seed}")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_development_summary.json"))
    value = run(parser.parse_args())
    print(json.dumps({"status": value["status"], "gate": value["gate"]}, ensure_ascii=False))
