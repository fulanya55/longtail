"""OOF hard-negative retraining for the 310D ISRJ periodicity expert."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import (
    _balanced_weights,
    _block_permutation,
    _block_shuffle,
    _extract,
    _scores,
)
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import (
    _load_roles,
    _matched_view,
    _rank_metrics,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (
    EXPECTED_V3_SHA256,
    _validate_config,
)
from experiments.run_shared import _data_path, load_config

ISRJ = 5
ROLES = ("fit", "validation", "early")
FAMILIES = ("aligned", "shuffled")


def _new_hgb(seed):
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=7,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=int(seed),
    )


def _oof_hard_negatives(x, y, seed, folds=5, quantile=0.75):
    y = np.asarray(y, dtype=np.uint8)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    score = np.full(len(y), np.nan, dtype=np.float32)
    for fold, (train, held) in enumerate(splitter.split(x, y)):
        model = _new_hgb(seed + fold + 1)
        model.fit(x[train], y[train], sample_weight=_balanced_weights(y[train]))
        score[held] = _scores(model, x[held])
    if not np.isfinite(score).all():
        raise RuntimeError("OOF score contains non-finite values")
    negative = y == 0
    cutoff = float(np.quantile(score[negative], float(quantile)))
    hard = negative & (score >= cutoff)
    if not hard.any():
        raise RuntimeError("empty OOF hard-negative set")
    return score, hard, cutoff


def _run_seed(args, config, weak, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(
        args.c6_root,
        seed,
        _data_path(config),
        config["normalization"],
        args.io_batch_size,
    )
    permutation, permutation_sha = _block_permutation(120, 20301000 + seed)
    raw = {role: _extract(roles[role]["signals"], device, args.feature_batch_size) for role in ROLES}
    shuffled = {
        role: _extract(
            _block_shuffle(roles[role]["signals"], permutation, args.block_size),
            device,
            args.feature_batch_size,
        )
        for role in ROLES
    }
    views = {}
    receipts = {}
    for role in ROLES:
        selected, target, receipt = _matched_view(
            roles[role]["targets6"], weak[roles[role]["sample_ids"]], ISRJ
        )
        if np.unique(target).size != 2:
            raise RuntimeError(f"matched view lacks both classes for {role}")
        views[role] = {"selected": selected, "target": target.astype(np.uint8)}
        receipts[role] = receipt
    families = {"aligned": raw, "shuffled": shuffled}
    results = {}
    for family_index, family in enumerate(FAMILIES):
        mean = families[family]["fit"].mean(axis=0).astype(np.float32)
        std = np.maximum(families[family]["fit"].std(axis=0), 1e-5).astype(np.float32)
        normalized = {
            role: ((families[family][role] - mean[None, :]) / std[None, :]).astype(np.float32)
            for role in ROLES
        }
        fit_x = normalized["fit"][views["fit"]["selected"]]
        fit_y = views["fit"]["target"]
        train_seed = 20341000 + family_index * 10000 + seed
        oof_score, hard, cutoff = _oof_hard_negatives(
            fit_x, fit_y, train_seed, args.oof_folds, args.hard_negative_quantile
        )
        base_weight = _balanced_weights(fit_y)
        hard_weight = base_weight.copy()
        hard_weight[hard] *= args.hard_negative_weight
        family_results = {}
        for variant, weight in (("base", base_weight), ("hard", hard_weight)):
            model = _new_hgb(train_seed + (100 if variant == "hard" else 0))
            model.fit(fit_x, fit_y, sample_weight=weight)
            metrics = {}
            predictions = {}
            for role in ("validation", "early"):
                x = normalized[role][views[role]["selected"]]
                score = _scores(model, x)
                metrics[role] = _rank_metrics(views[role]["target"], score)
                predictions[f"{role}_scores"] = score
                predictions[f"{role}_labels"] = views[role]["target"]
            out = root / f"{family}_{variant}"
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, out / "model.joblib")
            np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
            np.savez_compressed(out / "predictions.npz", **predictions)
            family_results[variant] = {
                "validation": metrics["validation"],
                "early": metrics["early"],
                "validation_early_auroc_gap": metrics["validation"]["auroc"] - metrics["early"]["auroc"],
                "model_sha256": _sha256(out / "model.joblib"),
                "normalization_sha256": _sha256(out / "normalization.npz"),
            }
        family_results["oof"] = {
            "folds": args.oof_folds,
            "hard_negative_quantile": args.hard_negative_quantile,
            "hard_negative_weight": args.hard_negative_weight,
            "negative_count": int((fit_y == 0).sum()),
            "positive_count": int((fit_y == 1).sum()),
            "hard_negative_count": int(hard.sum()),
            "score_cutoff": cutoff,
            "score_sha256": __import__("hashlib").sha256(oof_score.tobytes()).hexdigest(),
        }
        results[family] = family_results
    row = {
        "seed": seed,
        "status": "completed",
        "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_path),
        "permutation_sha256": permutation_sha,
        "matched_counts": receipts,
        "families": results,
        "test_v3_used": False,
    }
    _write_json(root / "seed_summary.json", row)
    return row


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
    labels = read_labels(data_path)
    values, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in args.seeds]
    def values(family, variant, role, metric):
        return [row["families"][family][variant][role][metric] for row in rows]
    aggregate = {
        family: {
            variant: {
                f"{role}_{metric}": _mean_ci(values(family, variant, role, metric), 20342000 + fi * 1000 + vi * 100 + ri * 10 + mi)
                for ri, role in enumerate(("validation", "early"))
                for mi, metric in enumerate(("auroc", "aupr"))
            }
            for vi, variant in enumerate(("base", "hard"))
        }
        for fi, family in enumerate(FAMILIES)
    }
    aligned_gain = np.asarray(values("aligned", "hard", "early", "auroc")) - np.asarray(values("aligned", "base", "early", "auroc"))
    aligned_aupr_gain = np.asarray(values("aligned", "hard", "early", "aupr")) - np.asarray(values("aligned", "base", "early", "aupr"))
    shuffle_gap = np.asarray(values("aligned", "hard", "early", "auroc")) - np.asarray(values("shuffled", "hard", "early", "auroc"))
    gap = np.asarray([row["families"]["aligned"]["hard"]["validation_early_auroc_gap"] for row in rows])
    comparisons = {
        "aligned_hard_vs_base_early_auroc": _mean_ci(aligned_gain.tolist(), 20343000),
        "aligned_hard_vs_base_early_aupr": _mean_ci(aligned_aupr_gain.tolist(), 20343001),
        "aligned_hard_vs_shuffled_hard_early_auroc": _mean_ci(shuffle_gap.tolist(), 20343002),
        "aligned_hard_validation_early_gap": _mean_ci(gap.tolist(), 20343003),
    }
    checks = {
        "aligned_auroc_gain_ge_0p010": comparisons["aligned_hard_vs_base_early_auroc"]["mean"] >= 0.010,
        "aligned_aupr_gain_nonnegative": comparisons["aligned_hard_vs_base_early_aupr"]["mean"] >= 0.0,
        "aligned_vs_shuffled_hard_ge_0p020": comparisons["aligned_hard_vs_shuffled_hard_early_auroc"]["mean"] >= 0.020,
        "four_of_five_positive_vs_base": int((aligned_gain > 0).sum()) >= 4,
        "four_of_five_above_shuffled_hard": int((shuffle_gap > 0).sum()) >= 4,
        "validation_early_gap_lt_0p15": comparisons["aligned_hard_validation_early_gap"]["mean"] < 0.15,
        "valid_oof_and_nonempty_hard_sets": all(row["families"][family]["oof"]["hard_negative_count"] > 0 for row in rows for family in FAMILIES),
        "same_hgb_configuration": True,
        "test_boundary_safe": all(not row["test_v3_used"] for row in rows),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "status": "completed",
        "experiment": "RouteA_ISRJ_Periodicity_OOFHardNegativeExpert",
        "seeds": args.seeds,
        "data_sha256": data_sha,
        "rows": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "component_gate": {"passed": bool(all(checks.values())), "checks": checks},
        "locked_test_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = args.output_root / "route_a_isrj_periodicity_oof_hard_negative_expert_summary.json"
    _write_json(summary, result)
    print(json.dumps({"status": "completed", "gate": result["component_gate"], "comparisons": comparisons, "summary": str(summary)}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_periodicity_oof_hard_negative_expert")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--hard-negative-quantile", type=float, default=0.75)
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    run(parser.parse_args())

