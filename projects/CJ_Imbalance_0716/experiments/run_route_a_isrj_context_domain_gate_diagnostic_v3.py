"""Frozen context-domain gate diagnostic for the ISRJ periodicity residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch  # Load PyTorch DLLs before sklearn/OpenMP on Windows PY10.
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


ISRJ_INDEX = 5
OTHER = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
VARIANTS = (
    "D0_C4",
    "D1_UnconditionalAligned",
    "D2_OracleContextAligned",
    "D3_SoftPredictedContextAligned",
    "D4_HardPredictedContextAligned",
    "D5_SoftPredictedContextBlockShuffled",
)


def _context_patterns(labels, weak):
    labels = np.asarray(labels, dtype=np.int8)
    weak = np.asarray(weak, dtype=bool)
    selected = (labels[:, ISRJ_INDEX] == 1) & weak[:, ISRJ_INDEX]
    if not np.any(selected):
        raise ValueError("fit role contains no weak ISRJ positives")
    return np.unique(labels[selected][:, OTHER], axis=0).astype(np.int8)


def _context_gates(other_targets, other_probabilities, other_predictions, patterns):
    targets = np.asarray(other_targets, dtype=np.int8)
    probabilities = np.clip(np.asarray(other_probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    predictions = np.asarray(other_predictions, dtype=np.int8)
    patterns = np.asarray(patterns, dtype=np.int8)
    oracle = np.any(np.all(targets[:, None, :] == patterns[None, :, :], axis=2), axis=1).astype(np.float32)
    hard = np.any(np.all(predictions[:, None, :] == patterns[None, :, :], axis=2), axis=1).astype(np.float32)
    likelihoods = []
    for pattern in patterns:
        likelihoods.append(np.prod(np.where(pattern[None, :] == 1, probabilities, 1.0 - probabilities), axis=1))
    soft = np.clip(np.sum(np.stack(likelihoods, axis=1), axis=1), 0.0, 1.0).astype(np.float32)
    return oracle, soft, hard


def _rank_metrics(target, weak, score):
    target = np.asarray(target, dtype=np.int8)
    weak = np.asarray(weak, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    weak_or_negative = ((target == 0) | ((target == 1) & weak))
    weak_target = ((target == 1) & weak)[weak_or_negative].astype(np.int8)
    return {
        "aupr": float(average_precision_score(target, score)),
        "weak_vs_negative_auroc": float(roc_auc_score(weak_target, score[weak_or_negative])),
    }


def _run_seed(args, labels6, weak6, seed):
    source_path = args.unified_root / f"seed_{seed}" / "C4_SameCapacity_NoVisibility" / "predictions.npz"
    residual_path = args.integration_root / f"seed_{seed}" / "isrj_expert_residuals.npz"
    source = np.load(source_path)
    residuals = np.load(residual_path)
    early_ids = np.asarray(source["sample_ids"], dtype=np.int64)
    if not np.array_equal(early_ids, np.asarray(residuals["early_sample_ids"], dtype=np.int64)):
        raise AssertionError("early sample IDs differ between C4 and residual artifacts")
    fit_ids = np.asarray(residuals["fit_sample_ids"], dtype=np.int64)
    patterns = _context_patterns(labels6[fit_ids], weak6[fit_ids])
    target = labels6[early_ids, ISRJ_INDEX].astype(np.int8)
    weak = weak6[early_ids]
    if not np.array_equal(np.asarray(source["targets"], dtype=np.int8), labels6[early_ids]):
        raise AssertionError("C4 target receipt does not match development-v3 labels")
    oracle, soft, hard = _context_gates(
        labels6[early_ids][:, OTHER], source["probabilities"][:, OTHER],
        source["predictions"][:, OTHER], patterns,
    )
    base = np.asarray(source["logits"], dtype=np.float32)[:, ISRJ_INDEX]
    aligned = np.asarray(residuals["early_aligned_residual"], dtype=np.float32)
    shuffled = np.asarray(residuals["early_shuffled_residual"], dtype=np.float32)
    scores = {
        VARIANTS[0]: base,
        VARIANTS[1]: base + aligned,
        VARIANTS[2]: base + aligned * oracle,
        VARIANTS[3]: base + aligned * soft,
        VARIANTS[4]: base + aligned * hard,
        VARIANTS[5]: base + shuffled * soft,
    }
    metrics = {name: _rank_metrics(target, weak[:, ISRJ_INDEX], score) for name, score in scores.items()}
    weak_positive = (target == 1) & weak[:, ISRJ_INDEX]
    negative = target == 0
    in_domain_negative = negative & (oracle > 0.5)
    out_domain_negative = negative & (oracle < 0.5)
    coverage = {
        gate: {
            "weak_positive_mean": float(np.mean(values[weak_positive])),
            "in_domain_negative_mean": float(np.mean(values[in_domain_negative])),
            "out_domain_negative_mean": float(np.mean(values[out_domain_negative])),
        }
        for gate, values in (("oracle", oracle), ("soft", soft), ("hard", hard))
    }
    return {
        "seed": int(seed),
        "fit_context_patterns": patterns.tolist(),
        "fit_context_pattern_count": int(len(patterns)),
        "metrics": metrics,
        "coverage": coverage,
        "source_sha256": _sha256(source_path),
        "residual_sha256": _sha256(residual_path),
    }


def _delta(rows, candidate, baseline, metric):
    values = [row["metrics"][candidate][metric] - row["metrics"][baseline][metric] for row in rows]
    return {
        **_mean_ci(values, 20308000 + sum(ord(c) for c in candidate + baseline + metric)),
        "per_seed": [{"seed": row["seed"], "delta": float(value)} for row, value in zip(rows, values)],
        "positive_seeds": int(sum(value > 0 for value in values)),
    }


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
    labels6 = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameters), args.weak_threshold_db)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "experiment_config.json", {
        "experiment": "RouteA_ISRJ_ContextDomainGateDiagnostic_v3",
        "seeds": seeds,
        "learned_parameters": 0,
        "oracle_is_diagnostic_only": True,
        "gates": {"oracle_vs_unconditional_weak_auroc": .010,
                  "soft_vs_unconditional_weak_auroc": .005,
                  "soft_oracle_recovery_fraction": .50,
                  "soft_vs_block_control_weak_auroc": .005,
                  "positive_seeds": 4},
        "locked_test_used": False, "test_v3_used": False,
    })
    rows = [_run_seed(args, labels6, weak6, seed) for seed in seeds]
    aggregate = {
        variant: {
            metric: _mean_ci([row["metrics"][variant][metric] for row in rows],
                             20309000 + vi * 100 + mi)
            for mi, metric in enumerate(("aupr", "weak_vs_negative_auroc"))
        }
        for vi, variant in enumerate(VARIANTS)
    }
    comparisons = {
        "oracle_vs_unconditional": _delta(rows, VARIANTS[2], VARIANTS[1], "weak_vs_negative_auroc"),
        "soft_vs_unconditional": _delta(rows, VARIANTS[3], VARIANTS[1], "weak_vs_negative_auroc"),
        "hard_vs_unconditional": _delta(rows, VARIANTS[4], VARIANTS[1], "weak_vs_negative_auroc"),
        "soft_aligned_vs_block_control": _delta(rows, VARIANTS[3], VARIANTS[5], "weak_vs_negative_auroc"),
    }
    if args.smoke:
        decision = {"passed": False, "evidence_level": "smoke_only", "checks": {}}
    else:
        oracle_gain = comparisons["oracle_vs_unconditional"]["mean"]
        soft_gain = comparisons["soft_vs_unconditional"]["mean"]
        recovery = float(soft_gain / oracle_gain) if oracle_gain > 0 else None
        checks = {
            "oracle_vs_unconditional_ge_0p010": oracle_gain >= .010,
            "soft_vs_unconditional_ge_0p005": soft_gain >= .005,
            "soft_positive_four_of_five": comparisons["soft_vs_unconditional"]["positive_seeds"] >= 4,
            "soft_recovers_half_oracle": recovery is not None and recovery >= .50,
            "soft_vs_block_control_ge_0p005": comparisons["soft_aligned_vs_block_control"]["mean"] >= .005,
        }
        decision = {"passed": bool(all(checks.values())), "checks": checks,
                    "soft_oracle_recovery_fraction": recovery,
                    "evidence_level": "five_seed_development_v3_frozen_diagnostic"}
    payload = {
        "status": "completed", "experiment": "RouteA_ISRJ_ContextDomainGateDiagnostic_v3",
        "smoke": bool(args.smoke), "seeds": seeds, "data_sha256": data_sha,
        "rows": rows, "aggregate": aggregate, "comparisons": comparisons,
        "deployable_fusion_permission_gate": decision,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(args.output_root / "route_a_isrj_context_domain_gate_diagnostic_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--integration-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_periodicity_bounded_residual_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_isrj_context_domain_gate_diagnostic_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
