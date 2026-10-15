"""Evaluate the single P2q high-lag strongest-peak rule on existing P2p development predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p2f_dedicated_sawfm_separator import metric
from cj0724.p2j_full_parameter_attachment import MODEL_SEEDS
from cj0724.p2m_analytic_saw_period import REGIMES


BOUNDARY_SECONDS = 800.0 / 10.0e6
PEARSON_TARGET = {"ID_control": 0.60, "near_low": 0.55, "near_high": 0.30, "far_low": 0.90, "far_high": 0.40}
EXACT_MAE_LIMIT_US = {"ID_control": 2.0, "near_low": 1.0, "near_high": 2.0, "far_low": 0.3, "far_high": 1.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def high_lag_strongest(legacy: np.ndarray, strongest: np.ndarray) -> np.ndarray:
    legacy = np.asarray(legacy, dtype=np.float64)
    strongest = np.asarray(strongest, dtype=np.float64)
    if legacy.shape != strongest.shape:
        raise ValueError("legacy and strongest shapes differ")
    return np.where(strongest >= BOUNDARY_SECONDS, strongest, legacy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="outputs/p2p_saw_peak_rule_development_20260801")
    parser.add_argument("--learned-baseline-root", default="outputs/p2m_analytic_saw_period_physics_20260801")
    parser.add_argument("--preregister", default="configs/p2q_high_lag_strongest_development_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p2q_high_lag_strongest_development_20260801")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source_root = project / args.source_root
    learned_root = project / args.learned_baseline_root
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for regime in REGIMES:
        for seed in MODEL_SEEDS:
            with np.load(source_root / regime / f"seed_{seed}" / "predictions.npz") as saved:
                target = saved["target"]
                predictions = {
                    "exact_legacy": saved["exact_legacy"],
                    "separated_legacy": saved["separated_legacy"],
                    "exact_candidate": high_lag_strongest(saved["exact_legacy"], saved["exact_strongest"]),
                    "separated_candidate": high_lag_strongest(saved["separated_legacy"], saved["separated_strongest"]),
                }
            with np.load(learned_root / regime / f"seed_{seed}" / "predictions.npz") as learned:
                learned_separated = learned["learned_separated"]
            row = {
                "regime": regime,
                "seed": seed,
                "exact_legacy": metric(predictions["exact_legacy"], target),
                "separated_legacy": metric(predictions["separated_legacy"], target),
                "exact_candidate": metric(predictions["exact_candidate"], target),
                "separated_candidate": metric(predictions["separated_candidate"], target),
                "learned_separated": metric(learned_separated, target),
                "candidate_use_fraction_exact": float(np.mean(predictions["exact_candidate"] != predictions["exact_legacy"])),
                "candidate_use_fraction_separated": float(np.mean(predictions["separated_candidate"] != predictions["separated_legacy"])),
            }
            rows.append(row)
            run_dir = output_root / regime / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            np.savez_compressed(run_dir / "predictions.npz", target=target, **predictions)

    per_regime = {}
    for regime in REGIMES:
        subset = [row for row in rows if row["regime"] == regime]
        per_regime[regime] = {
            "exact_mae_us_mean": float(np.mean([row["exact_candidate"]["mae_us"] for row in subset])),
            "exact_pearson_mean": float(np.mean([row["exact_candidate"]["pearson"] for row in subset])),
            "separated_mae_us_mean": float(np.mean([row["separated_candidate"]["mae_us"] for row in subset])),
            "legacy_separated_mae_us_mean": float(np.mean([row["separated_legacy"]["mae_us"] for row in subset])),
            "separated_pearson_mean": float(np.mean([row["separated_candidate"]["pearson"] for row in subset])),
            "legacy_separated_pearson_mean": float(np.mean([row["separated_legacy"]["pearson"] for row in subset])),
            "separated_pearson_by_seed": {str(row["seed"]): float(row["separated_candidate"]["pearson"]) for row in subset},
            "legacy_separated_pearson_by_seed": {str(row["seed"]): float(row["separated_legacy"]["pearson"]) for row in subset},
            "candidate_use_fraction_separated_mean": float(np.mean([row["candidate_use_fraction_separated"] for row in subset])),
        }
    gates = {
        "all_15_separated_runs_improve_mae_over_learned": bool(all(
            row["separated_candidate"]["mae"] < row["learned_separated"]["mae"] for row in rows
        )),
        "each_regime_separated_pearson": bool(all(
            value["separated_pearson_mean"] >= PEARSON_TARGET[regime] for regime, value in per_regime.items()
        )),
        "each_regime_separated_mae_no_more_than_legacy_factor": bool(all(
            value["separated_mae_us_mean"] <= 1.10 * value["legacy_separated_mae_us_mean"]
            for value in per_regime.values()
        )),
        "each_regime_exact_physics": bool(all(
            value["exact_pearson_mean"] >= 0.60 and value["exact_mae_us_mean"] <= EXACT_MAE_LIMIT_US[regime]
            for regime, value in per_regime.items()
        )),
        "hard_regimes_three_seed_direction": bool(all(
            per_regime[regime]["separated_pearson_by_seed"][str(seed)]
            > per_regime[regime]["legacy_separated_pearson_by_seed"][str(seed)]
            for regime in ("near_high", "far_high") for seed in MODEL_SEEDS
        )),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P2q high-lag strongest-peak SawFM development",
        "preregister_sha256": sha256_file(preregister),
        "rule": {"boundary_samples": 800, "below": "legacy", "at_or_above": "strongest"},
        "promotion_gates": gates,
        "development_passed": passed,
        "per_regime": per_regime,
        "runs": rows,
        "confirmation_identities_used": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if passed:
        (output_root / "method_freeze.json").write_text(json.dumps({
            "method": "P2q high-lag strongest-peak SawFM period rule",
            "boundary_samples": 800,
            "below_boundary": "legacy",
            "at_or_above_boundary": "strongest",
            "summary_sha256": sha256_file(output_root / "summary.json"),
            "preregister_sha256": sha256_file(preregister),
            "forbidden_prior_confirmation_seeds": [2800000000, 2900000000],
            "locked_test_used": False,
        }, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": passed, "promotion_gates": gates, "per_regime": per_regime}, indent=2))


if __name__ == "__main__":
    main()
