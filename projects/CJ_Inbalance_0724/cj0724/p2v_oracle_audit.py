"""Read-only Oracle and confidence audit for the failed P2v development run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CONSTANT_HZ = 4000.0
POWERS = (25, 35, 50)
REGIMES = ("ID_control", "near_low", "near_high", "far_negative", "far_negative_high")


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mae = float(np.mean(np.abs(prediction - target)))
    constant_mae = float(np.mean(np.abs(CONSTANT_HZ - target)))
    pearson = 0.0
    if len(target) >= 2 and np.std(prediction) > 0 and np.std(target) > 0:
        pearson = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "count": int(len(target)),
        "mae_hz": mae,
        "constant_mae_hz": constant_mae,
        "mae_reduction_vs_constant": float(1.0 - mae / max(constant_mae, 1.0e-12)),
        "pearson": pearson,
    }


def choose_oracle(target: np.ndarray, candidates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    errors = np.stack([np.abs(candidate - target) for candidate in candidates], axis=1)
    choice = errors.argmin(axis=1)
    prediction = np.choose(choice, candidates)
    return prediction, choice


def risk_metrics(error: np.ndarray, coherence: np.ndarray, threshold_hz: float = 5000.0) -> dict[str, float]:
    event = np.asarray(error) >= threshold_hz
    risk = 1.0 - np.asarray(coherence)
    if event.all() or (~event).all():
        roc_auc = 0.5
    else:
        roc_auc = float(roc_auc_score(event, risk))
    return {
        "catastrophic_threshold_hz": threshold_hz,
        "catastrophic_rate": float(event.mean()),
        "risk_aupr": float(average_precision_score(event, risk)),
        "risk_roc_auc": roc_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/p2v_vft_phase_increment_development_20260801")
    parser.add_argument("--output", default="outputs/p2v_vft_phase_increment_development_20260801/oracle_audit.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / args.root
    per_regime = {}
    for regime in REGIMES:
        with np.load(root / regime / "predictions.npz") as saved:
            arrays = {name: saved[name] for name in saved.files}
        target = arrays["target"]
        constant = np.full_like(target, CONSTANT_HZ)
        oracle_two, choice_two = choose_oracle(target, [arrays["C1_frozen_grid"], arrays["C2_phase_increment"]])
        oracle_three, choice_three = choose_oracle(target, [constant, arrays["C1_frozen_grid"], arrays["C2_phase_increment"]])
        pure = arrays["combination_size"] == 1
        compound = arrays["combination_size"] >= 2
        error = np.abs(arrays["C2_phase_increment"] - target)
        coverage = {}
        order = np.argsort(arrays["C2_coherence"])[::-1]
        for fraction in (0.25, 0.50, 0.75, 1.00):
            selected = order[: max(1, int(round(fraction * len(order))))]
            coverage[str(fraction)] = {
                "count": int(len(selected)),
                "mean_abs_error_hz": float(np.mean(error[selected])),
                "catastrophic_rate": float(np.mean(error[selected] >= 5000.0)),
            }
        per_regime[regime] = {
            "phase_pure": regression_metrics(arrays["C2_phase_increment"][pure], target[pure]),
            "phase_compound": regression_metrics(arrays["C2_phase_increment"][compound], target[compound]),
            "oracle_C1_C2": regression_metrics(oracle_two, target),
            "oracle_C0_C1_C2": regression_metrics(oracle_three, target),
            "oracle_C1_C2_choice_fraction": {
                "C1": float(np.mean(choice_two == 0)),
                "C2": float(np.mean(choice_two == 1)),
            },
            "oracle_C0_C1_C2_choice_fraction": {
                "C0": float(np.mean(choice_three == 0)),
                "C1": float(np.mean(choice_three == 1)),
                "C2": float(np.mean(choice_three == 2)),
            },
            "coherence_risk": risk_metrics(error, arrays["C2_coherence"]),
            "coherence_coverage": coverage,
            "by_power_oracle_C1_C2": {
                str(power): regression_metrics(
                    oracle_two[arrays["power"] == power], target[arrays["power"] == power]
                ) for power in POWERS
            },
        }
    oracle_two_sufficient = bool(all(
        value["oracle_C1_C2"]["mae_reduction_vs_constant"] >= 0.10
        and value["oracle_C1_C2"]["pearson"] >= 0.30
        for value in per_regime.values()
    ))
    oracle_three_sufficient = bool(all(
        value["oracle_C0_C1_C2"]["mae_reduction_vs_constant"] >= 0.10
        and value["oracle_C0_C1_C2"]["pearson"] >= 0.30
        for value in per_regime.values()
    ))
    result = {
        "experiment": "P2v post-failure Oracle and confidence audit",
        "evidence_role": "diagnostic_only_not_for_promotion",
        "oracle_C1_C2_sufficient": oracle_two_sufficient,
        "oracle_C0_C1_C2_sufficient": oracle_three_sufficient,
        "per_regime": per_regime,
        "decision": (
            "ORACLE_SUFFICIENT_BUT_ROUTING_FEATURE_INSUFFICIENT"
            if oracle_three_sufficient else
            "ORACLE_INSUFFICIENT_STOP_VFT_PHASE_INCREMENT_LINE"
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    output = project / args.output
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "oracle_C1_C2_sufficient": oracle_two_sufficient,
        "oracle_C0_C1_C2_sufficient": oracle_three_sufficient,
        "decision": result["decision"],
        "per_regime": {
            regime: {
                "oracle_two": value["oracle_C1_C2"],
                "oracle_three": value["oracle_C0_C1_C2"],
                "coherence_risk": value["coherence_risk"],
            } for regime, value in per_regime.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
