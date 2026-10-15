"""Rule-constrained countermeasure candidates and cost-weighted utility."""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np


DEFAULT_ACTIONS = {
    "FMB": "frequency_domain_filtering",
    "SawFM": "sweep_tracking_suppression",
    "SinFM": "adaptive_frequency_tracking",
    "MFT": "multi_false_target_discrimination",
    "VFT": "doppler_consistency_filtering",
    "ISRJ": "interrupted_sampling_suppression",
}


def candidate_actions(
    predicted_components: Sequence[str],
    weak_components: Optional[Sequence[str]] = None,
    uncertainty: float = 0.0,
) -> Dict[str, object]:
    components = [str(value) for value in predicted_components]
    weak = set(str(value) for value in (weak_components or []))
    actions = [DEFAULT_ACTIONS[name] for name in components if name in DEFAULT_ACTIONS]
    risk = "low"
    if float(uncertainty) >= 0.5 or len(weak) >= 2:
        risk = "high"
        actions.append("conservative_composite_strategy")
    elif float(uncertainty) >= 0.2 or weak:
        risk = "medium"
    return {
        "components": components,
        "weak_components": sorted(weak),
        "candidate_actions": sorted(set(actions)),
        "risk_level": risk,
        "uncertainty": float(uncertainty),
        "claim_boundary": "rule-level candidate only; not real-radar closed-loop control",
    }


def evaluate_decision_utility(
    expected_actions: Sequence[Sequence[str]],
    predicted_actions: Sequence[Sequence[str]],
    action_costs: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    if len(expected_actions) != len(predicted_actions):
        raise ValueError("expected_actions and predicted_actions must have equal length")
    costs = {str(key): float(value) for key, value in (action_costs or {}).items()}
    exact = []
    losses = []
    for expected, predicted in zip(expected_actions, predicted_actions):
        target = set(str(value) for value in expected)
        output = set(str(value) for value in predicted)
        exact.append(target == output)
        missed = target.difference(output)
        extra = output.difference(target)
        loss = sum(costs.get(action, 1.0) for action in missed)
        loss += 0.5 * sum(costs.get(action, 1.0) for action in extra)
        losses.append(loss)
    return {
        "action_accuracy": float(np.mean(exact)) if exact else float("nan"),
        "cost_weighted_decision_loss": (
            float(np.mean(losses)) if losses else float("nan")
        ),
    }
