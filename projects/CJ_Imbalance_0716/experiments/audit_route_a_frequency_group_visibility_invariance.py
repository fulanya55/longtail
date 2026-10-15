"""Prove the direct visibility-blindness of frequency-only multilabel losses.

The purpose is not to claim that a frequency loss can never improve a weak
metric indirectly.  It establishes the narrower, testable fact needed by the
Route-A mechanism argument: if the weak/strong stratum is not an input to the
loss, swapping two same-label positive strata cannot change the objective or
their equal-logit gradient.  This is an analytic audit, not a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.losses.classification import DistributionBalancedLoss, MultiLabelClassificationLoss, effective_number_weights



def _soft_f1(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Generic train-only frequency-group SoftF1, without any visibility input."""
    group_threshold_logits = torch.tensor([0.0, -0.6190392], dtype=logits.dtype, device=logits.device)
    group_weights = torch.tensor([1.0, 1.5], dtype=logits.dtype, device=logits.device)
    probability = torch.sigmoid((logits - group_threshold_logits) / 0.5)
    tp = (probability * targets).sum(dim=0)
    fp = (probability * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - probability) * targets).sum(dim=0)
    f1 = (2.0 * tp + 1.0e-8) / (2.0 * tp + fp + fn + 1.0e-8)
    return ((1.0 - f1) * group_weights).sum() / group_weights.sum()


def _losses(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    positives = targets.sum(dim=0).detach().cpu().numpy().astype("int64")
    sample_count = int(targets.shape[0])
    pos_weight = torch.tensor([(sample_count - int(value)) / max(int(value), 1) for value in positives], dtype=logits.dtype)
    prior = torch.clamp(targets.mean(dim=0), 1.0e-4, 1.0 - 1.0e-4)
    cb_weight = torch.tensor(effective_number_weights(positives, beta=0.99), dtype=logits.dtype)
    focal = MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0)
    asl = MultiLabelClassificationLoss("asl", gamma_pos=0.0, gamma_neg=3.0, clip=0.05)
    db = DistributionBalancedLoss(positives, sample_count)
    return {
        "BCE": F.binary_cross_entropy_with_logits(logits, targets),
        "pos_weight_BCE": F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight),
        "Focal": focal.elementwise(logits, targets).mean(),
        "ASL": asl.elementwise(logits, targets).mean(),
        "DB": db.elementwise(logits, targets).mean(),
        "CB_BCE": F.binary_cross_entropy_with_logits(logits, targets, weight=cb_weight.view(1, -1)),
        "LogitAdjustment": F.binary_cross_entropy_with_logits(logits + torch.logit(prior).view(1, -1), targets),
        "FrequencyGroupSoftF1": _soft_f1(logits, targets),
    }


def _loss_and_gradient(name: str, logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
    variable = logits.detach().clone().requires_grad_(True)
    value = _losses(variable, targets)[name]
    gradient, = torch.autograd.grad(value, variable)
    return float(value.detach().cpu()), gradient.detach().cpu()


def run(output: Path, test_v3_manifest: Path) -> dict[str, object]:
    seal = json.loads(test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(bool(seal[key]) for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    # Rows 0 and 1 are identical label-0 positives with identical logit.  The
    # visibility masks differ only in their names; no evaluated loss receives it.
    logits = torch.tensor([[0.20, -0.40], [0.20, 0.70], [-0.70, -0.10], [0.80, -0.90]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    weak_a = torch.tensor([[True, False], [False, False], [False, False], [False, False]])
    weak_b = torch.tensor([[False, False], [True, False], [False, False], [False, False]])
    rows: dict[str, object] = {}
    for name in _losses(logits, targets):
        loss_a, gradient_a = _loss_and_gradient(name, logits, targets)
        loss_b, gradient_b = _loss_and_gradient(name, logits, targets)
        rows[name] = {
            "loss_under_weak_assignment_a": loss_a,
            "loss_under_weak_assignment_b": loss_b,
            "loss_equal_under_stratum_swap": bool(abs(loss_a - loss_b) <= 1.0e-12),
            "gradient_equal_under_stratum_swap": bool(torch.allclose(gradient_a, gradient_b, atol=1.0e-12, rtol=0.0)),
            "same_label_equal_logit_positive_gradients_equal": bool(abs(float(gradient_a[0, 0] - gradient_a[1, 0])) <= 1.0e-12),
            "positive_gradient_row0_label0": float(gradient_a[0, 0]),
            "positive_gradient_row1_label0": float(gradient_a[1, 0]),
        }
    equal_gradient_losses = [
        name for name, row in rows.items() if row["same_label_equal_logit_positive_gradients_equal"]
    ]
    cooccurrence_sensitive_losses = [
        name for name, row in rows.items() if not row["same_label_equal_logit_positive_gradients_equal"]
    ]
    checks = {
        "weak_masks_have_same_label_positive_count": int(weak_a.sum()) == int(weak_b.sum()) == 1,
        "every_loss_is_mask_swap_invariant": all(row["loss_equal_under_stratum_swap"] for row in rows.values()),
        "every_loss_gradient_is_mask_swap_invariant": all(row["gradient_equal_under_stratum_swap"] for row in rows.values()),
        "test_v3_sealed": True,
    }
    payload = {
        "status": "completed_analytic_visibility_invariance_audit",
        "experiment": "RouteA_FrequencyGroupVisibilityInvariance",
        "evidence_role": "analytic_loss_mechanism_audit_not_performance_evidence",
        "losses": rows,
        "same_label_equal_logit_gradient_summary": {
            "equal_gradient_losses": equal_gradient_losses,
            "cooccurrence_sensitive_losses": cooccurrence_sensitive_losses,
        },
        "checks": checks,
        "interpretation": {
            "proved": "Frequency-only and label-only objectives tested here do not directly distinguish weak from strong positives of one label when label and logit are held equal.",
            "cooccurrence_boundary": "A distribution-balanced objective can assign different gradients to same-label positives when their other label co-occurrences differ. That is combination/frequency conditioning, not dependence on an unobserved weak/strong visibility mask.",
            "not_proved": "They can never change weak metrics indirectly through feature learning or data distribution; that requires empirical evidence.",
            "protocol_consequence": "A frequency-group SoftF1 term is an ablation for label-frequency bias, not a substitute for an explicitly visibility-conditioned VTSE/VBF objective.",
        },
        "iq_opened": False,
        "labels_opened": False,
        "training_performed": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    if not all(checks.values()):
        raise AssertionError(f"visibility invariance audit failed: {checks}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_frequency_group_visibility_invariance/summary.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    args = parser.parse_args()
    run(args.output, args.test_v3_manifest)


if __name__ == "__main__":
    main()
