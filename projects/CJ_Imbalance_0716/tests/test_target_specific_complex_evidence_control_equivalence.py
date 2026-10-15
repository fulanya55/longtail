"""Contract test for the TS-CLE cyclic-source capacity control.

The three local branches are architecturally identical.  When the control
state is the corresponding cyclic permutation of a candidate state, the two
models must produce the same logits, and one local-only gradient step must
preserve that equivalence.  This prevents a branch-index random-initialisation
effect from being misreported as source-specific evidence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cjlt.models.target_specific_complex_evidence import TargetSpecificComplexEvidenceClassifier


_LOCAL_PREFIXES = ("local_filters.", "local_mixers.", "evidence_heads.", "local_classifiers.")
_ROUTE = (1, 2, 0)  # target position -> control local-branch position


def _permuted_control_state(candidate: TargetSpecificComplexEvidenceClassifier):
    """Return the control state in which control branch ``route[p]`` is candidate ``p``."""
    source = candidate.state_dict()
    result = {key: value.detach().clone() for key, value in source.items()}
    for prefix in _LOCAL_PREFIXES:
        for target_position, control_branch in enumerate(_ROUTE):
            src = f"{prefix}{target_position}."
            dst = f"{prefix}{control_branch}."
            for key, value in source.items():
                if key.startswith(src):
                    result[dst + key[len(src):]] = value.detach().clone()
    return result


def _local_only_step(model, x, y):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(_LOCAL_PREFIXES) or name == "local_scale"
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(model(x)[:, [3, 4, 5]], y[:, [3, 4, 5]])
    loss.backward()
    optimizer.step()


def test_cyclic_source_control_is_permutation_equivalent_when_initialized_matched():
    torch.manual_seed(20802026)
    candidate = TargetSpecificComplexEvidenceClassifier(channels=4, topk=8, source_control=False).eval()
    control = TargetSpecificComplexEvidenceClassifier(channels=4, topk=8, source_control=True).eval()
    control.load_state_dict(_permuted_control_state(candidate))
    x = torch.randn(5, 2, 375)
    with torch.no_grad():
        assert torch.equal(candidate(x), control(x))


def test_cyclic_source_control_equivalence_survives_local_only_target_bce_step():
    torch.manual_seed(20802027)
    candidate = TargetSpecificComplexEvidenceClassifier(channels=4, topk=8, source_control=False)
    control = TargetSpecificComplexEvidenceClassifier(channels=4, topk=8, source_control=True)
    control.load_state_dict(_permuted_control_state(candidate))
    candidate.train()
    control.train()
    x = torch.randn(6, 2, 375)
    y = torch.randint(0, 2, (6, 6), dtype=torch.float32)
    _local_only_step(candidate, x, y)
    _local_only_step(control, x, y)
    candidate.eval()
    control.eval()
    with torch.no_grad():
        assert torch.equal(candidate(x), control(x))
