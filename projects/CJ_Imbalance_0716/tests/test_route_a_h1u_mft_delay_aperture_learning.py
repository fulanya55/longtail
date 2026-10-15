from pathlib import Path

import numpy as np
import torch

from experiments.run_route_a_h1u_mft_delay_aperture_learning import ARMS, BANDS, H1uModel, balanced_loss, make_roles


ROOT = Path(__file__).resolve().parents[1]


def test_h1u_all_arms_are_shape_safe_and_equal_capacity():
    iq = torch.randn(3, 2, 10000)
    models = [H1uModel(arm).eval() for arm in ARMS]
    output = [model(iq) for model in models]
    assert all(value.shape == (3, 6) and torch.isfinite(value).all() for value in output)
    assert len({sum(p.numel() for p in model.parameters()) for model in models}) == 1


def test_h1u_correct_and_phase_control_features_differ_without_metadata():
    reference = H1uModel("M1").reference
    iq = torch.zeros(1, 2, 10000)
    value = torch.zeros(10000, dtype=torch.complex64)
    for delay in (2000, 4000, 6000):
        value[delay:delay + len(reference)] += reference
    iq[:, 0], iq[:, 1] = value.real, value.imag
    correct, control = H1uModel("M1").residual_features(iq), H1uModel("M2").residual_features(iq)
    assert correct.shape == control.shape == (1, 3)
    assert torch.isfinite(correct).all() and torch.isfinite(control).all()
    assert not torch.allclose(correct, control)


def test_h1u_balanced_loss_requires_and_balances_three_mft_strata():
    logits = torch.zeros(6, 6, requires_grad=True); labels = torch.zeros(6, 6); labels[2:4, 3] = 1; labels[4:, 3] = 1
    weak = torch.tensor([False, False, True, True, False, False])
    value = balanced_loss(logits, labels, weak); value.backward()
    assert torch.isfinite(value) and logits.grad is not None


def test_h1u_roles_are_disjoint_and_exact_for_a_complete_source_shape():
    # 63 combinations: absent has 80, MFT weak/strong each have 40 identities.
    combination, labels, weak = [], [], []
    for mask in range(1, 64):
        base = np.zeros(6, dtype=np.float32); base[3] = bool(mask & 8)
        for index in range(80):
            combination.append(mask); labels.append(base.copy()); weak.append(bool(base[3]) and index < 40)
    roles = make_roles(np.asarray(combination), np.asarray(labels), np.asarray(weak), 3101)
    joined = np.concatenate([roles[key] for key in ("fit", "validation", "early")])
    assert len(np.unique(joined)) == len(joined) == 5040
    assert {key: len(value) for key, value in roles.items() if key != "state"} == {"fit": 3024, "validation": 1008, "early": 1008}


def test_h1u_preregistration_binds_fixed_bands_and_test_boundary():
    plan = (ROOT / "docs/plans/route_a_h1u_mft_delay_aperture_learning_20260803.md").read_text(encoding="utf-8")
    assert BANDS == ((1600, 3000), (3500, 5000), (5400, 7000))
    for token in ("B0", "M1", "M2", "test-v3", "0.030", "0.025"):
        assert token in plan
