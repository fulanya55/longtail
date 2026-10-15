import numpy as np
import pytest
import torch

from experiments.run_route_a_h1t_asymmetric_visibility_transfer import (
    binary_posterior_kl,
    composition_permute_teacher,
)


def test_target_local_kl_is_zero_for_identical_logits_and_masked_entries():
    logits = torch.tensor([[0.2, -0.3, 0.7], [0.5, 0.4, -0.2]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    assert binary_posterior_kl(logits, logits, mask, 2.0).item() == pytest.approx(0.0, abs=1e-7)
    assert binary_posterior_kl(logits, logits + 1.0, torch.zeros_like(mask), 2.0).item() == pytest.approx(0.0)


def test_composition_permutation_preserves_target_teacher_marginals():
    labels = np.zeros((4, 6), dtype=np.uint8); labels[:, 3] = 1
    weak = np.zeros_like(labels, dtype=bool); weak[:, 3] = True
    teacher = np.arange(12, dtype=np.float32).reshape(4, 3)
    permuted, detail = composition_permute_teacher(teacher, labels, weak)
    assert np.array_equal(np.sort(permuted[:, 0]), np.sort(teacher[:, 0]))
    assert not np.array_equal(permuted[:, 0], teacher[:, 0])
    assert detail["MFT"]["score_marginal_exact"]
