import numpy as np
import torch

from cj0724.p2w_isrj_width_oracle_ood import reconstruct_isrj_component
from cj0724.p2z_isrj_reset_consistent_separator import phase_curvature_loss, stratified_roles_10


def test_phase_curvature_loss_zero_for_global_phase_rotation() -> None:
    widths = np.asarray([1.0e-6, 12.0e-6, 40.0e-6])
    iq = reconstruct_isrj_component(widths, np.asarray([2, 3, 4]), np.asarray([0.0, 7.0e-6, 13.0e-6]))
    target = torch.from_numpy(iq)
    signal = torch.complex(target[:, 0], target[:, 1]) * torch.exp(torch.tensor(0.7j))
    estimate = torch.stack((signal.real, signal.imag), dim=1)
    assert float(phase_curvature_loss(estimate, target)) < 1.0e-8


def test_phase_curvature_loss_detects_time_shuffle() -> None:
    iq = reconstruct_isrj_component(np.asarray([5.0e-6]), np.asarray([3]), np.asarray([7.0e-6]))
    target = torch.from_numpy(iq)
    shuffled = target[:, :, torch.randperm(target.shape[-1], generator=torch.Generator().manual_seed(7))]
    assert float(phase_curvature_loss(shuffled, target)) > 0.01


def test_stratified_roles_10_preserves_each_combination() -> None:
    combinations = np.asarray([[int((key >> bit) & 1) for bit in range(6)] for key in range(1, 64)], dtype=np.uint8)
    labels = np.repeat(combinations, 10, axis=0)
    roles = stratified_roles_10(labels, 17)
    assert {name: len(indices) for name, indices in roles.items()} == {"fit": 378, "selection": 126, "calibration": 126}
    assert np.intersect1d(roles["fit"], roles["selection"]).size == 0
