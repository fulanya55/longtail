import numpy as np
import torch

from cj0724.p3c_mft_sparse_deconvolution import estimate_sparse_count, reconstruct_mft_component
from cj0724.p3or_mft_component_separator import waveform_objective


def test_p3or_exact_teacher_preserves_sparse_count() -> None:
    delays = np.full((2, 8), np.nan)
    amplitudes = np.full((2, 8), np.nan)
    delays[0, :2] = [300, 420]
    delays[1, :5] = [300, 350, 400, 450, 500]
    amplitudes[:, :] = np.nan
    amplitudes[0, :2] = 1.0
    amplitudes[1, :5] = 1.0
    teacher = reconstruct_mft_component(delays / 10.0e6, amplitudes)
    assert estimate_sparse_count(teacher)["prediction"].tolist() == [2.0, 5.0]


def test_p3or_waveform_objective_uses_normalized_component() -> None:
    class Separator(torch.nn.Module):
        def forward_components(self, iq, _):
            return {"normalized_estimate": iq}

    iq = torch.randn(2, 3, 2, 3000)
    target = iq[:, 0]
    total, parts, estimate = waveform_objective(Separator(), iq, target)
    assert torch.isfinite(total)
    assert set(parts) == {"cosine_loss", "time_l1", "spectral"}
    assert estimate.shape == (6, 2, 3000)


def test_p3or_dataset_reads_each_power_at_the_requested_sample_index() -> None:
    class View:
        def __init__(self, expected_index):
            self.expected_index = expected_index

        def __getitem__(self, index):
            assert index == self.expected_index
            return {"sample_index": index, "iq": torch.zeros(2, 3000)}

    from cj0724.p3or_mft_component_separator import MFTComponentDataset

    dataset = object.__new__(MFTComponentDataset)
    dataset.entries = [("ID_control", 7, np.zeros((2, 3000), dtype=np.float32), 3.0, 2, 1007)]
    dataset.views = {"ID_control": {25: View(7), 35: View(7), 50: View(7)}}
    item = dataset[0]
    assert item["iq"].shape == (3, 2, 3000)
    assert item["receiver_noise_seed"] == 1007
