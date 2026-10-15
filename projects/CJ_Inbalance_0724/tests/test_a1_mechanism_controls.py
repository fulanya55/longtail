import numpy as np
import torch

from cj0724.a1_mechanism_controls import _filter_bank


def test_filter_bank_shapes_and_controls_differ():
    frequencies = np.arange(-1000.0, 1000.1, 500.0, dtype=np.float32)
    correct = _filter_bank(frequencies, 100, 5.0e10, torch.device("cpu"))
    wrong = _filter_bank(frequencies, 100, 0.0, torch.device("cpu"))
    random = _filter_bank(frequencies, 100, 0.0, torch.device("cpu"), 7)
    assert correct.shape == (10, 2, 100)
    assert not torch.allclose(correct, wrong)
    assert not torch.allclose(correct, random)
