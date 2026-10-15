import numpy as np
import torch

from experiments.run_route_a_v19_hardnegative_vft_morphology_probe import MorphologyProbe, _metric, _weak_metric


def test_morphology_probe_shape_and_parameter_count():
    model = MorphologyProbe(268, 64, .1)
    assert model(torch.zeros(3, 268)).shape == (3,)
    assert sum(item.numel() for item in model.parameters()) == 17281


def test_heldout_metric_and_weak_slice_use_real_labels():
    labels = np.zeros((48, 6), dtype=np.uint8)
    labels[:24, 4] = 1
    labels[:, 5] = 1
    weak = np.zeros_like(labels, dtype=bool)
    weak[:20, 4] = True
    ids = np.arange(48)
    scores = np.linspace(.9, .1, 48)
    assert _metric(labels[:, 4], scores)["auroc"] == 1.0
    assert _weak_metric(labels, weak, ids, scores)["positive_count"] == 20
