import numpy as np
import torch

from experiments.run_route_a_vft_isrj_raw_source_diagnostic import _hard_source_view, _phase_scramble, raw_source_features


def test_source_features_keep_fixed_dimension_and_phase_control_changes_them():
    torch.manual_seed(7)
    signal = torch.randn(3, 2, 3000)
    raw = raw_source_features(signal)
    control = raw_source_features(_phase_scramble(signal))
    assert raw.shape == control.shape == (3, 32)
    assert torch.isfinite(raw).all() and torch.isfinite(control).all()
    assert not torch.allclose(raw, control)


def test_hard_view_matches_context_without_using_isrj_as_a_shortcut():
    targets = np.array([
        [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1],
    ], dtype=np.uint8)
    weak = np.zeros_like(targets, dtype=bool); weak[:3, 4] = True
    selected, labels, receipt = _hard_source_view(targets, weak)
    assert selected.tolist() == [True, True, False, True, True, False]
    assert labels.tolist() == [1, 1, 0, 0]
    assert receipt["weak_vft_positive_count"] == 2
    assert receipt["vft_absent_isrj_present_negative_count"] == 2
