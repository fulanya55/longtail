import unittest

import numpy as np
import torch

from experiments.run_route_a_mft_matched_filter_delay_profile_v3 import (
    PRI_S,
    FS_HZ,
    build_lfm_reference,
    circular_correlation_profile,
    composition_matched_view,
    summarize_profile,
)


class MatchedFilterDelayProfileTest(unittest.TestCase):
    def test_aligned_reference_recovers_known_circular_delays(self):
        device = torch.device("cpu")
        reference = build_lfm_reference(device)
        signal = 0.8 * torch.roll(reference, 430) + 1.2 * torch.roll(reference, 770)
        iq = torch.stack([signal.real, signal.imag], dim=0)[None]
        profile = circular_correlation_profile(iq, reference)[0]
        peaks = torch.topk(profile[400:1001], k=8).indices.cpu().numpy() + 400
        self.assertTrue(np.min(np.abs(peaks - 430)) <= 2)
        self.assertTrue(np.min(np.abs(peaks - 770)) <= 2)

    def test_feature_dimension_is_fixed_and_finite(self):
        length = int(round(FS_HZ * PRI_S))
        profile = torch.rand(3, length)
        features = summarize_profile(profile)
        self.assertEqual(features.shape, (3, 308))
        self.assertTrue(torch.isfinite(features).all())

    def test_composition_matching_excludes_unshared_patterns(self):
        targets = np.asarray(
            [
                [1, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 0],
            ],
            dtype=np.uint8,
        )
        weak = np.zeros_like(targets, dtype=bool)
        weak[0, 3] = True
        selected, labels, receipt = composition_matched_view(targets, weak)
        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(labels.tolist(), [1.0, 0.0])
        self.assertEqual(receipt["shared_composition_count"], 1)


if __name__ == "__main__":
    unittest.main()

