import unittest

import torch

from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import (
    SCORE_NAMES,
    morphology_scores,
    nms_peaks,
)


class PulseCompressionMorphologyTest(unittest.TestCase):
    def test_nms_enforces_peak_separation(self):
        values = torch.zeros(1, 80)
        values[0, 10] = 9.0
        values[0, 11] = 8.0
        values[0, 40] = 7.0
        peaks = nms_peaks(values, count=2, radius=8)
        self.assertEqual(peaks.tolist(), [[9.0, 7.0]])

    def test_scores_are_fixed_and_finite(self):
        profile = torch.rand(4, 3000) + 0.01
        scores = morphology_scores(profile)
        self.assertEqual(scores.shape, (4, len(SCORE_NAMES)))
        self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()

