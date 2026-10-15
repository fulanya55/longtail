import json
from pathlib import Path

import numpy as np
import torch

from cjlt.models.mft_matched_filter_support import fixed_phase_permutation, lfm_reference
from experiments.run_route_a_h0u_mft_delay_aperture import BANDS, separated_peak_score


ROOT = Path(__file__).resolve().parents[1]


def _replica_record(delays: tuple[int, int, int]) -> torch.Tensor:
    reference = lfm_reference()
    signal = torch.zeros((1, 2, 10_000), dtype=torch.float32)
    value = torch.zeros(10_000, dtype=torch.complex64)
    for delay in delays:
        value[delay:delay + reference.numel()] += reference
    signal[:, 0] = value.real
    signal[:, 1] = value.imag
    return signal


def test_h0u_candidate_responds_to_the_three_separated_lfm_replicas():
    iq = _replica_record((2_000, 4_000, 6_000))
    reference = lfm_reference()
    candidate = separated_peak_score(iq, reference)
    permuted = separated_peak_score(iq, fixed_phase_permutation(reference))
    assert candidate.shape == permuted.shape == (1,)
    assert torch.isfinite(candidate).all() and torch.isfinite(permuted).all()
    assert candidate.item() > permuted.item()


def test_h0u_delay_bands_cover_contract_and_remain_disjoint():
    assert BANDS == ((1600, 3000), (3500, 5000), (5400, 7000))
    assert all(left < right for left, right in BANDS)
    assert all(BANDS[index][1] < BANDS[index + 1][0] for index in range(len(BANDS) - 1))
    for delay in (1800, 2200, 3800, 4200, 5800, 6200):
        assert any(start <= delay < stop for start, stop in BANDS)


def test_h0u_minimum_spacing_is_computed_within_each_identity():
    delays = np.array([[2035, 3820, 5863], [2121, 3929, 6171]], dtype=float)
    minimum = np.min(np.diff(np.sort(delays, axis=1), axis=1))
    assert minimum == 1785


def test_h0u_plan_and_generator_have_fixed_new_protocol_contract():
    plan = (ROOT / "docs/plans/route_a_h0u_mft_delay_aperture_identifiability_20260803.md").read_text(encoding="utf-8")
    generator = (ROOT / "experiments/generate_route_a_h0u_mft_delay_aperture_composite.m").read_text(encoding="utf-8")
    for text in (plan, generator):
        assert "3900000000" in text
        assert "1 ms" in text or "1e-3" in text
        assert "test-v3" in text or "test_v3" in text
    assert "mft_minimum_spacing_samples" in generator
    assert "h0u_mft_delay_aperture_20260803_r1" in generator


def test_h0u_pre_generation_selection_is_identity_isolated():
    path = ROOT / "outputs/tapc/route_a_h0u_mft_delay_aperture_generation/pre_generation_identity_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sample_seed_base"] == 3_900_000_000
    assert payload["identity_overlap_count"] == 0
    assert payload["test_files_opened"] is False
