from __future__ import annotations

import torch

from cjlt.models.mft_crosspulse_coherence import crosspulse_mft_scores, lfm_pulse
from experiments.generate_route_a_h0n_mft_crosspulse import _reference
from experiments.run_route_a_h0n_mft_crosspulse import _load


def _mft_batch(phases: torch.Tensor) -> torch.Tensor:
    ref = lfm_pulse()
    pri = 3000
    x = torch.zeros((len(phases), 4, pri), dtype=torch.complex64)
    for p in range(4):
        x[:, p, 460 : 460 + len(ref)] += phases[:, p, None] * ref
    return torch.stack((x.real, x.imag), dim=1).reshape(len(phases), 2, -1)


def test_crosspulse_score_separates_whole_pri_phase_intervention() -> None:
    coherent = _mft_batch(torch.ones((5, 4), dtype=torch.complex64))
    scrambled = _mft_batch(torch.tensor([[1, 1j, -1, -1j]] * 5, dtype=torch.complex64))
    ref = lfm_pulse()
    coherent_score, coherent_invariant = crosspulse_mft_scores(coherent, ref)
    scrambled_score, scrambled_invariant = crosspulse_mft_scores(scrambled, ref)
    assert torch.all(coherent_score > 0.99)
    assert torch.all(scrambled_score < 0.02)
    assert torch.allclose(coherent_invariant, scrambled_invariant, atol=1e-5)


def test_crosspulse_contract_rejects_single_pri_shape() -> None:
    x = torch.zeros((2, 2, 3000), dtype=torch.float32)
    try:
        crosspulse_mft_scores(x, lfm_pulse())
    except ValueError as error:
        assert "sample length" in str(error)
    else:
        raise AssertionError("single-PRI input must be rejected")


def test_h0n_generator_reference_matches_fixed_diagnostic_reference() -> None:
    generated = _reference()
    diagnostic = lfm_pulse().numpy()
    assert generated.shape == (1000,)
    assert torch.allclose(torch.from_numpy(generated / (generated.size**0.5)), torch.from_numpy(diagnostic))


def test_h0n_loader_normalizes_complex_hdf5_payload(tmp_path) -> None:
    import h5py
    import numpy as np

    path = tmp_path / "tiny.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iq_real", data=np.ones((2, 12000), dtype=np.float32))
        handle.create_dataset("iq_imag", data=np.zeros((2, 12000), dtype=np.float32))
        handle.create_dataset("pair_seed", data=np.array([1, 2]))
        handle.create_dataset("receiver_noise_seed", data=np.array([3, 4]))
    iq, metadata = _load(path)
    assert iq.shape == (2, 2, 12000)
    assert np.allclose(np.mean(np.square(iq[:, 0]) + np.square(iq[:, 1]), axis=1), 1.0)
    assert metadata["pair_seed"].tolist() == [1, 2]
