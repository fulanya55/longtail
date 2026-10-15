import numpy as np

from cj0724.p3c_mft_sparse_deconvolution import estimate_sparse_count, reconstruct_mft_component


def test_sparse_deconvolution_recovers_hand_constructed_counts() -> None:
    delays = np.full((3, 8), np.nan)
    amplitudes = np.full((3, 8), np.nan)
    for row, count in enumerate((1, 4, 8)):
        delays[row, :count] = (300 + np.arange(count)) / 10.0e6
        amplitudes[row, :count] = np.linspace(0.5, 1.5, count)
    iq = reconstruct_mft_component(delays, amplitudes)
    result = estimate_sparse_count(iq)
    assert result["prediction"].tolist() == [1.0, 4.0, 8.0]


def test_sparse_deconvolution_shape_validation() -> None:
    try:
        estimate_sparse_count(np.zeros((2, 3000), dtype=np.float32))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected shape validation error")
