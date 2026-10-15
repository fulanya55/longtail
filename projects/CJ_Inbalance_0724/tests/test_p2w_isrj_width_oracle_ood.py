import numpy as np

from cj0724.p2w_isrj_width_oracle_ood import estimate_width_acf, metrics, reconstruct_isrj_component


def test_reconstruction_and_estimator_shapes() -> None:
    widths = np.asarray([5e-6, 12e-6, 20e-6])
    replicas = np.asarray([2, 3, 4])
    arrivals = np.asarray([10e-6, 20e-6, 30e-6])
    iq = reconstruct_isrj_component(widths, replicas, arrivals)
    result = estimate_width_acf(iq)
    assert iq.shape == (3, 2, 3000)
    assert result["prediction"].shape == (3,)
    assert np.isfinite(result["prediction"]).all()


def test_metrics_perfect_prediction() -> None:
    target = np.asarray([5e-6, 10e-6, 20e-6])
    result = metrics(target, target)
    assert result["mae_us"] == 0.0
    assert result["mae_reduction_vs_constant"] == 1.0
    assert result["pearson"] > 0.999
