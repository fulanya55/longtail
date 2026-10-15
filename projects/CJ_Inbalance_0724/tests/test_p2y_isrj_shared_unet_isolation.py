import numpy as np

from cj0724.p2y_isrj_shared_unet_isolation import evaluate_prediction, waveform_cosine


def test_waveform_cosine_identity_is_one():
    values = np.random.default_rng(3).normal(size=(4, 2, 3000)).astype(np.float32)
    result = waveform_cosine(values, values)
    assert np.allclose(result, 1.0, atol=1.0e-6)


def test_evaluate_prediction_splits_compound_rows():
    target = np.asarray([1.0, 2.0, 3.0, 4.0]) * 1.0e-6
    prediction = target.copy()
    combination_size = np.asarray([1, 2, 3, 1])
    result = evaluate_prediction(prediction, target, combination_size)
    assert result["overall"]["count"] == 4
    assert result["compound"]["count"] == 2
    assert result["overall"]["mae_s"] == 0.0
