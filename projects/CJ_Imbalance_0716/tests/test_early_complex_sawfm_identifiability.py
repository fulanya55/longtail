import numpy as np
import torch

from cjlt.models.phase_multiscale import DelayedFusionMultiScaleIQClassifier
from experiments.run_route_a_early_complex_sawfm_identifiability import (
    _complex_physics_descriptor,
    _fit_fixed_probe,
    _fixed_probe_scores,
)


def test_complex_physics_descriptor_is_fixed_and_finite():
    rng = np.random.default_rng(11)
    real = rng.normal(size=(4, 16, 3000)).astype(np.float32)
    imag = rng.normal(size=(4, 16, 3000)).astype(np.float32)
    result = _complex_physics_descriptor(real, imag)
    assert result.shape == (4, 474)
    assert np.isfinite(result).all()


def test_dfms_structure_maps_expose_stage1_without_changing_forward_shape():
    model = DelayedFusionMultiScaleIQClassifier(num_classes=6, width=16).eval()
    values = torch.randn(2, 2, 3000)
    with torch.no_grad():
        maps = model.forward_structure_maps(values)
        logits = model(values)
    assert maps["stage1_real"].shape == (2, 16, 3000)
    assert maps["stage1_imag"].shape == (2, 16, 3000)
    assert maps["stage2_real"].shape == (2, 32, 1500)
    assert logits.shape == (2, 6)


def test_fixed_probe_uses_requested_regularization_and_scores_domains():
    groups = []
    for name, offset in (("root", 0.0), ("40", 0.2)):
        values = np.asarray([[0.0 + offset], [0.1 + offset], [1.0 + offset], [1.1 + offset]], dtype=np.float32)
        groups.append({"name": name, "visibility_features": values, "targets": np.asarray([0, 0, 1, 1], dtype=np.uint8)})
    bundle = _fit_fixed_probe(groups, "P1_VisibilityStats", 0.1)
    scores = _fixed_probe_scores(bundle, groups, "P1_VisibilityStats")
    assert set(scores) == {"40"}
    assert scores["40"].shape == (4,)
