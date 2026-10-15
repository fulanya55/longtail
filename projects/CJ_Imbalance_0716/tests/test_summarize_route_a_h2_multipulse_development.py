from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h2_summary", ROOT / "experiments" / "summarize_route_a_h2_multipulse_development.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_weak_vft_metrics_are_recomputed_from_saved_raw_predictions(tmp_path: Path) -> None:
    run_root = tmp_path / "C3-mp_seed7"
    run_root.mkdir()
    labels = np.zeros((4, 6), dtype=bool)
    labels[[0, 2], 4] = True
    power = np.full((4, 6), np.nan, dtype=np.float64)
    power[[0, 2], 4] = 8.0
    probability = np.full((4, 6), 0.5, dtype=np.float64)
    probability[:, 4] = np.asarray([0.9, 0.8, 0.7, 0.1])
    np.savez_compressed(
        run_root / "development_predictions.npz",
        probabilities=probability,
        labels=labels,
        component_jnr_db=power,
    )
    result = {
        "mode": "C3-mp",
        "seed": 7,
        "metrics": {
            "macro_aupr": 0.6,
            "vft_weak_auroc": 0.75,
            "per_label": [{"aupr": 0.1 + index / 10.0} for index in range(6)],
        },
        "threshold_metrics": {"per_label": [{"fpr": 0.1, "weak_recall": 0.2} for _ in range(6)]},
    }
    derived = MODULE._derived_prediction_metrics(tmp_path, result)
    assert derived["vft_weak_auroc"] == 0.75
    assert np.isclose(derived["vft_weak_aupr"], 5.0 / 6.0)
    assert np.isclose(derived["non_vft_macro_aupr"], np.mean([0.1, 0.2, 0.3, 0.4, 0.6]))
    assert derived["vft_weak_recall"] == 0.2
    assert derived["sawfm_weak_recall"] == 0.2
