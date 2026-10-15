from __future__ import annotations

import numpy as np

from experiments.analyze_route_a_p1_p2_isrj_h0_failure import _quantiles


def test_failure_diagnosis_quantiles_are_ordered_and_finite() -> None:
    value = _quantiles(np.asarray((1.0, 2.0, 3.0, 4.0)))
    assert value["q25"] <= value["median"] <= value["q75"]
    assert value["mean"] == 2.5
