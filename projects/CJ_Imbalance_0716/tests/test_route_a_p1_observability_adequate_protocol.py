from __future__ import annotations

from experiments import generate_route_a_p1_observability_adequate_pairs as p1


def test_p1_has_strictly_larger_physical_apertures_without_test_role() -> None:
    assert p1.PULSE == 1_000 and p1.PRI == 5_000 and p1.PULSE_COUNT == 6
    assert p1._mft(7).shape == (p1.LENGTH,)
    assert p1._vft(7).shape == (p1.LENGTH,)
    assert p1.FORMAL_BASE_SEED != p1.SMOKE_BASE_SEED
    signals, labels, power = p1._triplet(p1.SMOKE_BASE_SEED, "VFT")
    assert len(signals) == 3 and labels.shape == (3, 6) and power.shape == (3, 6)
    assert labels[:, p1.LABELS.index("VFT")].tolist() == [0, 1, 1]
    assert power[:, p1.LABELS.index("VFT")].tolist()[1:] == [8.0, 20.0]


def test_p1_zero_delay_is_an_identity_copy() -> None:
    x = p1._reference()[:16]
    assert (p1._delay(x, 0) == x).all()
