from __future__ import annotations

from experiments import generate_route_a_p2_full_target_pairs as p2


def test_p2_uses_new_seed_blocks_and_high_replica_isrj() -> None:
    assert p2.FORMAL_BASE_SEED != p2.SMOKE_BASE_SEED
    assert p2.FORMAL_BASE_SEED not in (4_200_000_000, 4_250_000_000)
    assert p2._isrj_high_replica(5).shape == (p2.LENGTH,)
    signals, labels, power = p2._triplet(p2.SMOKE_BASE_SEED, "ISRJ")
    assert len(signals) == 3 and labels[:, 5].tolist() == [0, 1, 1]
    assert power[:, 5].tolist()[1:] == [8.0, 20.0]
