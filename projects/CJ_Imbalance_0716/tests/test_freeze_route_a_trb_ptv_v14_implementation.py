from pathlib import Path

from experiments.freeze_route_a_trb_ptv_v14_implementation import sha256


def test_sha256_is_repeatable_for_the_preregistered_manifest():
    path = Path("docs/plans/route_a_trb_ptv_implementation_manifest_20260801.json")
    assert len(sha256(path)) == 64
    assert sha256(path) == sha256(path)
