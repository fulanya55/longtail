from pathlib import Path

from experiments.run_route_a_h3_frozen_unified_external_confirmation import SEEDS, _verify_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_h3_uses_exactly_the_five_frozen_candidate_seeds():
    assert SEEDS == (2039, 2040, 2041, 2042, 2043)


def test_h3_frozen_manifest_artifacts_verify_before_external_source_opening():
    import json
    manifest = json.loads((ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json").read_text(encoding="utf-8"))
    _verify_manifest(manifest)
