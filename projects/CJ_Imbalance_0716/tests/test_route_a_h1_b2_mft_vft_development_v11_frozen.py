from pathlib import Path

from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import VARIANTS


def test_frozen_variant_contract():
    assert VARIANTS[0] == "Q0_H1"
    assert VARIANTS[4] == "Q4_VFTAlignedRidgeFusion"
    assert len(VARIANTS) == 6


def test_test_v3_manifest_path_is_separate():
    path = Path("frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    assert "test_v3" in str(path)
