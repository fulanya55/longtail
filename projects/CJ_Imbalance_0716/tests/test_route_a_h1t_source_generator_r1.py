from pathlib import Path


def test_h1t_r1_preserves_physics_and_replaces_only_identity_range():
    source = (Path(__file__).resolve().parents[1] / "experiments" /
              "generate_route_a_h1t_asymmetric_power_source_r1.m").read_text(encoding="utf-8")
    assert "baseSeed = 164000000" in source
    assert "levels = [10, 30, 50]" in source
    assert "samplesPerPattern = 128" in source
    assert "160000001,160008064" in source
    assert "test_split_generated',false" in source
