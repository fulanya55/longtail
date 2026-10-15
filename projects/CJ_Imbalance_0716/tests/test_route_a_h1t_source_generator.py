from pathlib import Path


def test_h1t_generator_has_fresh_train_only_paired_contract():
    source = (Path(__file__).resolve().parents[1] / "experiments" /
              "generate_route_a_h1t_asymmetric_power_source.m").read_text(encoding="utf-8")
    assert "baseSeed = 160000000" in source
    assert "levels = [10, 30, 50]" in source
    assert "samplesPerPattern = 128" in source
    assert "'train', 'resume'" in source
    assert "test_split_generated', false" in source
    assert "27000001,27025200" in source
    assert "receiver_noise_mapping" in source
