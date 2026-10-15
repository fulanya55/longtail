from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/generate_route_a_dual_axis_sawfm_development_v1.m"


def test_generator_is_new_identity_train_only_and_sawfm_scoped():
    text = SOURCE.read_text(encoding="utf-8")
    assert "baseSeed = 26080701" in text
    assert "'train'" in text
    assert "{'SawFM'}" in text
    assert "{'FMB', 'SinFM', 'MFT', 'VFT', 'ISRJ'}" in text
    assert "test_views_generated = false" in text
    assert "paired_seed_verified = result.paired_seed_verified" in text


def test_generator_freezes_visibility_levels_and_no_overwrite():
    text = SOURCE.read_text(encoding="utf-8")
    assert "requiredLevels = [20, 30]" in text
    assert "existingPolicy = 'error'" in text
    assert "low_visibility_JNR_dB = 20" in text
    assert "high_visibility_JNR_dB = 30" in text
    assert "test_v3_used = false" in text
