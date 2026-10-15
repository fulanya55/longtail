from pathlib import Path

from cj0724.p3ol_data_audit import _role_paths


def test_p3ol_role_paths_are_absolute_d_drive_safe():
    paths = _role_paths(Path("D:/CJ0724_rechunked_data"), "p3ol_component_conditioned_train_seed1530000000")
    assert set(paths) == {"ID_control", "low_count", "close_spacing", "ultra_close", "weak_target"}
    assert all(path.is_absolute() for files in paths.values() for path in files.values())
